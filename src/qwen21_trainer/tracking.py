"""Opt-in W&B tracking. Importing this module never imports W&B.

The tracker owns one run in the training process, not in CPU data workers.
SDK 0.30 settings disable automatic console/code/git/system collection. Explicit
sanitized configuration and generated samples are separately controllable;
arbitrary metric fields, dataset images and model weights are never collected.
"""

from __future__ import annotations

import csv
import importlib
import io
import math
import os
import re
import stat
import tempfile
import uuid
from collections.abc import Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

_TRAIN_NUMBERS = {
    "seed",
    "rank",
    "alpha",
    "learning_rate",
    "num_warmup",
    "weight_decay",
    "max_steps",
    "gradient_accumulation_steps",
    "num_workers",
    "prefetch_factor",
    "save_every",
}
_TRAIN_BOOLS = {"gradient_checkpointing", "checkpointing_offload"}
_PLAN_NUMBERS = {
    "num_images",
    "weighted_images_per_pass",
    "optimizer_updates",
    "effective_batch_size",
    "rank",
    "alpha",
    "lora_scale",
    "schedule_mu",
}
_ENUMS = {
    "lr_scheduler": {"diffsynth", "constant", "constant_with_warmup", "cosine"},
    "loss_type": {"mse", "wavelet"},
    "loss_weighting": {"diffsynth", "none"},
    "optimizer": {"adamw", "adopt_adv"},
    "latent_mode": {"posterior_mean"},
}
_OPT_NUMBERS = {
    "eps",
    "nesterov_coef",
    "beta2_min",
    "ema_alpha",
    "tiny_spike",
    "centered_wd",
    "k_warmup_steps",
    "k_logging",
}
_OPT_BOOLS = {
    "amsgrad",
    "maximize",
    "fisher_wd",
    "cautious_wd",
    "use_atan2",
    "stochastic_rounding",
    "nesterov",
    "kourkoutas_beta",
    "spectral_normalization",
    "factored_2nd",
    "nnmf_factor",
    "vector_reshape",
    "compiled_optimizer",
}
_OPT_ENUMS = {
    "orthogonal_gradient": {"disabled", "flattened", "iterative"},
    "state_precision": {"auto", "fp32", "factored", "bf16_sr", "fp16", "int8_sr"},
    "centered_wd_mode": {"full", "float8", "int8", "int4"},
}
_METRICS = (
    "step",
    "loss",
    "learning_rate",
    "elapsed_seconds",
    "images_seen",
    "data_wait_seconds",
)


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _select(source, numbers=(), bools=(), enums=None):
    """Explicit keys AND types: a path hidden under a numeric key is not sent."""
    if not isinstance(source, Mapping):
        return {}
    result = {k: source[k] for k in numbers if k in source and _finite_number(source[k])}
    result.update({k: source[k] for k in bools if k in source and type(source[k]) is bool})
    for key, allowed in (enums or {}).items():
        value = source.get(key)
        if type(value) is str and value in allowed:
            result[key] = value
    return result


def _hyperparameters(cfg, plan):
    training = getattr(cfg, "training", {})
    result = _select(training, _TRAIN_NUMBERS, _TRAIN_BOOLS, _ENUMS)
    result.update(_select(plan, _PLAN_NUMBERS, enums=_ENUMS))
    params = plan.get("optimizer_params", training.get("optimizer_params", {}))
    selected = _select(params, _OPT_NUMBERS, _OPT_BOOLS, _OPT_ENUMS)
    betas = params.get("betas") if isinstance(params, Mapping) else None
    if isinstance(betas, (list, tuple)) and len(betas) == 2 and all(map(_finite_number, betas)):
        selected["betas"] = list(betas)
    if selected:
        result["optimizer_params"] = selected
    prefetch = _select(
        plan.get("cpu_prefetch", {}),
        {"num_workers", "prefetch_factor", "max_inflight_tasks"},
        {"pin_memory"},
        {"backend": {"spawn", "synchronous"}},
    )
    if prefetch:
        result["cpu_prefetch"] = prefetch
    return result


def _public_url(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
        ):
            return None
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    except ValueError:
        return None


def _sample_image(output_dir, sample, step, wandb, sanitizer):
    """Read one explicit PNG under samples, using no-follow directory handles.

    Re-encode decoded pixels, not the original file (PNG text metadata and source
    filenames are not approved payloads). No directory scanning or data access.
    """
    from PIL import Image

    if not isinstance(sample, Mapping):
        raise ValueError("W&B sample 必須是明確的生成圖片紀錄。")
    if type(sample.get("step")) is not int or sample["step"] != step:
        raise ValueError("W&B sample 與訓練 step 不一致。")
    if not isinstance(sample.get("prompt"), str):
        raise ValueError("W&B sample prompt 必須是字串。")
    for key in ("seed", "index"):
        if type(sample.get(key)) is not int or sample[key] < 0:
            raise ValueError("W&B sample seed/index 必須是非負整數。")
    root = Path(output_dir).resolve()
    path = Path(sample["path"])
    if ".." in path.parts:
        raise ValueError("W&B sample 路徑不可離開 samples 目錄。")
    if not path.is_absolute():
        # Relative paths are interpreted against the training working directory,
        # just like Path.save() in the sample producer.
        path = Path.cwd() / path
    relative = path.relative_to(root / "samples")
    if not relative.parts or path.suffix.lower() != ".png":
        raise ValueError("W&B sample 必須是 samples 內的 PNG。")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptors = []
    try:
        descriptors.append(os.open(root, directory_flags))
        for part in ("samples", *relative.parts[:-1]):
            descriptors.append(os.open(part, directory_flags, dir_fd=descriptors[-1]))
        descriptor = os.open(
            relative.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=descriptors[-1],
        )
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("W&B sample 必須是正常檔案。")
            with Image.open(stream) as original:
                if original.format != "PNG" or getattr(original, "n_frames", 1) != 1:
                    raise ValueError("W&B sample 必須是單張 PNG。")
                original.load()
                # Image.new + paste deliberately omits all original metadata.
                pixels = Image.new("RGBA" if "A" in original.getbands() else "RGB", original.size)
                pixels.paste(original)
        try:
            caption = sanitizer.text(
                f"step={step} seed={sample['seed']} index={sample['index']}\n{sample['prompt']}"
            )
            return wandb.Image(pixels, caption=caption, file_type="png")
        finally:
            pixels.close()
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


_TRAINING_LOG_FIELDS = (
    "step",
    "loss",
    "loss_average",
    "learning_rate",
    "elapsed_seconds",
    "images_seen",
    "data_wait_seconds",
)
_DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


@contextmanager
def _nofollow_directory(path):
    """Open every directory component without following symlinks or '..'."""
    path = Path(path)
    if ".." in path.parts:
        raise ValueError("W&B training-log 路徑不合法。")
    path = path.absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _training_log_snapshot(output_dir):
    """Rebuild numeric CSV only, never upload source text, console or logs.

    The caller must close/flush its metrics writer before exiting the tracker.
    Missing metrics (e.g. early setup failure) do not manufacture an artifact.
    Validate the entire source before touching a snapshot; only canonical numeric
    values and fixed allowlisted column names can reach the SDK staging area.
    """
    with _nofollow_directory(output_dir) as root:
        try:
            descriptor = os.open(
                "metrics.csv", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root
            )
        except FileNotFoundError:
            return None
        with os.fdopen(descriptor, "r", encoding="utf-8", newline="") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("W&B training-log 必須是正常檔案。")
            reader = csv.reader(stream, strict=True)
            fields = next(reader, [])
            if (
                not fields
                or "step" not in fields
                or len(fields) != len(set(fields))
                or not set(fields).issubset(_TRAINING_LOG_FIELDS)
            ):
                raise ValueError("W&B training-log 欄位不合法。")
            columns = [field for field in _TRAINING_LOG_FIELDS if field in fields]
            buffer = io.StringIO(newline="")
            writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            count = 0
            for cells in reader:
                if len(cells) != len(fields):
                    raise ValueError("W&B training-log 資料列不完整。")
                clean = {}
                for field, text in zip(fields, cells):
                    if not _DECIMAL.fullmatch(text):
                        raise ValueError("W&B training-log 必須只含有限數值。")
                    if field in {"step", "images_seen"}:
                        if not re.fullmatch(r"[0-9]+", text):
                            raise ValueError("W&B training-log 計數必須是非負整數。")
                        value = int(text)
                        if field == "step" and value == 0:
                            raise ValueError("W&B training-log step 必須是正整數。")
                    else:
                        value = float(text)
                    if not _finite_number(value):
                        raise ValueError("W&B training-log 必須只含有限數值。")
                    clean[field] = value
                writer.writerow(clean)
                count += 1
            content = buffer.getvalue().encode("utf-8")
        try:
            os.mkdir("wandb-logs", mode=0o700, dir_fd=root)
        except FileExistsError:
            pass
        destination = os.open(
            "wandb-logs", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root
        )
        temporary = f".metrics-{uuid.uuid4().hex}.tmp"
        try:
            try:
                existing = os.stat("metrics.csv", dir_fd=destination, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if not stat.S_ISREG(existing.st_mode):
                    raise ValueError("W&B training-log snapshot 必須是正常檔案。")
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=destination,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, "metrics.csv", src_dir_fd=destination, dst_dir_fd=destination)
        finally:
            try:
                os.unlink(temporary, dir_fd=destination)
            except FileNotFoundError:
                pass
            os.close(destination)
    return Path(output_dir).absolute() / "wandb-logs" / "metrics.csv", count, columns


class TrainingTracker:
    """``with TrainingTracker(cfg, plan, output_dir) as tracker: tracker.log(row)``.

    Disabled tracking is a genuine no-op (no SDK import, files, or network).
    Enabled tracking fails closed; it never silently falls back to disabled mode.
    Call only from the single training process; scoped SDK environment/stdout
    changes are not intended for concurrent tracker calls from multiple threads.
    """

    def __init__(self, cfg, plan, output_dir):
        self._cfg = cfg
        self._plan = plan
        self._output_dir = Path(output_dir)
        options = getattr(cfg, "wandb", {})
        self._enabled = options.get("enabled", False)
        self._mode = options.get("mode", "online")
        self._project = options.get("project", "qwen-image-21-trainer")
        self._name = options.get("name") or self._output_dir.name
        self._entity = options.get("entity") or None
        self._log_every = options.get("log_every", 1)
        # Old namespace fixtures deliberately retain the scalar-only contract.
        self._log_config = options.get("log_config", False)
        self._log_samples = options.get("log_samples", False)
        self._log_training_log = options.get("log_training_log", False)
        self._wandb = None
        self._sanitizer = None
        self._run = None
        self._entered = False
        self._info = {
            "enabled": self._enabled,
            "mode": self._mode,
            "project": self._project,
            "name": self._name,
            "entity": self._entity,
            "run_id": None,
            "run_url": None,
        }

    @property
    def info(self):
        """JSON-safe public metadata; never contains SDK settings or credentials."""
        return dict(self._info)

    @contextmanager
    def _sdk_scope(self):
        # W&B's core service uses TMPDIR even if init(dir=...) is supplied.
        # Scope and restore overrides, including Python's cached tempfile root.
        # Do not read, copy, or replace the API key.
        root = self._output_dir.resolve() / "wandb"
        locations = {
            "WANDB_DIR": self._output_dir.resolve(),
            "WANDB_CACHE_DIR": root / "cache",
            "WANDB_CONFIG_DIR": root / "config",
            "WANDB_DATA_DIR": root / "data",
            "WANDB_ARTIFACT_DIR": root / "artifacts",
            "TMPDIR": root / "tmp",
        }
        for directory in locations.values():
            directory.mkdir(parents=True, exist_ok=True)
        overrides = {key: str(value) for key, value in locations.items()}
        overrides.update({"WANDB_ERROR_REPORTING": "false", "WANDB_SILENT": "true"})
        old_env = {key: os.environ.get(key) for key in overrides}
        old_tempdir = tempfile.tempdir
        try:
            os.environ.update(overrides)
            tempfile.tempdir = overrides["TMPDIR"]
            # Some SDK failure paths print exception text despite console='off'.
            with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
                yield
        finally:
            tempfile.tempdir = old_tempdir
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def __enter__(self):
        if self._entered:
            raise RuntimeError("同一個 TrainingTracker 不可重複進入。")
        self._entered = True
        if not self._enabled:
            return self
        if self._mode not in {"online", "offline"}:
            raise ValueError("W&B mode 只支援 online 或 offline。")
        if type(self._log_every) is not int or self._log_every < 1:
            raise ValueError("W&B log_every 必須是正整數。")
        if self._mode == "online" and not os.environ.get("WANDB_API_KEY", "").strip():
            raise RuntimeError("W&B online 模式需要環境變數 WANDB_API_KEY；不會互動式登入。")
        try:
            with self._sdk_scope():
                from .config_snapshot import (
                    ConfigSanitizer,
                    configuration_snapshot,
                    write_configuration_snapshot,
                )

                self._sanitizer = ConfigSanitizer()
                for field in ("project", "name", "entity"):
                    value = getattr(self, f"_{field}")
                    if value is not None:
                        value = self._sanitizer.text(value)
                        setattr(self, f"_{field}", value)
                        self._info[field] = value
                wandb = importlib.import_module("wandb")
                self._wandb = wandb
                run_config = _hyperparameters(self._cfg, self._plan)
                if self._log_config:
                    effective, source, metadata = configuration_snapshot(self._cfg, self._sanitizer)
                    run_config["resolved_config"] = effective
                    run_config["config_snapshot"] = metadata
                    if source is not None:
                        run_config["source_config"] = source
                settings = wandb.Settings(
                    console="off",
                    silent=True,
                    quiet=True,
                    disable_code=True,
                    save_code=False,
                    disable_git=True,
                    disable_job_creation=True,
                    x_disable_meta=True,
                    x_disable_stats=True,
                    x_disable_machine_info=True,
                    x_disable_viewer=True,
                    x_save_requirements=False,
                    symlink=False,
                    sagemaker_disable=True,
                    config_paths=(),
                    program="qwen21-trainer",
                    program_abspath="",
                    program_relpath="",
                    host="",
                    username="",
                    email="",
                )
                self._run = wandb.init(
                    project=self._project,
                    entity=self._entity,
                    name=self._name,
                    mode=self._mode,
                    dir=str(self._output_dir.resolve()),
                    config=run_config,
                    settings=settings,
                    save_code=False,
                    sync_tensorboard=False,
                    monitor_gym=False,
                    reinit="create_new",
                    resume="never",
                )
                if self._run is None:
                    raise RuntimeError("missing run")
                if self._run.settings.mode != self._mode:
                    raise RuntimeError("SDK changed the requested tracking mode")
                self._run.define_metric("train/step")
                self._run.define_metric("train/*", step_metric="train/step")
                self._run.define_metric("loss/*", step_metric="train/step")
                self._run.define_metric("lr/*", step_metric="train/step")
                if self._log_samples:
                    self._run.define_metric("samples/*", step_metric="train/step")
                    self._run.define_metric("sample_*", step_metric="train/step")
                if self._log_config:
                    paths = write_configuration_snapshot(
                        self._output_dir, effective, source, metadata
                    )
                    artifact = wandb.Artifact(
                        name=f"config-{self._run.id}",
                        type="training-config",
                        metadata=metadata,
                        description="Sanitized, reserialized configuration; source comments omitted.",
                    )
                    for path in paths:
                        artifact.add_file(str(path), name=path.name)
                    self._run.log_artifact(artifact)
                self._info["run_id"] = str(self._run.id)
                # Offline SDK URL access warns; there is deliberately no remote URL.
                if self._mode == "online":
                    url = _public_url(self._run.url)
                    self._info["run_url"] = self._sanitizer.text(url) if url else None
        except BaseException as exc:
            self._finish(1, suppress=True)
            if not isinstance(exc, Exception):
                raise
            raise RuntimeError("W&B 初始化失敗；已停止訓練，未切換為靜默停用。") from None
        return self

    def log(self, row, *, samples=None):
        if not self._enabled:
            return
        if self._run is None:
            raise RuntimeError("W&B 尚未啟動或已結束，無法記錄訓練指標。")
        step = row.get("step")
        if type(step) is not int or step < 1:
            raise ValueError("W&B 訓練 step 必須是正整數。")
        # Opt-out must not even inspect supplied image paths or iterate samples.
        selected_samples = samples if self._log_samples else None
        if (
            not selected_samples
            and step != 1
            and step != self._plan.get("optimizer_updates")
            and step % self._log_every
        ):
            return
        payload = {}
        for key in _METRICS:
            if key not in row:
                continue
            if not _finite_number(row[key]):
                raise ValueError("W&B 訓練指標必須是有限數值；未送出此筆資料。")
            payload[f"train/{key}"] = row[key]
        # Krea-style names, not its epoch/moving-average semantics: the trainer
        # supplies this run's cumulative optimizer-update average (including
        # unlogged updates) and the LR actually used, before scheduler.step().
        # Never recompute an average from cadence-filtered history.
        for source, alias in (
            ("loss", "loss/current"),
            ("loss_average", "loss/average"),
            ("learning_rate", "lr/dit"),
        ):
            if source in row:
                if not _finite_number(row[source]):
                    raise ValueError("W&B 訓練指標必須是有限數值；未送出此筆資料。")
                payload[alias] = row[source]
        elapsed = row.get("elapsed_seconds")
        if elapsed is not None and elapsed > 0:
            rate = step / elapsed
            if not _finite_number(rate):
                raise ValueError("W&B 每秒步數不是有限數值；未送出此筆資料。")
            payload["train/steps_per_second"] = rate
        try:
            with self._sdk_scope():
                if selected_samples:
                    selected_samples = list(selected_samples)
                    indices = set()
                    for sample in selected_samples:
                        index = sample.get("index") if isinstance(sample, Mapping) else None
                        if type(index) is not int or index < 0 or index in indices:
                            raise ValueError("W&B sample index 必須是唯一的非負整數。")
                        indices.add(index)
                    images = [
                        _sample_image(self._output_dir, sample, step, self._wandb, self._sanitizer)
                        for sample in selected_samples
                    ]
                    payload["samples/images"] = images
                    # Reuse the very same Image instances: no second PNG encoding
                    # or duplicate media file for the single-image panel aliases.
                    for sample, image in zip(selected_samples, images):
                        payload[f"sample_{sample['index']}"] = image
                # One committed history row: never commit scalars before media.
                self._run.log(payload, step=step)
        except Exception:
            raise RuntimeError("W&B 記錄訓練指標失敗；已停止訓練。") from None

    def _record_training_log(self, run):
        snapshot = _training_log_snapshot(self._output_dir)
        if snapshot is None:
            self._info["training_log_status"] = "missing"
            return
        path, count, columns = snapshot
        artifact = self._wandb.Artifact(
            name=f"training-log-{run.id}",
            type="training-log",
            description=(
                "Validated numeric metrics.csv snapshot, not console or raw logs. "
                "loss_average is the run cumulative optimizer-update mean, "
                "not Krea's epoch moving average. learning_rate is the used update LR."
            ),
            metadata={"rows": count, "columns": columns, "numeric_only": True},
        )
        artifact.add_file(str(path), name="metrics.csv")
        run.log_artifact(artifact)
        # 'recorded' includes offline staging; it does not assert remote upload.
        self._info["training_log_status"] = "recorded"

    def _finish(self, exit_code, *, suppress, training_log=False):
        if self._run is None:
            return
        run, self._run = self._run, None
        artifact_error = None
        if training_log and self._log_training_log:
            try:
                with self._sdk_scope():
                    self._record_training_log(run)
            except BaseException as exc:
                artifact_error = exc
                self._info["training_log_status"] = "failed"
                self._info["training_log_error"] = "W&B training-log 保存失敗。"
                exit_code = 1
        self._info["exit_code"] = exit_code
        finish_error = None
        try:
            with self._sdk_scope():
                run.finish(exit_code=exit_code)
        except BaseException as exc:
            finish_error = exc
            self._info["finish_error"] = "W&B 結束失敗。"
        if not suppress:
            error = artifact_error if artifact_error is not None else finish_error
            if error is not None and not isinstance(error, Exception):
                raise error
            if artifact_error is not None:
                raise RuntimeError(
                    "W&B training-log 保存失敗；未確認 artifact 完整保存。"
                ) from None
            if finish_error is not None:
                raise RuntimeError("W&B 結束失敗；無法確認指標已完整保存。") from None

    def __exit__(self, exc_type, exc_value, traceback):
        self._finish(0 if exc_type is None else 1, suppress=exc_type is not None, training_log=True)
        return False
