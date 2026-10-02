"""Privacy/lifecycle contract and a real (CPU-only) W&B offline round trip."""

import builtins
import json
import os
import tempfile
import traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from qwen21_trainer.tracking import TrainingTracker

PRIVATE = "/private/training-data/do-not-upload"
SECRET = "fake-secret-never-a-real-credential"
PROMPT = "PRIVATE_PROMPT_SENTINEL"


def fixture_config(**options):
    return SimpleNamespace(
        wandb={
            "enabled": True,
            "mode": "offline",
            "project": "qwen-image-21-trainer",
            "entity": "",
            "name": "",
            "log_every": 1,
            **options,
        },
        training={
            "seed": 42,
            "rank": 4,
            "max_steps": 5,
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
            "gradient_accumulation_steps": 2,
            "gradient_checkpointing": True,
            "checkpointing_offload": False,
            "num_workers": 0,
            "prefetch_factor": 2,
            "save_every": 5,
            "loss_type": "wavelet",
            "loss_weighting": "diffsynth",
            "optimizer": "adamw",
            "optimizer_params": {"betas": [0.9, 0.999], "eps": 1e-8, "key": SECRET},
            "output_dir": PRIVATE,
            "cache_dir": PRIVATE,
            "secret": SECRET,
        },
        model={"root": PRIVATE},
        dataset={"path": PRIVATE},
        sample={"prompt": PROMPT},
        path=Path(PRIVATE) / "config.toml",
    )


def fixture_plan():
    return {
        "config": {"path": PRIVATE, "prompt": PROMPT, "api_key": SECRET},
        "num_images": 3,
        "weighted_images_per_pass": 6,
        "optimizer_updates": 5,
        "rank": 4,
        "alpha": 4,
        "effective_batch_size": 2,
        "loss_type": "wavelet",
        "loss_weighting": "diffsynth",
        "optimizer": "adamw",
        "optimizer_params": {"eps": 1e-8, "betas": [0.9, 0.999], "key": SECRET},
        "cpu_prefetch": {
            "num_workers": 0,
            "prefetch_factor": 2,
            "max_inflight_tasks": 0,
            "backend": "synchronous",
            "pin_memory": False,
            "data_path": PRIVATE,
        },
        "latent_mode": "posterior_mean",
        "schedule_mu": 0.8,
        "warm_start": PRIVATE,
        "prompt": PROMPT,
        "secret": SECRET,
    }


def row(step):
    return {
        "step": step,
        "loss": 1.0 / step,
        "learning_rate": 0.0001,
        "elapsed_seconds": float(step * 2),
        "images_seen": step * 2,
        "data_wait_seconds": 0.01 * step,
        "prompt": PROMPT,
        "image": PRIVATE,
        "api_key": SECRET,
    }


@pytest.fixture
def sdk(monkeypatch):
    run = Mock(id="offline-test-id", url="https://example.invalid/team/project/runs/id")

    def init(**kwargs):
        run.settings = SimpleNamespace(mode=kwargs["mode"])
        return run

    fake = SimpleNamespace(Settings=Mock(side_effect=lambda **kw: kw), init=Mock(side_effect=init))
    monkeypatch.setattr("qwen21_trainer.tracking.importlib.import_module", Mock(return_value=fake))
    return fake, run


def test_disabled_never_imports_sdk_or_writes_files(tmp_path, monkeypatch):
    output = tmp_path / "not-created"
    imports = []
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        imports.append(name)
        assert not name.startswith("wandb")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    loader = Mock(side_effect=AssertionError("SDK must not import"))
    monkeypatch.setattr("qwen21_trainer.tracking.importlib.import_module", loader)
    with TrainingTracker(fixture_config(enabled=False), fixture_plan(), output) as tracker:
        tracker.log({"completely": "ignored"})
        assert tracker.info["enabled"] is False
    loader.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("key", [None, "", "  "])
def test_online_missing_key_fails_before_import_or_files(tmp_path, monkeypatch, key):
    if key is None:
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
    else:
        monkeypatch.setenv("WANDB_API_KEY", key)
    loader = Mock(side_effect=AssertionError("must fail before SDK"))
    monkeypatch.setattr("qwen21_trainer.tracking.importlib.import_module", loader)
    output = tmp_path / "not-created"
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        with TrainingTracker(fixture_config(mode="online"), fixture_plan(), output):
            pytest.fail("body should not run")
    loader.assert_not_called()
    assert not output.exists()


def test_kwargs_hyperparams_metrics_and_environment_are_private(tmp_path, sdk, capsys):
    fake, run = sdk
    scoped_keys = (
        "WANDB_DIR",
        "WANDB_CACHE_DIR",
        "WANDB_CONFIG_DIR",
        "WANDB_DATA_DIR",
        "WANDB_ARTIFACT_DIR",
        "WANDB_ERROR_REPORTING",
        "WANDB_SILENT",
        "TMPDIR",
    )
    before_env = {key: os.environ.get(key) for key in scoped_keys}
    before_tempdir = tempfile.tempdir
    with TrainingTracker(fixture_config(), fixture_plan(), tmp_path / "run-name") as tracker:
        tracker.log(row(1))
        snapshot = tracker.info
        snapshot["name"] = "mutated"
        assert tracker.info["name"] == "run-name"
    assert {key: os.environ.get(key) for key in scoped_keys} == before_env
    assert tempfile.tempdir == before_tempdir
    kwargs = fake.init.call_args.kwargs
    assert kwargs["dir"] == str(tmp_path / "run-name")
    assert kwargs["entity"] is None
    assert kwargs["name"] == "run-name"
    assert kwargs["mode"] == "offline"
    assert kwargs["resume"] == "never"
    assert kwargs["reinit"] == "create_new"
    assert kwargs["save_code"] is kwargs["monitor_gym"] is kwargs["sync_tensorboard"] is False
    settings = kwargs["settings"]
    for key in (
        "disable_code",
        "disable_git",
        "disable_job_creation",
        "x_disable_stats",
        "x_disable_meta",
        "x_disable_machine_info",
        "silent",
        "quiet",
    ):
        assert settings[key] is True
    assert settings["console"] == "off"
    assert settings["save_code"] is False
    config = kwargs["config"]
    assert config["num_images"] == 3
    assert config["optimizer_params"] == {"eps": 1e-8, "betas": [0.9, 0.999]}
    assert config["cpu_prefetch"]["backend"] == "synchronous"
    payload = run.log.call_args.args[0]
    assert set(payload) == {
        "train/step",
        "train/loss",
        "train/learning_rate",
        "train/elapsed_seconds",
        "train/images_seen",
        "train/data_wait_seconds",
        "train/steps_per_second",
        "loss/current",
        "lr/dit",
    }
    assert payload["train/steps_per_second"] == 0.5
    assert run.log.call_args.kwargs == {"step": 1}
    run.finish.assert_called_once_with(exit_code=0)
    assert tracker.info["exit_code"] == 0
    assert tracker.info["run_id"] == "offline-test-id"
    assert tracker.info["run_url"] is None
    run.watch.assert_not_called()
    run.save.assert_not_called()
    run.log_artifact.assert_not_called()
    serialized = json.dumps({"config": config, "payload": payload, "info": tracker.info})
    captured = capsys.readouterr()
    for sentinel in (PRIVATE, SECRET, PROMPT):
        assert sentinel not in serialized + captured.out + captured.err


def test_path_in_allowed_key_and_unknown_nested_values_never_upload(tmp_path, sdk):
    cfg, plan = fixture_config(), fixture_plan()
    cfg.training["learning_rate"] = PRIVATE
    cfg.training["gradient_checkpointing"] = SECRET
    plan["rank"] = PRIVATE
    plan["optimizer"] = PROMPT
    plan["optimizer_params"] = {"eps": PRIVATE, "betas": [0.9, SECRET], "unknown": SECRET}
    plan["cpu_prefetch"] = {"num_workers": PRIVATE, "backend": SECRET, "extra": PROMPT}
    with TrainingTracker(cfg, plan, tmp_path):
        pass
    payload = json.dumps(sdk[0].init.call_args.kwargs["config"])
    for sentinel in (PRIVATE, SECRET, PROMPT):
        assert sentinel not in payload


def test_cadence_logs_first_interval_and_last(tmp_path, sdk):
    with TrainingTracker(fixture_config(log_every=3), fixture_plan(), tmp_path) as tracker:
        for step in range(1, 6):
            tracker.log(row(step))
    assert [call.kwargs["step"] for call in sdk[1].log.call_args_list] == [1, 3, 5]


@pytest.mark.parametrize("elapsed", [None, 0.0, -1.0])
def test_optional_metrics_are_not_fabricated(tmp_path, sdk, elapsed):
    event = {"step": 1, "loss": 0.5}
    if elapsed is not None:
        event["elapsed_seconds"] = elapsed
    with TrainingTracker(fixture_config(), fixture_plan(), tmp_path) as tracker:
        tracker.log(event)
    payload = sdk[1].log.call_args.args[0]
    assert "train/steps_per_second" not in payload
    assert "train/data_wait_seconds" not in payload
    assert "train/images_seen" not in payload


@pytest.mark.parametrize("error", [RuntimeError("training failure"), KeyboardInterrupt()])
@pytest.mark.parametrize("finish_fails", [False, True])
def test_original_error_survives_and_finish_gets_one(tmp_path, sdk, error, finish_fails):
    if finish_fails:
        sdk[1].finish.side_effect = RuntimeError(SECRET)
    with pytest.raises(type(error)) as caught:
        with TrainingTracker(fixture_config(), fixture_plan(), tmp_path) as tracker:
            raise error
    assert caught.value is error
    sdk[1].finish.assert_called_once_with(exit_code=1)
    assert tracker.info["exit_code"] == 1
    if finish_fails:
        assert SECRET not in json.dumps(tracker.info)
        assert tracker.info["finish_error"] == "W&B 結束失敗。"


@pytest.mark.parametrize("operation", ["init", "define_metric", "log", "finish"])
def test_sdk_failures_have_no_secret_in_traceback_or_output(tmp_path, sdk, capsys, operation):
    fake, run = sdk

    def fail(*args, **kwargs):
        print(SECRET)
        raise RuntimeError(SECRET + PRIVATE + PROMPT)

    target = fake.init if operation == "init" else getattr(run, operation)
    target.side_effect = fail
    with pytest.raises(RuntimeError) as caught:
        with TrainingTracker(fixture_config(), fixture_plan(), tmp_path) as tracker:
            if operation == "log":
                tracker.log(row(1))
    if operation == "init":
        run.finish.assert_not_called()
    elif operation == "define_metric":
        run.finish.assert_called_once_with(exit_code=1)
    text = "".join(traceback.format_exception(caught.value))
    capture = capsys.readouterr()
    for sentinel in (SECRET, PRIVATE, PROMPT):
        assert sentinel not in text + capture.out + capture.err


def test_log_failure_is_sanitized_and_marks_failed_run(tmp_path, sdk, capsys):
    sdk[1].log.side_effect = RuntimeError(SECRET)
    with pytest.raises(RuntimeError, match="記錄訓練指標失敗") as caught:
        with TrainingTracker(fixture_config(), fixture_plan(), tmp_path) as tracker:
            tracker.log(row(1))
    sdk[1].finish.assert_called_once_with(exit_code=1)
    assert SECRET not in "".join(traceback.format_exception(caught.value))
    assert SECRET not in str(capsys.readouterr())


@pytest.mark.parametrize("bad", [True, "private", float("nan"), float("inf")])
def test_non_numeric_or_nonfinite_metrics_rejected(tmp_path, sdk, bad):
    event = row(1)
    event["loss"] = bad
    with pytest.raises(ValueError):
        with TrainingTracker(fixture_config(), fixture_plan(), tmp_path) as tracker:
            tracker.log(event)
    sdk[1].log.assert_not_called()
    sdk[1].finish.assert_called_once_with(exit_code=1)


@pytest.mark.parametrize(
    "url",
    ["https://:password@example.invalid/run/id", "https://user:password@example.invalid/run/id"],
)
def test_public_metadata_rejects_url_credentials(tmp_path, sdk, monkeypatch, url):
    monkeypatch.setenv("WANDB_API_KEY", SECRET)
    sdk[1].url = url
    with TrainingTracker(fixture_config(mode="online"), fixture_plan(), tmp_path) as tracker:
        assert tracker.info["run_url"] is None


def test_online_mock_uses_key_only_via_environment(tmp_path, sdk, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", SECRET)
    sdk[1].url = "https://example.invalid/run/id?credential=do-not-copy#fragment"
    with TrainingTracker(
        fixture_config(mode="online", entity="team", name="named"), fixture_plan(), tmp_path
    ) as tracker:
        assert tracker.info["run_url"] == "https://example.invalid/run/id"
    kwargs = sdk[0].init.call_args.kwargs
    assert "api_key" not in kwargs
    assert kwargs["entity"] == "team" and kwargs["name"] == "named"
    assert SECRET not in json.dumps(tracker.info)


@pytest.mark.parametrize("actual_mode", ["disabled", "online"])
def test_sdk_mode_change_is_never_silent_fallback(tmp_path, sdk, actual_mode):
    fake, run = sdk
    run.settings = SimpleNamespace(mode=actual_mode)
    fake.init.side_effect = None
    fake.init.return_value = run
    with pytest.raises(RuntimeError, match="初始化失敗"):
        with TrainingTracker(fixture_config(), fixture_plan(), tmp_path):
            pytest.fail("must not continue after SDK mode change")
    run.finish.assert_called_once_with(exit_code=1)


def _offline_records(path):
    # 0.30 no longer ships the legacy Python DataStore reader. Decode its
    # LevelDB-style journal: 7-byte :W&B header, 32KiB blocks, CRC32-checked
    # FULL/FIRST/MIDDLE/LAST chunks containing the SDK's own protobuf Record.
    import struct
    import zlib

    from wandb.proto import wandb_internal_pb2

    raw = path.read_bytes()
    assert struct.unpack("<4sHB", raw[:7]) == (b":W&B", 0xBEE1, 0)
    offset, pending, records = 7, bytearray(), []
    while offset < len(raw):
        remaining = 32768 - offset % 32768
        if remaining < 7:
            assert raw[offset : offset + remaining] == b"\0" * remaining
            offset += remaining
            continue
        checksum, length, kind = struct.unpack("<IHB", raw[offset : offset + 7])
        assert kind in {1, 2, 3, 4}
        assert length + 7 <= remaining
        offset += 7
        chunk = raw[offset : offset + length]
        assert len(chunk) == length
        assert zlib.crc32(bytes([kind]) + chunk) & 0xFFFFFFFF == checksum
        offset += length
        if kind in {1, 2}:
            assert not pending
        else:
            assert pending
        pending.extend(chunk)
        if kind in {1, 4}:
            record = wandb_internal_pb2.Record()
            record.ParseFromString(bytes(pending))
            records.append(record)
            pending.clear()
    assert not pending
    return records


def test_real_offline_sdk_roundtrip(tmp_path, monkeypatch):
    """Runs the actual pinned SDK and decodes its actual binary event journal."""
    wandb = pytest.importorskip("wandb")
    assert wandb.__version__ == "0.30.0"
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    # Offline is explicitly selected in init; a bogus loopback API also ensures
    # this fixture cannot accidentally send training payloads to a real service.
    monkeypatch.setenv("WANDB_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("WANDB_MODE", "online")
    output = tmp_path / "offline-proof"
    tracker = TrainingTracker(fixture_config(), fixture_plan(), output)
    with tracker:
        for step in (1, 2):
            tracker.log(row(step))
    # The core journal writer flushes at service teardown, not necessarily at
    # Run.finish() return. This public SDK call replaces waiting for atexit.
    wandb.teardown()
    assert tracker.info["exit_code"] == 0
    assert tracker.info["run_url"] is None
    journals = list(output.glob("wandb/offline-run-*/run-*.wandb"))
    assert len(journals) == 1
    records = _offline_records(journals[0])
    kinds = Counter(record.WhichOneof("record_type") for record in records)
    histories = []
    for record in records:
        if record.HasField("history"):
            histories.append(
                {
                    item.key or ".".join(item.nested_key): json.loads(item.value_json)
                    for item in record.history.item
                }
            )
    assert len(histories) == 2
    assert [entry["train/step"] for entry in histories] == [1, 2]
    assert [entry["train/loss"] for entry in histories] == [1.0, 0.5]
    assert [entry["_step"] for entry in histories] == [1, 2]
    assert all(entry["train/steps_per_second"] == 0.5 for entry in histories)
    assert kinds["stats"] == kinds["output"] == kinds["output_raw"] == kinds["artifact"] == 0
    assert not list(output.rglob("wandb-metadata.json"))
    assert not list(output.rglob("requirements.txt"))
    serialized = b"".join(record.SerializeToString() for record in records)
    for sentinel in (PRIVATE, SECRET, PROMPT, str(output), str(Path(__file__).resolve())):
        assert sentinel.encode() not in serialized
    # Check local logs too, not just the payload journal. Only synthetic test
    # sentinels are scanned; the real API credential is never read or printed.
    local_files = [path for path in output.rglob("*") if path.is_file()]
    for path in local_files:
        contents = path.read_bytes()
        assert all(sentinel.encode() not in contents for sentinel in (PRIVATE, SECRET, PROMPT))
        assert path.resolve().is_relative_to(output.resolve())
    exits = [record.exit.exit_code for record in records if record.HasField("exit")]
    assert exits == [0]
    uploaded_files = [
        item.path for record in records if record.HasField("files") for item in record.files.files
    ]
    assert not any("requirements" in name or "code/" in name for name in uploaded_files)
    report = {
        "sdk_version": wandb.__version__,
        "mode": "offline",
        "real_sdk": True,
        "synthetic_metrics_only": True,
        "run_id": tracker.info["run_id"],
        "journal": str(journals[0]),
        "history_rows": histories,
        "record_counts": dict(kinds),
        "exit_codes": exits,
        "private_sentinels_absent": True,
        "uploaded_file_records": uploaded_files,
        "metadata_json_absent": True,
    }
    (tmp_path / "offline-evidence.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
