"""Strict, model-library-free TOML/YAML configuration for the local T2I trainer.

All path values in the section dictionaries are resolved absolute strings;
``Config.path`` is a resolved ``Path``. Loading does not create directories or
require model weights, training data, PyTorch, or a GPU.
"""

from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DEFAULTS: dict[str, dict[str, Any]] = {
    "model": {
        "root": "../models/Qwen-Image-2.1",
        "format": "official",
        "processor_path": "",
        "device": "cuda",
        "attention": "segmented",
        "cpu_offload": False,
    },
    "dataset": {
        "path": "../data/train",
        "caption_extension": ".txt",
        "max_pixels": 262144,
        "recursive": True,
    },
    "training": {
        "output_dir": "../output/small",
        "cache_dir": "../cache/small",
        "seed": 42,
        "rank": 32,
        "alpha": None,
        "include_blocks": [],
        "exclude_blocks": [],
        "learning_rate": 1e-4,
        "lr_scheduler": "diffsynth",
        "num_warmup": 0,
        "weight_decay": 0.01,
        "loss_type": "mse",
        "loss_weighting": "diffsynth",
        "optimizer": "adamw",
        "optimizer_params": {},
        "max_steps": 300,
        "gradient_accumulation_steps": 1,
        "num_workers": 0,
        "prefetch_factor": 2,
        "save_every": 100,
        "gradient_checkpointing": True,
        "checkpointing_offload": False,
    },
    "wandb": {
        "enabled": False,
        "project": "qwen-image-21-trainer",
        "entity": "",
        "name": "",
        "mode": "online",
        "log_every": 1,
        "log_config": True,
        "log_samples": True,
        "log_training_log": True,
    },
    "sample": {
        "enabled": False,
        "every": 100,
        "prompts": [],
        "prompt": "a photo of a small ceramic cup on a wooden table",
        "negative_prompt": "",
        "width": 512,
        "height": 512,
        "steps": 30,
        "cfg_scale": 3.0,
        "seed": 42,
    },
}


def _path(value: str | Path, base: Path, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{label} 必須是非空白路徑。")
    if "\x00" in str(value):
        raise ValueError(f"{label} 的路徑不可含 NUL 字元。")
    try:
        result = Path(value).expanduser()
        if not result.is_absolute():
            result = base / result
        return result.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"無法解析 {label} 的路徑：{value}") from exc


def _integer(value: Any, label: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} 必須是大於或等於 {minimum} 的整數（不可使用布林值）。")


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} 必須是有限數值（不可使用布林值）。")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} 必須是有限數值。") from exc
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        condition = "大於零" if positive else "大於或等於零"
        raise ValueError(f"{label} 必須是{condition}的有限數值。")
    return result


def _sections(raw: dict[str, Any], base: Path) -> dict[str, dict[str, Any]]:
    unknown = set(raw) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"不支援的設定區段：{', '.join(sorted(map(str, unknown)))}")
    sections = {}
    for name, defaults in _DEFAULTS.items():
        supplied = raw.get(name, {})
        if not isinstance(supplied, dict):
            raise ValueError(f"[{name}] 必須是 TOML 設定表格。")
        unknown_keys = set(supplied) - set(defaults)
        if unknown_keys:
            raise ValueError(
                f"[{name}] 有不支援的設定欄位：{', '.join(sorted(map(str, unknown_keys)))}"
            )
        values = deepcopy({**defaults, **supplied})
        for key, default in defaults.items():
            value = values[key]
            if type(default) is bool and type(value) is not bool:
                raise ValueError(f"{name}.{key} 必須是布林值 true 或 false。")
            if isinstance(default, str) and not isinstance(value, str):
                raise ValueError(f"{name}.{key} 必須是字串。")
        sections[name] = values

    model, dataset, training, sample = (
        sections[name] for name in ("model", "dataset", "training", "sample")
    )
    if model["device"] not in ("cuda", "cpu"):
        raise ValueError("model.device 只支援 cuda 或 cpu。")
    if model["attention"] not in ("segmented", "flex"):
        raise ValueError("model.attention 只支援 segmented 或 flex。")
    if model["format"] not in ("official", "comfy_bf16"):
        raise ValueError("model.format 只支援 official 或 comfy_bf16。")
    if model["processor_path"]:
        model["processor_path"] = str(_path(model["processor_path"], base, "model.processor_path"))
    if model["format"] == "comfy_bf16" and not model["processor_path"]:
        raise ValueError("comfy_bf16 格式必須明確設定 model.processor_path。")
    extension = dataset["caption_extension"]
    if not re.fullmatch(r"\.[^./\\\x00\s]+", extension):
        raise ValueError("dataset.caption_extension 必須是單一副檔名，例如 .txt。")
    if extension.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise ValueError("dataset.caption_extension 不可使用圖片副檔名。")
    _integer(dataset["max_pixels"], "dataset.max_pixels", 1024)
    for key in ("rank", "max_steps", "gradient_accumulation_steps", "save_every"):
        _integer(training[key], f"training.{key}", 1)
    _integer(training["seed"], "training.seed", 0)
    _integer(training["num_workers"], "training.num_workers", 0)
    _integer(training["prefetch_factor"], "training.prefetch_factor", 1)
    training["learning_rate"] = _number(
        training["learning_rate"], "training.learning_rate", positive=True
    )
    training["weight_decay"] = _number(training["weight_decay"], "training.weight_decay")
    from .lr_schedule import normalize_lr_options
    from .training_options import normalize_training_options

    sections["training"] = training = normalize_lr_options(normalize_training_options(training))
    wandb = sections["wandb"]
    if wandb["mode"] not in ("online", "offline"):
        raise ValueError("wandb.mode 只支援 online 或 offline。")
    if not wandb["project"].strip() or any(c in wandb["project"] for c in "\r\n/\\"):
        raise ValueError("wandb.project 必須是非空專案名稱，不能包含換行或路徑分隔符。")
    _integer(wandb["log_every"], "wandb.log_every", 1)
    for key in ("entity", "name"):
        if any(c in wandb[key] for c in "\r\n"):
            raise ValueError(f"wandb.{key} 不可含換行。")
    _integer(sample["every"], "sample.every", 1)
    prompts = sample["prompts"]
    if (
        not isinstance(prompts, list)
        or len(prompts) > 16
        or any(not isinstance(p, str) or not p.strip() for p in prompts)
    ):
        raise ValueError("sample.prompts 必須是最多 16 個非空字串的陣列。")
    if sample["enabled"] and model["cpu_offload"]:
        raise ValueError("訓練中 sample 暫不支援 model.cpu_offload；請關閉其中一項。")
    for key in ("width", "height"):
        _integer(sample[key], f"sample.{key}", 32)
        if sample[key] % 32:
            raise ValueError(f"sample.{key} 必須是 32 的倍數。")
    _integer(sample["steps"], "sample.steps", 1)
    _integer(sample["seed"], "sample.seed", 0)
    sample["cfg_scale"] = _number(sample["cfg_scale"], "sample.cfg_scale")

    for name, key in (
        ("model", "root"),
        ("dataset", "path"),
        ("training", "output_dir"),
        ("training", "cache_dir"),
    ):
        sections[name][key] = str(_path(sections[name][key], base, f"{name}.{key}"))

    dataset_path = Path(dataset["path"])
    for key in ("output_dir", "cache_dir"):
        destination = Path(training[key])
        if dataset_path == destination or dataset_path.is_relative_to(destination):
            raise ValueError(f"training.{key} 不可等於資料集目錄或成為資料集的祖先目錄。")
        if key == "cache_dir" and destination.is_relative_to(dataset_path):
            raise ValueError("training.cache_dir 不可位於資料集目錄內。")
    return sections


@dataclass(frozen=True)
class Config:
    """Validated configuration; dictionaries contain independent resolved values.

    ``to_dict()`` returns a JSON-serializable snapshot with keys ``path``,
    ``model``, ``dataset``, ``training``, and ``sample``. No batch-size,
    image-editing, or other unsupported training settings are accepted.
    """

    path: Path
    model: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    sample: dict[str, Any] = field(default_factory=dict)
    wandb: dict[str, Any] = field(default_factory=dict)
    datasets: list[dict[str, Any]] = field(default_factory=list)
    source_document: dict[str, Any] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    source_format: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        path = _path(self.path, Path.cwd(), "設定檔")
        values = _sections({name: getattr(self, name) for name in _DEFAULTS}, path.parent)
        object.__setattr__(self, "path", path)
        for name, section in values.items():
            object.__setattr__(self, name, section)
        if not isinstance(self.datasets, list):
            raise ValueError("datasets 必須是 [[datasets]] 陣列。")
        resolved_datasets = []
        for index, entry in enumerate(self.datasets or [self.dataset]):
            if not isinstance(entry, dict):
                raise ValueError(f"datasets[{index}] 必須是表格。")
            spec = dict(entry)
            repeats = spec.pop("repeats", 1)
            _integer(repeats, f"datasets[{index}].repeats", 1)
            unknown = set(spec) - set(_DEFAULTS["dataset"])
            if unknown:
                raise ValueError(f"datasets[{index}] 不支援欄位：{sorted(unknown)}")
            # Validate each root against output/cache boundaries using the same rules.
            per_dataset = _sections({**values, "dataset": {**self.dataset, **spec}}, path.parent)[
                "dataset"
            ]
            resolved_datasets.append({**per_dataset, "repeats": repeats})
        object.__setattr__(self, "datasets", resolved_datasets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            **{name: deepcopy(getattr(self, name)) for name in _DEFAULTS},
            "datasets": [dict(item) for item in self.datasets],
        }


def load_config(path: str | Path) -> Config:
    """Read native TOML/YAML, resolve paths and retain an as-loaded snapshot."""
    from .config_io import read_config_document

    resolved = _path(path, Path.cwd(), "設定檔")
    try:
        raw, source_format = read_config_document(resolved)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"找不到設定檔：{resolved}") from exc
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"無法讀取 TOML/YAML 設定檔 {resolved}：{exc}") from None
    unknown = set(raw) - set(_DEFAULTS) - {"datasets"}
    if unknown:
        raise ValueError(f"不支援的設定區段：{', '.join(sorted(unknown))}")
    if "datasets" in raw and not raw["datasets"]:
        raise ValueError("datasets 不可為空陣列。")
    result = Config(path=resolved, **raw)
    object.__setattr__(result, "source_document", deepcopy(raw))
    object.__setattr__(result, "source_format", source_format)
    return result
