"""Validated optimizer-update LR schedules without heavyweight import-time deps.

``diffsynth`` preserves the original runner's PyTorch ConstantLR defaults: the
first five optimizer updates use one third of the base LR. It is deliberately
not an alias for the genuinely constant Hugging Face schedule.

The other schedules use Hugging Face's zero-indexed convention. Construction
sets the LR for schedule step 0; call ``optimizer.step()`` before each
``scheduler.step()``. Thus N updates use schedule steps 0 through N - 1, while
the final scheduler step prepares step N. With warmup, the first update uses
LR zero; cosine reaches zero at step N, not at the final update's used LR.

Schedulers are newly constructed for each training run, including LoRA
warm-starts. This module does not implement full-state training resume.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

_SCHEDULES = ("diffsynth", "constant", "constant_with_warmup", "cosine")
_LEGACY_FACTOR = 1.0 / 3.0
_LEGACY_TOTAL_ITERS = 5


def normalize_lr_options(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a training settings mapping, supply LR defaults, and validate.

    ``max_steps`` must already be present as a positive integer optimizer-update
    budget (normally supplied by Config). This helper only supplies defaults for
    ``lr_scheduler`` and ``num_warmup``; it never invents a training budget
    or modifies the input, including nested optimizer options.
    """
    if not isinstance(settings, Mapping):
        raise ValueError("training 設定必須是對應表格。")
    result = deepcopy(dict(settings))
    name = result.setdefault("lr_scheduler", "diffsynth")
    warmup = result.setdefault("num_warmup", 0)
    maximum = result.get("max_steps")
    if not isinstance(name, str) or name not in _SCHEDULES:
        raise ValueError(
            "training.lr_scheduler 只支援 diffsynth、constant、constant_with_warmup 或 cosine。"
        )
    if type(maximum) is not int or maximum < 1:
        raise ValueError("training.max_steps 必須是正整數（不可使用布林值）。")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("training.num_warmup 必須是非負整數（不可使用布林值）。")
    if warmup > maximum:
        raise ValueError("training.num_warmup 不可超過 training.max_steps。")
    if name in ("diffsynth", "constant") and warmup != 0:
        raise ValueError(f"training.lr_scheduler={name} 要求 num_warmup=0，不會忽略暖身設定。")
    if name == "cosine" and warmup >= maximum:
        raise ValueError("cosine 要求 training.num_warmup 小於 training.max_steps。")
    return result


def build_lr_scheduler(optimizer: Any, settings: Mapping[str, Any]) -> Any:
    """Build a fresh scheduler; step it once after each optimizer update.

    Validate before importing model libraries or mutating optimizer LR fields.
    The fixed cosine ``num_cycles=0.5`` is a single half-cosine over the remaining
    update budget, not cosine restarts or an adjustable cycles option.
    """
    settings = normalize_lr_options(settings)
    name = settings["lr_scheduler"]
    if name == "diffsynth":
        from torch.optim.lr_scheduler import ConstantLR

        return ConstantLR(
            optimizer, factor=_LEGACY_FACTOR, total_iters=_LEGACY_TOTAL_ITERS, last_epoch=-1
        )

    from transformers import get_scheduler

    return get_scheduler(
        name=name,
        optimizer=optimizer,
        num_warmup_steps=settings["num_warmup"],
        num_training_steps=settings["max_steps"],
        scheduler_specific_kwargs={"num_cycles": 0.5} if name == "cosine" else {},
    )


def lr_scheduler_report(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe schedule contract without importing torch/transformers.

    Report schedule indices separately from the LR actually used for an update.
    The runtime's per-update ``learning_rate`` remains the authoritative used LR.
    """
    settings = normalize_lr_options(settings)
    name = settings["lr_scheduler"]
    warmup = settings["num_warmup"]
    maximum = settings["max_steps"]
    descriptions = {
        "diffsynth": "First five optimizer updates use base_lr / 3, then base_lr.",
        "constant": "Every optimizer update uses base_lr; no warmup.",
        "constant_with_warmup": "Optional linear warmup from zero, then base_lr.",
        "cosine": "Optional linear warmup, then one half-cosine decay to zero at max_steps.",
    }
    return {
        "name": name,
        "implementation": (
            "torch.optim.lr_scheduler.ConstantLR"
            if name == "diffsynth"
            else "transformers.get_scheduler"
        ),
        "description": descriptions[name],
        "max_steps": maximum,
        "num_warmup": warmup,
        "num_cycles": 0.5 if name == "cosine" else None,
        "legacy_factor": _LEGACY_FACTOR if name == "diffsynth" else None,
        "legacy_total_iters": _LEGACY_TOTAL_ITERS if name == "diffsynth" else None,
        "step_unit": "optimizer_update",
        "step_timing": "initialize_at_step_0_then_step_after_each_optimizer_update",
        "first_update_lr_factor": (
            _LEGACY_FACTOR if name == "diffsynth" else (0.0 if warmup else 1.0)
        ),
        "last_update_schedule_step": maximum - 1,
        "after_last_update_schedule_step": maximum,
        "zero_lr_at_schedule_step": maximum if name == "cosine" else None,
        "warm_start": "reset_optimizer_and_scheduler",
    }
