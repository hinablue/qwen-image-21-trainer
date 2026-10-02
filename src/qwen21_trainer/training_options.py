"""Strict, torch-free validation for the supported training recipe options."""

from __future__ import annotations

import math
import struct
from copy import deepcopy

from .block_filter import validate_block_patterns

_ADOPT_BOOLS = {
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
_ADOPT_NUMBERS = {"eps", "nesterov_coef", "beta2_min", "ema_alpha", "tiny_spike", "centered_wd"}
_ADOPT_INTS = {"k_warmup_steps", "k_logging"}
_ADOPT_ENUMS = {
    "orthogonal_gradient": {"disabled", "flattened", "iterative"},
    "state_precision": {"auto", "fp32", "factored", "bf16_sr", "fp16", "int8_sr"},
    "centered_wd_mode": {"full", "float8", "int8", "int4"},
}


def _number(value, name, *, maximum=None):
    if type(value) not in (int, float):
        raise ValueError(f"{name} 必須是非負有限數值，不能是布林值。")
    try:
        finite = math.isfinite(value)
    except OverflowError as exc:
        raise ValueError(f"{name} 數值過大。") from exc
    if not finite or value < 0:
        raise ValueError(f"{name} 必須是非負有限數值，不能是布林值。")
    if maximum is not None and value >= maximum:
        raise ValueError(f"{name} 必須小於 {maximum}。")


def resolve_lora_alpha(rank, alpha=None):
    """Omission means rank; an explicit 1 is a real alpha, not a sentinel."""
    if type(rank) is not int or rank < 1:
        raise ValueError("training.rank 必須是正整數。")
    value = rank if alpha is None else alpha
    if type(value) not in (int, float):
        raise ValueError("training.alpha 必須是正有限數值，不能是布林值。")
    try:
        numeric = float(value)
        scale = numeric / rank
        scale32 = struct.unpack("f", struct.pack("f", scale))[0]
    except (OverflowError, ValueError):
        raise ValueError("training.alpha 或 alpha/rank 超出可用範圍。") from None
    if not math.isfinite(numeric) or numeric <= 0 or not math.isfinite(scale32) or scale32 <= 0:
        raise ValueError("training.alpha 必須為正且 alpha/rank 在 FP32 可用範圍內。")
    if type(value) is int and int(numeric) != value:
        raise ValueError("training.alpha 無法精確保存為浮點數。")
    return value


def normalize_training_options(training):
    result = deepcopy(training)
    for key in ("include_blocks", "exclude_blocks"):
        result[key] = validate_block_patterns(result.get(key, []), f"training.{key}")
    if "rank" in result:
        result["alpha"] = resolve_lora_alpha(result["rank"], result.get("alpha"))
    elif "alpha" in result:
        raise ValueError("training.alpha 需要搭配 training.rank。")
    loss = result.setdefault("loss_type", "mse")
    weighting = result.setdefault("loss_weighting", "diffsynth")
    name = result.setdefault("optimizer", "adamw")
    if loss not in ("mse", "wavelet"):
        raise ValueError("training.loss_type 只支援 mse 或 wavelet。")
    if weighting not in ("diffsynth", "none"):
        raise ValueError("training.loss_weighting 只支援 diffsynth 或 none。")
    if not isinstance(name, str) or name.lower() not in ("adamw", "adopt_adv"):
        raise ValueError("training.optimizer 只支援 adamw 或 adopt_adv。")
    name = result["optimizer"] = name.lower()
    params = result.setdefault("optimizer_params", {})
    if not isinstance(params, dict):
        raise ValueError("training.optimizer_params 必須是 TOML 表格。")
    if {"lr", "weight_decay", "params"} & params.keys():
        raise ValueError(
            "learning_rate／weight_decay 請設定在 [training]，不可在 optimizer_params 重複設定。"
        )
    bools = _ADOPT_BOOLS if name == "adopt_adv" else {"amsgrad", "maximize"}
    numbers = _ADOPT_NUMBERS if name == "adopt_adv" else {"eps"}
    integers = _ADOPT_INTS if name == "adopt_adv" else set()
    enums = _ADOPT_ENUMS if name == "adopt_adv" else {}
    allowed = bools | numbers | integers | set(enums) | {"betas"}
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"{name} 不支援的 optimizer_params：{sorted(unknown)}")
    for key, value in params.items():
        label = f"training.optimizer_params.{key}"
        if key in bools and type(value) is not bool:
            raise ValueError(f"{label} 必須是布林值。")
        if key in numbers:
            _number(value, label, maximum=1 if key in {"beta2_min", "ema_alpha"} else None)
        if key in integers and (type(value) is not int or value < 0):
            raise ValueError(f"{label} 必須是非負整數。")
        if key in enums and (not isinstance(value, str) or value not in enums[key]):
            raise ValueError(f"{label} 必須是 {sorted(enums[key])} 其中之一。")
        if key == "betas":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"{label} 必須有兩個數值。")
            for beta in value:
                _number(beta, label, maximum=1)
    if name == "adopt_adv" and params.get("kourkoutas_beta", False):
        beta2 = params.get("betas", (0.9, 0.9999))[1]
        if beta2 <= params.get("beta2_min", 0.9):
            raise ValueError("Kourkoutas beta 要求 betas[1] > beta2_min。")
    return result
