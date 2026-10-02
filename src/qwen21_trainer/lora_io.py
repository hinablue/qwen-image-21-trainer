"""嚴格、alpha-aware 的普通線性 LoRA checkpoint IO。

檔案保存未預乘 scaling 的 A/B，以及每層 FP64 ``<module>.alpha``。
只支援單一 default adapter、uniform rank/alpha、普通 PEFT Linear；不承諾
ai-toolkit fused-MLP、DoRA、rsLoRA、量化 adapter 或改 rank/alpha 的相容性。

``read_checkpoint`` 完整驗證後回傳 CPU tensors：``training_state`` 的 keys 是
``<module>.lora_A.default.weight`` / B，直接用於 PEFT warm-start，絕不先呼叫
DiffSynth converter；``inference_state`` 另含每層 alpha，交給固定版 DiffSynth
``pipe.load_lora(..., state_dict=checkpoint.inference_state)`` 恰好套用一次 scaling。
不要改用 path 載入；upstream path loader 會連 alpha 一起轉成 pipeline dtype。
``legacy`` 僅表示沒有 alpha tensors、沒有新格式標記的舊 A/B-only 檔案；其
alpha 明確解讀為 rank。這是 weights-only warm-start，不恢復 optimizer/RNG。
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

FORMAT = "qwen21.raw-lora"
FORMAT_VERSION = "1"
_FORMAT_METADATA = {
    "lora_format": FORMAT,
    "lora_format_version": FORMAT_VERSION,
    "lora_storage": "raw",
}
_WEIGHT_KEY = re.compile(r"^(.+)\.lora_([AB])(?:\.default)?\.weight$")
_WEIGHT_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
_ALPHA_DTYPES = _WEIGHT_DTYPES | {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


@dataclass(frozen=True)
class LoRACheckpoint:
    """Validated CPU weights; mappings are caller-owned and must not be pre-scaled.

    training_state: canonical default-adapter raw A/B, no alpha tensors.
    inference_state: independent raw A/B copies plus FP64 per-layer alpha tensors.
    rank/alpha: uniform effective configuration (positive int/finite positive float).
    metadata: original safetensors string metadata, never silently repaired.
    legacy: A/B-only, unmarked checkpoint interpreted with alpha == rank.
    """

    training_state: dict[str, torch.Tensor]
    inference_state: dict[str, torch.Tensor]
    rank: int
    alpha: float
    metadata: dict[str, str]
    legacy: bool


@dataclass
class _ModelSpec:
    rank: int
    alpha: float
    parameters: dict[str, torch.nn.Parameter]


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} 必須是正有限數值，不能是布林值。")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} 必須是正有限數值。") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} 必須是正有限數值。")
    return number


def _same(left: float, right: float, *, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=0.0)


def _scale(alpha: float, rank: int) -> float:
    scale = _positive(alpha / rank, "alpha/rank scaling")
    represented = torch.tensor(scale, dtype=torch.float32).item()
    if not math.isfinite(represented) or represented == 0:
        raise ValueError("alpha/rank 必須能表示成非零有限 FP32 scaling。")
    return scale


def _target_ok(target: str) -> bool:
    # Do not guess or strip foreign prefixes, fused target names, or adapter names.
    return bool(target) and all(part and not part.isspace() for part in target.split("."))


def _validate_metadata(
    metadata: Mapping[str, Any], rank: int, alpha: float, *, tolerance: float = 1e-12
) -> bool:
    markers = set(_FORMAT_METADATA) & metadata.keys()
    known = bool(markers)
    if known:
        for key, expected in _FORMAT_METADATA.items():
            if metadata.get(key) != expected:
                raise ValueError(f"不支援或不完整的 LoRA 格式標記 {key}。")
        if not {"rank", "alpha", "scale"} <= metadata.keys():
            raise ValueError("新格式 metadata 必須包含 rank、alpha、scale。")
    # Reject alternate declarations that could mean baked/scaled/unknown storage.
    for key in ("storage", "weight_storage", "weights_storage", "lora_weight_storage"):
        if key in metadata and metadata[key] != "raw":
            raise ValueError(f"不支援的 {key}；只接受 raw A/B。")
    for key in ("baked", "alpha_baked", "weights_baked", "lora_baked", "scale_baked"):
        if key in metadata and str(metadata[key]).lower() not in {"false", "0"}:
            raise ValueError(f"不接受 baked／不明 scaling 標記：{key}。")
    if "format" in metadata and metadata["format"] not in {"pt", FORMAT}:
        raise ValueError("不支援的 checkpoint format，不能猜測是否已 baked。")
    if "format_version" in metadata and metadata["format_version"] != FORMAT_VERSION:
        raise ValueError("不支援的 checkpoint format_version。")
    for key in ("rank", "lora_rank", "ss_network_dim"):
        if key in metadata:
            number = _positive(metadata[key], key)
            if not number.is_integer() or number != rank:
                raise ValueError(f"metadata {key} 與真實 rank={rank} 不一致。")
    for key in ("alpha", "lora_alpha", "ss_network_alpha"):
        if key in metadata and not _same(_positive(metadata[key], key), alpha, tolerance=tolerance):
            raise ValueError(f"metadata {key} 與真實 alpha={alpha!r} 不一致。")
    for key in ("scale", "lora_scale", "scaling"):
        if key in metadata and not _same(
            _positive(metadata[key], key), alpha / rank, tolerance=tolerance
        ):
            raise ValueError(f"metadata {key} 與 alpha/rank 不一致。")
    return known


def _decode(state: Mapping[str, torch.Tensor], metadata: dict[str, str]) -> LoRACheckpoint:
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    alphas: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("LoRA keys/tensors 格式不正確。")
        match = _WEIGHT_KEY.fullmatch(key)
        if match:
            target, part = match.groups()
            if not _target_ok(target):
                raise ValueError(f"不合法的 LoRA target：{key}")
            pair = pairs.setdefault(target, {})
            if part in pair:
                raise ValueError(f"重複的 LoRA tensor（混用命名）：{key}")
            if value.dtype not in _WEIGHT_DTYPES or value.ndim != 2 or min(value.shape) <= 0:
                raise ValueError(f"LoRA weight 必須是非空浮點二維矩陣：{key}")
            if not torch.isfinite(value).all().item():
                raise ValueError(f"LoRA weight 包含 NaN／Inf：{key}")
            pair[part] = value.detach().cpu().contiguous().clone()
        elif key.endswith(".alpha") and _target_ok(key[:-6]):
            if value.dtype not in _ALPHA_DTYPES or value.ndim != 0:
                raise ValueError(f"alpha 必須是數值 scalar tensor（不能是布林值）：{key}")
            _positive(value.item(), key)
            alphas[key[:-6]] = value.detach().cpu().clone()
        else:
            raise ValueError(f"不支援／多餘的 LoRA key，不能默默丟棄：{key}")
    if not pairs:
        raise ValueError("checkpoint 沒有 LoRA A/B weights。")
    if any(set(pair) != {"A", "B"} for pair in pairs.values()):
        raise ValueError("LoRA A/B 必須完整配對。")
    ranks = set()
    for target, pair in pairs.items():
        rank = pair["A"].shape[0]
        if pair["B"].shape[1] != rank or pair["A"].dtype != pair["B"].dtype:
            raise ValueError(f"LoRA A/B shape／dtype 不一致：{target}")
        ranks.add(rank)
    if len(ranks) != 1:
        raise ValueError("只支援 uniform LoRA rank。")
    rank = ranks.pop()
    known = bool(set(_FORMAT_METADATA) & metadata.keys())
    if alphas and set(alphas) != set(pairs):
        raise ValueError("每個 LoRA target 都必須恰有一個 alpha；不可缺漏／多餘。")
    if known and not alphas:
        raise ValueError("新格式 checkpoint 缺少必要的每層 alpha tensors，不能當作 legacy。")
    # FP32 external alpha metadata may differ by its last representable bits.
    # BF16 rounding is deliberately not excused: use CPU FP64 state at inference.
    tolerance = 1e-12 if known or all(v.dtype == torch.float64 for v in alphas.values()) else 1e-7
    alpha = _positive(next(iter(alphas.values())).item(), "alpha") if alphas else float(rank)
    if any(value.item() != alpha for value in alphas.values()):
        raise ValueError("只支援 uniform LoRA alpha。")
    if known and any(value.dtype != torch.float64 for value in alphas.values()):
        raise ValueError("新格式 alpha tensors 必須是 FP64，避免 dtype 轉換破壞 scaling。")
    _scale(alpha, rank)
    _validate_metadata(metadata, rank, alpha, tolerance=tolerance)
    training_state = {
        f"{target}.lora_{part}.default.weight": pairs[target][part]
        for target in sorted(pairs)
        for part in ("A", "B")
    }
    inference_state = {key: value.clone() for key, value in training_state.items()}
    for target in sorted(pairs):
        inference_state[f"{target}.alpha"] = torch.tensor(alpha, dtype=torch.float64)
    return LoRACheckpoint(training_state, inference_state, rank, alpha, metadata.copy(), not alphas)


def read_checkpoint(path: str | Path) -> LoRACheckpoint:
    """完整驗證 safetensors 並讀成 CPU raw tensors；永不修改來源檔案。"""
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            state = {key: handle.get_tensor(key) for key in handle.keys()}
    except SafetensorError as exc:
        raise ValueError(f"LoRA safetensors 檔案損壞／格式不正確：{path}") from exc
    return _decode(state, metadata)


def _inspect_model(model: torch.nn.Module) -> _ModelSpec:
    from peft.tuners.lora.layer import Linear, LoraLayer

    try:
        dit = model.pipe.dit
    except AttributeError as exc:
        raise ValueError("model 必須提供 pipe.dit。") from exc
    parameters: dict[str, torch.nn.Parameter] = {}
    ranks, alphas = set(), []
    configs = getattr(dit, "peft_config", {})
    if configs and set(configs) != {"default"}:
        raise ValueError("只支援 default adapter。")
    for name, module in dit.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        if type(module) is not Linear or type(module.get_base_layer()) is not torch.nn.Linear:
            raise ValueError(f"只支援普通 PEFT Linear：{name}")
        if not _target_ok(name):
            raise ValueError("LoRA Linear 必須有非空 module path。")
        for field in ("r", "lora_alpha", "scaling", "lora_A", "lora_B", "lora_dropout"):
            if set(getattr(module, field)) != {"default"}:
                raise ValueError(f"{name}.{field} 只支援完整單一 default adapter。")
        if (
            module.use_dora.get("default", False)
            or module.use_rslora.get("default", False)
            or module.lora_bias.get("default", False)
            or module.lora_variant
            or module.lora_magnitude_vector
            or module.fan_in_fan_out
            or module.merged_adapters
            or module.disable_adapters
            or module.active_adapters != ["default"]
        ):
            raise ValueError(f"{name} 不是可匯出的普通 raw LoRA（DoRA／rsLoRA／merged 等不支援）。")
        rank = module.r["default"]
        if type(rank) is not int or rank < 1:
            raise ValueError(f"{name} rank 必須是正整數。")
        alpha = _positive(module.lora_alpha["default"], f"{name}.alpha")
        scale = _positive(module.scaling["default"], f"{name}.scaling")
        if not _same(scale, _scale(alpha, rank)):
            raise ValueError(f"{name} scaling 必須等於 alpha/rank，不能是 rsLoRA 或另加倍率。")
        base = module.get_base_layer()
        for part, expected in (("A", (rank, base.in_features)), ("B", (base.out_features, rank))):
            layer = getattr(module, f"lora_{part}")["default"]
            if type(layer) is not torch.nn.Linear or layer.bias is not None:
                raise ValueError(f"{name}.lora_{part} 必須是無 bias 的普通 Linear。")
            parameter = layer.weight
            if tuple(parameter.shape) != expected or parameter.dtype not in _WEIGHT_DTYPES:
                raise ValueError(f"{name}.lora_{part} shape／dtype 不符。")
            if parameter.device.type == "meta" or not torch.isfinite(parameter).all().item():
                raise ValueError(f"{name}.lora_{part} 包含 meta／NaN／Inf。")
            parameters[f"{name}.lora_{part}.default.weight"] = parameter
        if module.lora_A["default"].weight.dtype != module.lora_B["default"].weight.dtype:
            raise ValueError(f"{name} LoRA A/B dtype 不一致。")
        ranks.add(rank)
        alphas.append(alpha)
    if not parameters or len(ranks) != 1 or any(not _same(a, alphas[0]) for a in alphas):
        raise ValueError("model 必須含 uniform rank/alpha 的 default LoRA Linear。")
    if len({id(parameter) for parameter in parameters.values()}) != len(parameters):
        raise ValueError("不支援共享／綁定的 LoRA parameters。")
    return _ModelSpec(ranks.pop(), alphas[0], parameters)


def _match_model(spec: _ModelSpec, checkpoint: LoRACheckpoint) -> None:
    if spec.rank != checkpoint.rank or not _same(spec.alpha, checkpoint.alpha):
        raise ValueError("checkpoint 與 model 的 rank／alpha 不一致；不支援暗中 rescale。")
    expected, actual = set(spec.parameters), set(checkpoint.training_state)
    if expected != actual:
        raise ValueError(
            f"checkpoint targets 不一致；缺少={sorted(expected - actual)}；多餘={sorted(actual - expected)}"
        )
    for key, parameter in spec.parameters.items():
        if parameter.shape != checkpoint.training_state[key].shape:
            raise ValueError(f"checkpoint 與 model 的 shape 不一致：{key}")


def save_checkpoint(model: torch.nn.Module, path: str | Path, metadata: Mapping[str, Any]) -> None:
    """從真實 PEFT module 取得 rank/alpha/scale，原子儲存 raw A/B 與 FP64 alpha。

    呼叫端 metadata 若宣稱不同 rank/alpha/scale 或已 baked，即拒絕；不覆蓋錯誤
    聲明來掩蓋設定失配。完整驗證後才建立暫存檔、原子替換目的檔。
    """
    spec = _inspect_model(model)
    # Validate caller declarations before adding our complete format contract.
    if set(_FORMAT_METADATA) & metadata.keys():
        for key, expected in _FORMAT_METADATA.items():
            if key in metadata and metadata[key] != expected:
                raise ValueError(f"呼叫端 {key} 格式宣告不一致。")
    supplied = {key: value for key, value in metadata.items() if key not in _FORMAT_METADATA}
    _validate_metadata(supplied, spec.rank, spec.alpha)
    merged = {
        **metadata,
        **_FORMAT_METADATA,
        "rank": spec.rank,
        "alpha": spec.alpha,
        "scale": spec.alpha / spec.rank,
    }
    if not all(isinstance(key, str) for key in merged):
        raise ValueError("metadata keys 必須是字串。")
    stored_metadata = {key: str(value) for key, value in merged.items()}
    exported = model.export_trainable_state_dict(model.state_dict(), remove_prefix="pipe.dit.")
    if set(exported) != set(spec.parameters):
        raise ValueError(
            "匯出必須完整且僅含 default LoRA A/B，不可含 base／額外 trainable tensors。"
        )
    state = {}
    for key, parameter in spec.parameters.items():
        value = exported[key].detach().cpu().contiguous().clone()
        if value.dtype != parameter.dtype or not torch.equal(value, parameter.detach().cpu()):
            raise ValueError(f"export 不是原始未縮放 LoRA weight：{key}")
        state[key] = value
    for key in spec.parameters:
        if key.endswith(".lora_A.default.weight"):
            target = key.removesuffix(".lora_A.default.weight")
            state[f"{target}.alpha"] = torch.tensor(spec.alpha, dtype=torch.float64)
    checkpoint = _decode(state, stored_metadata)
    _match_model(spec, checkpoint)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        save_file(state, temporary, metadata=stored_metadata)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_training_checkpoint(model: torch.nn.Module, path: str | Path) -> dict[str, Any]:
    """全檔與 model rank/alpha/target/shape 驗證成功後一次載入 raw A/B。

    不使用 GeneralLoRALoader；不修改 base、scaling、requires_grad、grad buffers、
    training mode 或 RNG。先準備所有 dtype/device 轉換、檢查溢位，才寫入參數。
    """
    checkpoint = read_checkpoint(path)
    spec = _inspect_model(model)
    _match_model(spec, checkpoint)
    staged = {}
    for key, parameter in spec.parameters.items():
        value = checkpoint.training_state[key].to(device=parameter.device, dtype=parameter.dtype)
        if not torch.isfinite(value).all().item():
            raise ValueError(f"checkpoint 轉成 model dtype 後溢位：{key}")
        staged[key] = value
    # All expected failures occur above; backups also cover an unexpected copy error.
    backups = {key: value.detach().clone() for key, value in spec.parameters.items()}
    with torch.no_grad():
        try:
            for key, parameter in spec.parameters.items():
                parameter.copy_(staged[key])
        except Exception:
            for key, parameter in spec.parameters.items():
                parameter.copy_(backups[key])
            raise
    return {
        "path": str(Path(path)),
        "rank": checkpoint.rank,
        "alpha": checkpoint.alpha,
        "scale": checkpoint.alpha / checkpoint.rank,
        "legacy": checkpoint.legacy,
        "tensor_count": len(checkpoint.training_state),
        "target_count": len(checkpoint.training_state) // 2,
        "storage": "raw",
        "resume_kind": "weights-only",
    }
