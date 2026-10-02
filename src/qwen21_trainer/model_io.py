"""Local-only model loading; Comfy BF16 is adapted in memory, never rewritten.

The adapter changes checkpoint layout only. Architecture, processor, scheduler,
loss and model loading/strict assignment remain the pinned DiffSynth code.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import UPSTREAM_REVISION

ROLES = ("transformer", "text_encoder", "vae")
ATTRS = {"transformer": "dit", "text_encoder": "text_encoder", "vae": "vae"}
COMFY_FILES = {
    "transformer": "diffusion_models/qwen_image_2.1_bf16.safetensors",
    "text_encoder": "text_encoders/qwen3vl_8b_bf16.safetensors",
    "vae": "vae/qwen_image_2.1_vae_bf16.safetensors",
}
# Increment when mapping semantics change: old latent/text caches must not survive.
ADAPTER_VERSION = "qwen21-local-model-io-v1"
_QUANT = re.compile(
    r"(?:^|[^a-z0-9])(quant(?:ized|ization|_config)?|comfy_quant|qweight|qzeros|"
    r"weight_scale(?:_2)?|scale_weight|input_scale|scales|fp8|fp4|nf4|int8|int4|gguf)"
    r"(?:$|[^a-z0-9])",
    re.IGNORECASE,
)


def _format(cfg):
    value = cfg.model.get("format", "official")
    if value not in {"official", "comfy_bf16"}:
        raise ValueError(f"不支援 model.format：{value!r}")
    return value


def _header(path):
    """Read descriptors, not tensor payloads; safetensors validates offsets."""
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as handle:
        entries = {
            key: {
                "shape": tuple(handle.get_slice(key).get_shape()),
                "dtype": handle.get_slice(key).get_dtype(),
            }
            for key in handle.keys()
        }
        metadata = handle.metadata() or {}
    return entries, metadata


def _require_bf16(entries, metadata, path):
    for key, value in entries.items():
        if _QUANT.search(key) or value["dtype"] != "BF16":
            raise ValueError(f"comfy_bf16 只支援未量化 BF16 權重：{path}:{key} ({value['dtype']})")
    for key, value in metadata.items():
        if _QUANT.search(key) or _QUANT.search(str(value)):
            raise ValueError(f"comfy_bf16 拒絕量化 metadata：{path}:{key}")
    if not entries:
        raise ValueError(f"空白 checkpoint：{path}")


def _vae_key(key):
    """Wan/Comfy Qwen2.1 VAE names to pinned DiffSynth 2D VAE names."""
    resnet = {
        "residual.0": "norm1",
        "residual.2": "conv1",
        "residual.3": "norm2",
        "residual.6": "conv2",
        "shortcut": "conv_shortcut",
    }

    def residual(inner):
        n = 2 if inner[0] == "residual" else 1
        return [resnet[".".join(inner[:n])], *inner[n:]]

    parts = key.split(".")
    try:
        if parts[0] in {"conv1", "conv2"} and parts[1:] in [["weight"], ["bias"]]:
            return ".".join(
                [{"conv1": "quant_conv", "conv2": "post_quant_conv"}[parts[0]], *parts[1:]]
            )
        side, rest = parts[0], parts[1:]
        if side not in {"encoder", "decoder"}:
            raise ValueError
        if rest[0] == "conv1":
            target = [side, "conv_in", *rest[1:]]
        elif rest[0] == "head" and rest[1] in {"0", "2"}:
            target = [side, {"0": "norm_out", "2": "conv_out"}[rest[1]], *rest[2:]]
        elif rest[0] == "middle":
            part = {"0": "resnets.0", "1": "attentions.0", "2": "resnets.1"}[rest[1]]
            inner = residual(rest[2:]) if part.startswith("resnets") else rest[2:]
            target = [side, "mid_block", part, *inner]
        else:
            group = "downsamples" if side == "encoder" else "upsamples"
            if (
                rest[0] != group
                or rest[2] != group
                or not rest[1].isdigit()
                or not rest[3].isdigit()
            ):
                raise ValueError
            block = "down_blocks" if side == "encoder" else "up_blocks"
            sampler = "downsampler" if side == "encoder" else "upsampler"
            stage, index, inner = rest[1], rest[3], rest[4:]
            if inner[0] in {"resample", "time_conv"}:
                # Samplers follow two encoder / three decoder residual blocks.
                if index != ("2" if side == "encoder" else "3"):
                    raise ValueError
                target = [side, block, stage, sampler, *inner]
            else:
                target = [side, block, stage, "resnets", index, *residual(inner)]
        return ".".join(target)
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(f"不明 Comfy VAE key：{key}") from exc


@dataclass(frozen=True)
class TensorView:
    source: str
    shape: tuple[int, ...]
    rows: tuple[int, int] | None = None
    squeeze: bool = False

    def apply(self, tensor):
        if self.rows is not None:
            tensor = tensor[self.rows[0] : self.rows[1]]
        if self.squeeze:
            tensor = tensor.squeeze(2)
        return tensor


def comfy_mapping_plan(role, entries):
    """Map every key once, without reading tensors or allocating model weights.

    TE output is the official HF namespace, NOT the DiffSynth wrapper namespace;
    the official converter adds its wrapper prefix exactly once at loading time.
    Full architecture validation is performed separately by validate_comfy_headers.
    """
    if role not in ROLES:
        raise ValueError(f"不明模型 role：{role}")
    plan = {}

    def add(target, view):
        if target in plan:
            raise ValueError(f"重複 mapping target：{role}:{target}")
        plan[target] = view

    for key, descriptor in entries.items():
        shape = tuple(descriptor["shape"])
        if role == "transformer":
            if key.endswith(".img_mlp.gate_up.weight"):
                if len(shape) != 2 or shape[0] % 2 or shape[0] == 0:
                    raise ValueError(f"gate_up 必須是偶數 rows 的 2D tensor：{key}:{shape}")
                half = shape[0] // 2
                for name, rows in (("gate_layer", (0, half)), ("proj", (half, shape[0]))):
                    target = key.removesuffix("gate_up.weight") + name + ".weight"
                    add(target, TensorView(key, (half, shape[1]), rows=rows))
            else:
                add(key, TensorView(key, shape))
        elif role == "text_encoder":
            if key.startswith(("model.layers.", "model.embed_tokens.", "model.norm.")):
                target = "model.language_model." + key.removeprefix("model.")
            elif key == "lm_head.weight" or key.startswith("model.visual."):
                target = key
            else:
                raise ValueError(f"不明 Comfy text_encoder key：{key}")
            add(target, TensorView(key, shape))
        else:
            squeeze = len(shape) == 5
            if squeeze and shape[2] != 1:
                raise ValueError(f"VAE 只可移除 size-1 temporal axis：{key}:{shape}")
            target_shape = (*shape[:2], *shape[3:]) if squeeze else shape
            add(_vae_key(key), TensorView(key, target_shape, squeeze=squeeze))
    return plan


def _wrapper_state(role, state):
    if role == "text_encoder":
        from diffsynth.utils.state_dict_converters.qwen_image_21_text_encoder import (
            QwenImage21TextEncoderStateDictConverter,
        )

        converted = QwenImage21TextEncoderStateDictConverter(state)
        if len(converted) != len(state):
            raise ValueError("text_encoder 官方 wrapper converter 產生 key collision")
        return converted
    return state


@lru_cache(maxsize=3)
def _expected_shapes(role):
    import torch
    from diffsynth.models.qwen_image_21_dit import QwenImage21DiT
    from diffsynth.models.qwen_image_21_text_encoder import QwenImage21TextEncoder
    from diffsynth.models.qwen_image_21_vae import QwenImage21VAE

    cls = {
        "transformer": QwenImage21DiT,
        "text_encoder": QwenImage21TextEncoder,
        "vae": QwenImage21VAE,
    }[role]
    # Shape-only construction. Do NOT use this meta-buffer instance for real loads:
    # the official loader keeps nonpersistent rotary buffers on CPU instead.
    with torch.device("meta"):
        model = cls()
    return {key: tuple(value.shape) for key, value in model.state_dict().items()}


def _assert_shapes(role, actual):
    expected = _expected_shapes(role)
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    wrong = {
        key: (actual[key], expected[key])
        for key in expected.keys() & actual.keys()
        if actual[key] != expected[key]
    }
    if missing or unexpected or wrong:
        raise ValueError(
            f"{role} 不符合 pinned Qwen-Image-2.1 完整 key/shape："
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, shape_mismatches={dict(list(wrong.items())[:8])}"
        )


def validate_comfy_headers(role, paths):
    """Fail closed for quantized, mixed dtype, partial or unknown checkpoints."""
    if role not in ROLES or len(paths) != 1:
        raise ValueError("comfy_bf16 每個已知 role 必須恰好一個 checkpoint")
    entries, metadata = _header(paths[0])
    _require_bf16(entries, metadata, paths[0])
    plan = comfy_mapping_plan(role, entries)
    if role == "transformer":
        if not any(view.rows for view in plan.values()):
            raise ValueError("comfy_bf16 transformer 未找到 fused gate_up，拒絕不明來源格式")
        if any(".img_mlp.gate_layer." in key or ".img_mlp.proj." in key for key in entries):
            raise ValueError("comfy_bf16 transformer 拒絕混用 split/fused MLP 來源格式")
    shapes = _wrapper_state(role, {key: view.shape for key, view in plan.items()})
    _assert_shapes(role, shapes)
    return plan


def model_assets(cfg):
    """Discover local assets; inspect all headers, never load unrequested tensors."""
    fmt = _format(cfg)
    root = Path(cfg.model["root"])
    assets = {"format": fmt}
    if fmt == "comfy_bf16":
        for role, relative in COMFY_FILES.items():
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f"缺少本機 Comfy BF16 模型：{path}（不會自動下載）")
            assets[role] = [path]
    else:
        for role in ROLES:
            pattern = (
                "model*.safetensors"
                if role == "text_encoder"
                else "diffusion_pytorch_model*.safetensors"
            )
            files = sorted((root / role).glob(pattern))
            if not files:
                raise FileNotFoundError(
                    f"缺少官方模型檔案：{root / role / pattern}。請設定 model.root 或執行 download。"
                )
            assets[role] = files
    processor = Path(cfg.model.get("processor_path") or root / "processor")
    if not processor.is_dir() or not (processor / "tokenizer_config.json").is_file():
        raise FileNotFoundError(f"缺少完整 processor 目錄：{processor}")
    assets["processor"] = processor
    if fmt == "comfy_bf16":
        for role in ROLES:
            validate_comfy_headers(role, assets[role])
    else:
        # Preserve the original official-folder contract and accepted dtypes.
        has_split_mlp = False
        for file in assets["transformer"]:
            entries, _ = _header(file)
            for key, value in entries.items():
                if ".img_mlp.gate_up." in key or key.endswith("comfy_quant"):
                    raise ValueError(
                        "官方模式需要 split-MLP 權重；Comfy fused BF16 請選 model.format=comfy_bf16。"
                    )
                has_split_mlp |= ".img_mlp.gate_layer.weight" in key
                if value["dtype"] not in {"BF16", "F16", "F32", "F64", "I64", "I32", "BOOL"}:
                    raise ValueError(f"不支援量化權重 dtype：{file.name}:{key}")
        if not has_split_mlp:
            raise ValueError("transformer 未找到 Qwen 2.1 split-MLP gate_layer，拒絕載入不明架構。")
    return assets


def source_signature(cfg, assets):
    """Versioned provenance: actual weight paths/stat, processor paths/content.

    A Comfy model root can contain unrelated models: never recursively hash that
    entire tree. External processor JSONs AND non-JSON tokenizer files are hashed.
    Weight content is intentionally not fully hashed (tens of GB per invocation).
    """
    fmt = _format(cfg)
    records = [["adapter", ADAPTER_VERSION], ["format", fmt], ["upstream", UPSTREAM_REVISION]]
    for role in ROLES:
        for path in assets[role]:
            path = Path(path)
            stat = path.stat()
            records.append([role, str(path.resolve()), stat.st_size, stat.st_mtime_ns])
    root = Path(cfg.model["root"])
    processor = Path(assets["processor"])
    records.append(["processor", str(processor.resolve())])
    configs = (
        {path for path in root.rglob("*.json") if ".cache" not in path.relative_to(root).parts}
        if fmt == "official"
        else set()
    )
    processor_files = {
        path
        for path in processor.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(processor).parts
    }
    # The processor itself often lives under ~/.cache/huggingface: exclude
    # nested cache bookkeeping only, never its absolute-path '.cache' ancestor.
    for path in sorted(configs | processor_files):
        records.append(
            [
                "content",
                str(path.absolute()),
                str(path.resolve()),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            ]
        )
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def load_comfy_state_dict(role, paths):
    """Read CPU/mmap tensors and return BF16 views; no casting, copy or save."""
    import torch
    from safetensors.torch import load_file

    before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    plan = validate_comfy_headers(role, paths)
    raw = load_file(str(paths[0]), device="cpu")
    mapped = {key: view.apply(raw[view.source]) for key, view in plan.items()}
    if any(
        value.dtype != torch.bfloat16 or tuple(value.shape) != plan[key].shape
        for key, value in mapped.items()
    ):
        raise ValueError(f"{role} tensor 與已驗證 header 不符")
    after = [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    if before != after:
        raise RuntimeError(f"載入時 checkpoint 發生變動：{paths}")
    return mapped


def _registry_config(role):
    from diffsynth.configs.model_configs import qwen_image_21_series

    name = "qwen_image_21_" + ATTRS[role]
    matches = [entry for entry in qwen_image_21_series if entry["model_name"] == name]
    if len(matches) != 1 or matches[0].get("quant_config"):
        raise RuntimeError(f"不相容的 DiffSynth registry：{name}")
    return matches[0]


def build_pipeline(cfg, assets, roles, *, training=False):
    import torch
    from diffsynth.core import ModelConfig
    from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline

    from .runtime import check_device

    roles = tuple(roles)
    if len(set(roles)) != len(roles) or any(role not in ROLES for role in roles):
        raise ValueError(f"重複或不明模型 roles：{roles}")
    fmt = _format(cfg)
    if assets.get("format", fmt) != fmt:
        raise ValueError("assets 與設定 model.format 不一致")
    device = check_device(cfg)
    load_device = "cpu" if training and cfg.model["cpu_offload"] else str(device)
    processor = (
        ModelConfig(path=str(assets["processor"]), skip_download=True)
        if "text_encoder" in roles
        else None
    )
    configs = (
        [ModelConfig(path=[str(p) for p in assets[role]], skip_download=True) for role in roles]
        if fmt == "official"
        else []
    )
    pipe = QwenImage21Pipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=load_device,
        model_configs=configs,
        processor_config=processor,
    )
    if fmt == "comfy_bf16":
        from diffsynth.models.model_loader import ModelPool

        # auto_load_model hashes the ORIGINAL FILE, ignoring state_dict for
        # detection. Passing mapped state_dict to ModelConfig alone cannot work.
        # Use the registry's explicit loader after complete architecture checks;
        # no global registry/hash monkeypatch and no synthetic checkpoint file.
        pool = ModelPool()
        for role in roles:
            state = load_comfy_state_dict(role, assets[role])
            vram = ModelConfig().vram_config()
            vram.update(computation_dtype=torch.bfloat16, computation_device=load_device)
            model = pool.load_model_file(
                _registry_config(role),
                [str(p) for p in assets[role]],
                vram,
                state_dict=state,
            )
            setattr(pipe, ATTRS[role], model)
            del state
        pipe.vram_management_enabled = pipe.check_vram_management_state()
    for role in roles:
        if getattr(pipe, ATTRS[role]) is None:
            raise RuntimeError(f"DiffSynth 無法辨識 {role} 權重，請確認是 Qwen-Image-2.1。")
    return pipe
