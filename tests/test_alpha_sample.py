"""Verify the public sample wrapper uses checkpoint alpha exactly once."""

import copy
from pathlib import Path

import pytest
import torch
from PIL import Image
from test_training_sampling import TinyFixtureEncoder, TinyFixtureProcessor, TinyFixtureVAE

from qwen21_trainer import runtime
from qwen21_trainer.config import Config
from qwen21_trainer.lora_io import read_checkpoint


@pytest.mark.parametrize("alpha", [16, 16.123456789012345])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_standalone_sample_uses_file_alpha_not_config_alpha(tmp_path, monkeypatch, alpha, dtype):
    from diffsynth.models.qwen_image_21_dit import QwenImage21DiT
    from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline

    torch.manual_seed(8)
    shape = dict(
        in_channels=64,
        out_channels=64,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=32,
        context_in_dim=64,
        axes_dims_rope=(8, 12, 12),
    )
    train_pipe = QwenImage21Pipeline(device="cpu", torch_dtype=dtype)
    train_pipe.dit = QwenImage21DiT(**shape).to(dtype=dtype)
    pristine = copy.deepcopy(train_pipe.dit.state_dict())
    model = runtime.make_training_module(
        train_pipe,
        {
            "rank": 32,
            "alpha": alpha,
            "gradient_checkpointing": False,
            "checkpointing_offload": False,
        },
        targets="to_q,to_k",
    )
    with torch.no_grad():
        for parameter in model.trainable_modules():
            parameter.normal_(std=0.1)
    path = tmp_path / "lora.safetensors"
    runtime.save_lora(model, path, {"rank": 32, "alpha": alpha})
    checkpoint = read_checkpoint(path)
    inference = QwenImage21Pipeline(device="cpu", torch_dtype=dtype)
    inference.dit = QwenImage21DiT(**shape).to(dtype=dtype)
    inference.dit.load_state_dict(pristine)
    inference.text_encoder = TinyFixtureEncoder().to(dtype=dtype)
    inference.processor = TinyFixtureProcessor()
    inference.vae = TinyFixtureVAE()
    modules = dict(inference.dit.named_modules())
    expected = {}
    for key, a in checkpoint.training_state.items():
        if key.endswith(".lora_A.default.weight"):
            target = key.removesuffix(".lora_A.default.weight")
            b = checkpoint.training_state[f"{target}.lora_B.default.weight"]
            scale = torch.tensor(alpha, dtype=torch.float64) / 32
            expected[target] = modules[target].weight.detach().clone() + b @ (a * scale)
    calls = []
    load = inference.load_lora

    def checked_load(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs.get("alpha") == 1.0 and "state_dict" in kwargs
        assert not kwargs.get("lora_config")
        return load(*args, **kwargs)

    inference.load_lora = checked_load
    monkeypatch.setattr(runtime, "build_pipeline", lambda *a, **kw: inference)
    monkeypatch.setattr(runtime, "model_assets", lambda cfg: {})
    cfg = Config(
        path=tmp_path / "config.toml",
        model={"device": "cpu"},
        training={"rank": 32, "alpha": 32},
        sample={"width": 64, "height": 64, "steps": 1, "cfg_scale": 1.0, "prompt": "fixture"},
    )
    result = runtime.sample(cfg, lora=path, output=tmp_path / "preview.png")
    assert len(calls) == 1
    for name, value in expected.items():
        assert torch.equal(modules[name].weight, value)
    with Image.open(result["image"]) as image:
        assert image.size == (64, 64)
    assert Path(result["image"]).is_file()
    assert not torch.cuda.is_initialized()
