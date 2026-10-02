"""Configuration and actual PEFT forward/backward alpha contract."""

import pytest
import torch
from torch.nn import functional as F

from qwen21_trainer.config import Config, load_config
from qwen21_trainer.runtime import make_training_module


def test_alpha_defaults_to_rank_and_yaml_toml_preserve(tmp_path):
    assert Config(path=tmp_path / "c.toml", training={"rank": 48}).training["alpha"] == 48
    for extension, content in [
        ("toml", "[training]\nrank=32\nalpha=16\n"),
        ("yaml", "training: {rank: 32, alpha: 16}\n"),
    ]:
        path = tmp_path / f"config.{extension}"
        path.write_text(content)
        cfg = load_config(path)
        assert cfg.training["rank"] == 32 and cfg.training["alpha"] == 16
        assert cfg.to_dict()["training"]["alpha"] == 16
        assert cfg.source_document["training"]["alpha"] == 16


@pytest.mark.parametrize("alpha", [0, -1, True, False, "16", float("nan"), float("inf")])
def test_bad_alpha_rejected(tmp_path, alpha):
    with pytest.raises(ValueError, match="alpha"):
        Config(path=tmp_path / "c.toml", training={"rank": 32, "alpha": alpha})


@pytest.mark.parametrize("alpha", [1, 12.5, 16, 32, 64])
def test_alpha_is_independent_not_a_sentinel(tmp_path, alpha):
    cfg = Config(path=tmp_path / "c.toml", training={"rank": 32, "alpha": alpha})
    assert cfg.training["alpha"] == alpha


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_peft_forward_and_backward_use_alpha_16_rank_32(dtype):
    from diffsynth.models.qwen_image_21_dit import QwenImage21DiT
    from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline

    torch.manual_seed(17)
    pipe = QwenImage21Pipeline(device="cpu", torch_dtype=dtype)
    pipe.dit = QwenImage21DiT(
        in_channels=4,
        out_channels=4,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=32,
        context_in_dim=64,
        axes_dims_rope=(8, 12, 12),
    ).to(dtype=dtype)
    model = make_training_module(
        pipe,
        {"rank": 32, "alpha": 16, "gradient_checkpointing": False, "checkpointing_offload": False},
        targets="to_q,to_k",
    )
    layers = [m for m in pipe.dit.modules() if hasattr(m, "lora_A")]
    assert layers and pipe.dit.peft_config["default"].lora_alpha == 16
    assert all(
        m.r["default"] == 32 and m.lora_alpha["default"] == 16 and m.scaling["default"] == 0.5
        for m in layers
    )
    layer = layers[0]
    a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
    with torch.no_grad():
        a.normal_(std=0.05)
        b.normal_(std=0.05)
    x = torch.randn(2, layer.in_features, dtype=dtype)
    expected_a, expected_b = (
        a.detach().clone().requires_grad_(),
        b.detach().clone().requires_grad_(),
    )
    base = F.linear(x, layer.base_layer.weight, layer.base_layer.bias)
    expected = base + F.linear(F.linear(x, expected_a), expected_b) * (16 / 32)
    actual = layer(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    torch.testing.assert_close(a.grad, expected_a.grad, rtol=0, atol=0)
    torch.testing.assert_close(b.grad, expected_b.grad, rtol=0, atol=0)
    assert layer.base_layer.weight.grad is None
    assert all("lora_" in name for name, p in model.named_parameters() if p.requires_grad)


def test_alpha_does_not_change_dataset_cache_identity(tmp_path):
    from PIL import Image

    from qwen21_trainer.data import load_manifest, prepare_dataset

    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (64, 64)).save(images / "a.png")
    (images / "a.txt").write_text("fixture")
    shared = {"cache_dir": str(tmp_path / "cache"), "output_dir": str(tmp_path / "out"), "rank": 32}
    a = Config(path=tmp_path / "config.toml", dataset={"path": str(images)}, training=shared)
    b = Config(
        path=tmp_path / "config.toml",
        dataset={"path": str(images)},
        training={**shared, "alpha": 16},
    )
    assert prepare_dataset(a)["dataset_fingerprint"] == load_manifest(b)["dataset_fingerprint"]
