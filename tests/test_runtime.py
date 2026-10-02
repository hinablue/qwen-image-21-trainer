from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from qwen21_trainer.runtime import (
    _cache_identity,
    model_assets,
    optimization_loop,
    source_signature,
    validate_tensors,
)


def settings(**overrides):
    result = {
        "rank": 2,
        "learning_rate": 0.01,
        "weight_decay": 0.01,
        "max_steps": 4,
        "gradient_accumulation_steps": 3,
        "seed": 42,
        "gradient_checkpointing": True,
        "checkpointing_offload": False,
    }
    result.update(overrides)
    return result


class ScalarModel(torch.nn.Module):
    def __init__(self, nonfinite=False):
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = 0
        self.nonfinite = nonfinite

    def trainable_modules(self):
        return self.parameters()

    def forward(self, sample):
        self.calls += 1
        return self.p * float("nan") if self.nonfinite else (self.p - sample) ** 2


def test_exact_step_budget_and_accumulation():
    model = ScalarModel()
    rows = optimization_loop(model, [torch.tensor(2.0), torch.tensor(3.0)], settings())
    assert model.calls == 12
    assert len(rows) == 4
    assert rows[-1]["images_seen"] == 12
    assert model.p.item() != 1.0
    assert rows[0]["learning_rate"] == pytest.approx(0.01 / 3)


def test_nan_fails_before_optimizer_update():
    model = ScalarModel(nonfinite=True)
    with pytest.raises(FloatingPointError):
        optimization_loop(model, [torch.tensor(1.0)], settings())
    assert model.p.item() == 1.0


def test_empty_samples_rejected():
    with pytest.raises(ValueError):
        optimization_loop(ScalarModel(), [], settings())


def tensors():
    return {
        "input_latents": torch.ones(1, 64, 4, 4),
        "prompt_embeds": torch.ones(1, 5, 4096),
        "edit_image_pad_mask": torch.zeros(1, 5, dtype=torch.bool),
    }


def test_cache_tensors_valid():
    validate_tensors(tensors())


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_latents", torch.ones(1, 16, 4, 4)),
        ("input_latents", torch.ones(1, 64, 3, 4)),
        ("prompt_embeds", torch.full((1, 5, 4096), float("inf"))),
        ("edit_image_pad_mask", torch.ones(1, 5, dtype=torch.bool)),
        ("edit_image_pad_mask", torch.zeros(1, 5)),
    ],
)
def test_invalid_cache_tensors(field, value):
    item = tensors()
    item[field] = value
    with pytest.raises(ValueError):
        validate_tensors(item)


def write_model_fixture(tmp_path, key="transformer_blocks.0.img_mlp.gate_layer.weight"):
    # Header-only validation fixtures, deliberately NOT real pretrained checkpoints.
    for role in ["transformer", "text_encoder", "vae", "processor"]:
        (tmp_path / role).mkdir()
    (tmp_path / "processor/tokenizer_config.json").write_text("{}")
    for role in ["transformer", "text_encoder", "vae"]:
        name = (
            "model.safetensors" if role == "text_encoder" else "diffusion_pytorch_model.safetensors"
        )
        save_file({key: torch.ones(2, 2, dtype=torch.bfloat16)}, str(tmp_path / role / name))
    return SimpleNamespace(model={"root": tmp_path})


def test_official_bf16_header_and_source_signature(tmp_path):
    cfg = write_model_fixture(tmp_path)
    assets = model_assets(cfg)
    first = source_signature(cfg, assets)
    (tmp_path / "processor/tokenizer_config.json").write_text('{"changed":true}')
    assert source_signature(cfg, assets) != first


def test_fused_checkpoint_rejected(tmp_path):
    cfg = write_model_fixture(tmp_path, "transformer_blocks.0.img_mlp.gate_up.weight")
    with pytest.raises(ValueError, match="split-MLP"):
        model_assets(cfg)


def test_missing_weights_never_downloads(tmp_path):
    with pytest.raises(FileNotFoundError):
        model_assets(SimpleNamespace(model={"root": tmp_path}))


def test_cache_identity_contains_data_and_model():
    first = _cache_identity({"dataset_fingerprint": "abc"}, "model1")
    assert first != _cache_identity({"dataset_fingerprint": "changed"}, "model1")
    assert first != _cache_identity({"dataset_fingerprint": "abc"}, "model2")
    assert first["latent_mode"] == "posterior_mean"


def test_actual_upstream_smoke(tmp_path):
    from qwen21_trainer.smoke import run_smoke

    result = run_smoke(tmp_path / "smoke")
    assert result["production_auto_targets"] == 224
    assert not result["cuda_visible"]
    assert all(x["reload_max_abs_error"] == 0 for x in result["runs"])
    assert all(x["optimizer_updates"] == 3 for x in result["runs"])
