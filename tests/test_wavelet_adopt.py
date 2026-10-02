"""Feature contract: real adv_optm and ai-toolkit-compatible Haar wavelet loss."""

import pytest
import torch

from qwen21_trainer.config import Config, load_config


def config(tmp_path, **training):
    return Config(path=tmp_path / "configs/test.toml", training=training)


def test_baseline_defaults_are_preserved(tmp_path):
    cfg = config(tmp_path)
    assert cfg.training["loss_type"] == "mse"
    assert cfg.training["loss_weighting"] == "diffsynth"
    assert cfg.training["optimizer"] == "adamw"
    assert cfg.training["optimizer_params"] == {}


def test_adopt_toml_params_and_snapshot_are_independent(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("""[training]
loss_type = "wavelet"
loss_weighting = "diffsynth"
optimizer = "Adopt_adv"
[training.optimizer_params]
cautious_wd = true
kourkoutas_beta = true
use_atan2 = true
betas = [0.9, 0.9999]
""")
    cfg = load_config(p)
    assert cfg.training["optimizer"] == "adopt_adv"
    snapshot = cfg.to_dict()
    snapshot["training"]["optimizer_params"]["cautious_wd"] = False
    assert cfg.training["optimizer_params"]["cautious_wd"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"loss_type": "wavlet"},
        {"loss_weighting": "auto"},
        {"optimizer": "adot_adv"},
        {"optimizer_params": []},
        {"optimizer_params": {"lr": 0.1}},
        {"optimizer_params": {"weight_decay": 0.1}},
        {"optimizer_params": {"cautious_wd": True}},  # invalid for baseline AdamW
        {"optimizer": "adopt_adv", "optimizer_params": {"cautious_wd": 1}},
        {"optimizer": "adopt_adv", "optimizer_params": {"not_a_parameter": True}},
        {"optimizer": "adopt_adv", "optimizer_params": {"betas": [0.9, 1.0]}},
        {"optimizer": "adopt_adv", "optimizer_params": {"betas": [True, 0.9999]}},
        {"optimizer": "adopt_adv", "optimizer_params": {"eps": float("nan")}},
        {"optimizer": "adopt_adv", "optimizer_params": {"state_precision": "made_up"}},
        {
            "optimizer": "adopt_adv",
            "optimizer_params": {"beta2_min": 0.99999, "kourkoutas_beta": True},
        },
    ],
)
def test_invalid_training_options_fail_early(tmp_path, kwargs):
    with pytest.raises(ValueError):
        config(tmp_path, **kwargs)


def reference_wavelet(pred, clean, noise):
    # Same operations as local ai-toolkit/toolkit/util/losses.py, no global DWT cache.
    from pytorch_wavelets import DWTForward

    dwt = DWTForward(J=1, mode="zero", wave="haar").float()
    with torch.no_grad():
        lo, hi = dwt(clean.float())
        target = torch.cat([lo, *torch.unbind(hi[0], dim=2)], dim=1)
    lo, hi = dwt(noise.float() - pred.float())
    actual = torch.cat([lo, *torch.unbind(hi[0], dim=2)], dim=1)
    return torch.nn.functional.mse_loss(actual, target)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 4, 4, 6), (2, 3, 5, 7)])
def test_wavelet_value_and_gradient_matches_aitk(dtype, shape):
    from qwen21_trainer.losses import HaarWaveletLoss

    torch.manual_seed(9)
    pred = torch.randn(shape, dtype=dtype, requires_grad=True)
    clean, noise = torch.randn_like(pred), torch.randn_like(pred)
    reference_pred = pred.detach().clone().requires_grad_()
    value = HaarWaveletLoss()(pred, clean, noise)
    reference = reference_wavelet(reference_pred, clean, noise)
    torch.testing.assert_close(value, reference, rtol=1e-6, atol=1e-6)
    value.backward()
    reference.backward()
    torch.testing.assert_close(pred.grad, reference_pred.grad, rtol=1e-6, atol=1e-6)
    assert value.dtype == torch.float32
    assert torch.isfinite(pred.grad).all()


def test_haar_equal_band_even_shape_parseval_equivalence():
    from qwen21_trainer.losses import HaarWaveletLoss

    torch.manual_seed(13)
    pred, clean, noise = (torch.randn(2, 3, 8, 12) for _ in range(3))
    wavelet = HaarWaveletLoss()(pred, clean, noise)
    mse = torch.nn.functional.mse_loss(pred, noise - clean)
    torch.testing.assert_close(wavelet, mse, rtol=1e-6, atol=1e-6)


def test_real_adopt_constructor_and_state_updates(tmp_path):
    from adv_optm import Adopt_adv

    from qwen21_trainer.optimizers import build_optimizer

    cfg = config(
        tmp_path,
        optimizer="adopt_adv",
        optimizer_params={
            "cautious_wd": True,
            "kourkoutas_beta": True,
            "use_atan2": True,
        },
    )
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
    opt = build_optimizer([p], cfg.training)
    assert isinstance(opt, Adopt_adv)
    assert opt.use_atan2 and opt.kourkoutas_beta
    assert opt.param_groups[0]["cautious_wd"]
    before = p.detach().clone()
    # Use a visible test LR, not a claim about pretrained training dynamics.
    opt.param_groups[0]["lr"] = 0.1
    for _ in range(4):
        (p.float().square().sum()).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    assert not torch.equal(before, p)
    assert opt.state[p]
    assert torch.isfinite(p).all()


def test_mse_default_keeps_exact_upstream_loss():
    from types import SimpleNamespace

    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    from diffsynth.diffusion.loss import FlowMatchSFTLoss

    from qwen21_trainer.losses import flow_matching_loss

    scheduler = FlowMatchScheduler("Qwen-Image")
    scheduler.set_timesteps(1000, training=True)
    pipe = SimpleNamespace(
        scheduler=scheduler,
        torch_dtype=torch.float32,
        device="cpu",
        in_iteration_models=(),
        model_fn=lambda latents, **kw: latents * 0.25,
    )
    inputs = {"input_latents": torch.ones(1, 4, 4, 4)}
    torch.manual_seed(18)
    expected = FlowMatchSFTLoss(pipe, **inputs)
    torch.manual_seed(18)
    actual = flow_matching_loss(pipe, inputs, loss_type="mse", weighting="diffsynth")
    assert torch.equal(expected, actual)


def test_none_weighting_does_not_call_training_weight():
    from types import SimpleNamespace

    from diffsynth.diffusion.flow_match import FlowMatchScheduler

    from qwen21_trainer.losses import HaarWaveletLoss, flow_matching_loss

    scheduler = FlowMatchScheduler("Qwen-Image")
    scheduler.set_timesteps(1000, training=True)

    def fail(*args):
        raise AssertionError("weighting disabled")

    scheduler.training_weight = fail
    pipe = SimpleNamespace(
        scheduler=scheduler,
        torch_dtype=torch.float32,
        device="cpu",
        in_iteration_models=(),
        model_fn=lambda latents, **kw: latents * 0.25,
    )
    value = flow_matching_loss(
        pipe,
        {"input_latents": torch.ones(1, 4, 4, 4)},
        loss_type="wavelet",
        weighting="none",
        wavelet=HaarWaveletLoss(),
    )
    assert torch.isfinite(value)


@pytest.mark.parametrize(
    "loss_type,optimizer",
    [
        ("mse", "adopt_adv"),
        ("wavelet", "adamw"),
        ("wavelet", "adopt_adv"),
    ],
)
def test_real_tiny_model_feature_combinations(tmp_path, loss_type, optimizer):
    from qwen21_trainer.smoke import run_smoke

    options = {"loss_type": loss_type, "optimizer": optimizer, "loss_weighting": "diffsynth"}
    if optimizer == "adopt_adv":
        options["optimizer_params"] = {
            "cautious_wd": True,
            "kourkoutas_beta": True,
            "use_atan2": True,
        }
    report = run_smoke(tmp_path / "smoke", training_options=options)
    assert len(report["runs"]) == 2
    for run in report["runs"]:
        assert run["optimizer"]["name"] == optimizer
        assert run["loss_type"] == loss_type
        assert run["optimizer_updates"] == 3
        assert run["changed_lora_tensors"] > 0
        assert run["reload_max_abs_error"] == 0
        if optimizer == "adopt_adv":
            assert run["optimizer"]["class"].startswith("adv_optm.")
            assert run["optimizer"]["adopt_use_atan2"]
            assert run["optimizer"]["adopt_kourkoutas_beta"]
            assert run["optimizer"]["effective_cautious_wd"]
