"""Actual tiny DiT updates match with/without spawned CPU prefetch."""

import pytest
import torch
from safetensors.torch import load_file

from qwen21_trainer.smoke import run_smoke


@pytest.mark.parametrize("loss_type,optimizer", [("mse", "adamw"), ("wavelet", "adopt_adv")])
def test_prefetch_preserves_tiny_losses_and_saved_lora(tmp_path, loss_type, optimizer):
    options = {"loss_type": loss_type, "optimizer": optimizer, "prefetch_factor": 2}
    if optimizer == "adopt_adv":
        options["optimizer_params"] = {
            "cautious_wd": True,
            "kourkoutas_beta": True,
            "use_atan2": True,
        }
    plain = run_smoke(tmp_path / "plain", training_options={**options, "num_workers": 0})
    prefetched = run_smoke(tmp_path / "prefetched", training_options={**options, "num_workers": 2})
    for a, b in zip(plain["runs"], prefetched["runs"]):
        assert a["dtype"] == b["dtype"]
        assert a["losses"] == b["losses"]
        assert b["num_workers"] == 2
        assert a["optimizer_updates"] == b["optimizer_updates"] == 3
        assert all(t >= 0 for t in b["data_wait_seconds"])
        first, second = load_file(a["checkpoint"]), load_file(b["checkpoint"])
        assert first.keys() == second.keys()
        assert all(torch.equal(first[k], second[k]) for k in first)
