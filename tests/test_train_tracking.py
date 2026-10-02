"""Train-to-tracker wiring with real optimizer and explicit local test doubles."""

import json
import random
import sys
from types import ModuleType

import pytest
import torch
from safetensors.torch import save_file

from qwen21_trainer import data, runtime
from qwen21_trainer.config import Config


class ToyModel(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.randn(1))
        self.fail = fail

    def trainable_modules(self):
        return self.parameters()

    def forward(self, sample):
        if self.fail:
            raise RuntimeError("model fixture failure")
        return (self.lora_A - sample["target"]).square().mean()


def fixture(tmp_path, monkeypatch, fail=False):
    cfg = Config(
        path=tmp_path / "cfg.toml",
        model={"device": "cpu"},
        training={
            "cache_dir": str(tmp_path / "cache"),
            "output_dir": str(tmp_path / "out"),
            "max_steps": 3,
        },
        wandb={"enabled": True, "mode": "offline"},
    )
    folder = tmp_path / "cache"
    folder.mkdir()
    save_file({"target": torch.zeros(1)}, str(folder / "one.safetensors"))
    monkeypatch.setattr(data, "load_manifest", lambda cfg: {"dataset_fingerprint": "fixture"})
    monkeypatch.setattr(runtime, "model_assets", lambda cfg: {})
    monkeypatch.setattr(
        runtime,
        "load_cache_index",
        lambda *a: {
            "identity": {"model_signature": "fixture"},
            "items": [{"file": "one.safetensors"}],
        },
    )
    monkeypatch.setattr(runtime, "build_pipeline", lambda *a, **kw: object())
    events, rows = [], []

    def model(*a, **kw):
        events.append("model")
        value = ToyModel(fail=fail)
        expected = torch.randn(1, generator=torch.Generator().manual_seed(cfg.training["seed"]))
        assert torch.equal(value.lora_A.detach(), expected)
        return value

    monkeypatch.setattr(runtime, "make_training_module", model)
    monkeypatch.setattr(
        runtime,
        "save_lora",
        lambda model, path, meta: save_file({"lora_A": model.lora_A.detach()}, str(path)),
    )
    module = ModuleType("qwen21_trainer.tracking")

    class FakeTracker:
        def __init__(self, cfg, plan, output_dir):
            self.info = {"enabled": cfg.wandb["enabled"], "mode": "offline", "run_id": "fixture"}

        def __enter__(self):
            events.append("enter")
            # SDK setup must occur before the trainer's manual_seed.
            torch.rand(7)
            random.random()
            return self

        def log(self, row):
            rows.append(dict(row))

        def __exit__(self, exc_type, exc, tb):
            events.append(("exit", exc_type))
            return False

    module.TrainingTracker = FakeTracker
    monkeypatch.setitem(sys.modules, "qwen21_trainer.tracking", module)
    return cfg, events, rows


def test_train_logs_steps_and_closes_tracking_after_model(tmp_path, monkeypatch):
    cfg, events, rows = fixture(tmp_path, monkeypatch)
    result = runtime.train(cfg)
    assert events == ["enter", "model", ("exit", None)]
    assert [row["step"] for row in rows] == [1, 2, 3]
    assert all("data_wait_seconds" in row for row in rows)
    assert result["optimizer_updates"] == 3
    saved = json.loads((tmp_path / "out/run.json").read_text())
    assert saved["wandb_runtime"]["run_id"] == "fixture"


def test_training_exception_reaches_tracker_exit(tmp_path, monkeypatch):
    cfg, events, rows = fixture(tmp_path, monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match="model fixture failure"):
        runtime.train(cfg)
    assert events == ["enter", "model", ("exit", RuntimeError)]
    assert rows == []
    assert not (tmp_path / "out/completed.json").exists()


def test_dry_run_does_not_initialize_tracking(tmp_path, monkeypatch):
    cfg, events, rows = fixture(tmp_path, monkeypatch)
    result = runtime.train(cfg, dry_run=True)
    assert result["optimizer_updates"] == 3
    assert events == rows == []
    assert not (tmp_path / "out").exists()
