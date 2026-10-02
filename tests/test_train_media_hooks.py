"""Train callback ordering; these are explicit sampler/tracker test doubles."""

import sys
from types import ModuleType

from safetensors.torch import save_file
from test_train_tracking import fixture

from qwen21_trainer import runtime


def test_sample_and_scalars_are_one_commit_after_checkpoint(tmp_path, monkeypatch):
    cfg, _, _ = fixture(tmp_path, monkeypatch)
    cfg.sample.update(enabled=True, every=2)
    cfg.training["save_every"] = 2
    events = []
    tracker_module = ModuleType("qwen21_trainer.tracking")
    sampler_module = ModuleType("qwen21_trainer.sampling")

    class Tracker:
        info = {"enabled": True}

        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def log(self, row, *, samples=None):
            events.append(("log", row["step"], bool(samples)))

    class Sampler:
        def __init__(self, *args):
            pass

        def should_sample(self, step):
            return step in (2, 3)

        def generate(self, step):
            events.append(("sample", step))
            return [{"step": step, "fixture_only": True}]

    def save(model, path, meta):
        events.append(("save", meta["step"]))
        save_file({"lora_A": model.lora_A.detach()}, str(path))

    tracker_module.TrainingTracker = Tracker
    sampler_module.TrainingSampler = Sampler
    monkeypatch.setitem(sys.modules, "qwen21_trainer.tracking", tracker_module)
    monkeypatch.setitem(sys.modules, "qwen21_trainer.sampling", sampler_module)
    monkeypatch.setattr(runtime, "save_lora", save)
    runtime.train(cfg)
    assert events == [
        ("log", 1, False),
        ("save", 2),
        ("sample", 2),
        ("log", 2, True),
        ("sample", 3),
        ("log", 3, True),
        ("save", 3),
    ]


def test_disabled_sampling_does_not_construct_sampler(tmp_path, monkeypatch):
    cfg, _, _ = fixture(tmp_path, monkeypatch)
    module = ModuleType("qwen21_trainer.sampling")

    def reject(*args):
        raise AssertionError("disabled sampler must not be constructed")

    module.TrainingSampler = reject
    monkeypatch.setitem(sys.modules, "qwen21_trainer.sampling", module)
    assert runtime.train(cfg)["optimizer_updates"] == 3
