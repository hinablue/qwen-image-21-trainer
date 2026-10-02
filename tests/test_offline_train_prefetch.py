"""Real offline W&B plus real spawn CPU prefetch and a toy train process."""

import csv
import json

import torch
from safetensors.torch import save_file
from test_tracking import _offline_records
from test_train_tracking import ToyModel

from qwen21_trainer import data, runtime
from qwen21_trainer.config import Config


def test_offline_train_with_cpu_workers_records_real_losses(tmp_path, monkeypatch):
    import wandb

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_BASE_URL", "http://127.0.0.1:9")
    cfg = Config(
        path=tmp_path / "config.toml",
        model={"device": "cpu"},
        training={
            "cache_dir": str(tmp_path / "cache"),
            "output_dir": str(tmp_path / "out"),
            "max_steps": 3,
            "gradient_accumulation_steps": 2,
            "num_workers": 2,
        },
        wandb={"enabled": True, "mode": "offline", "name": "cpu-integration"},
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    save_file({"target": torch.zeros(1)}, str(cache / "one.safetensors"))
    monkeypatch.setattr(data, "load_manifest", lambda cfg: {"dataset_fingerprint": "fixture"})
    monkeypatch.setattr(runtime, "model_assets", lambda cfg: {})
    monkeypatch.setattr(
        runtime,
        "load_cache_index",
        lambda *args: {
            "identity": {"model_signature": "fixture"},
            "items": [{"file": "one.safetensors"}],
        },
    )
    monkeypatch.setattr(runtime, "build_pipeline", lambda *args, **kwargs: object())
    monkeypatch.setattr(runtime, "make_training_module", lambda *args, **kwargs: ToyModel())
    monkeypatch.setattr(
        runtime,
        "save_lora",
        lambda model, path, meta: save_file({"lora_A": model.lora_A.detach()}, str(path)),
    )
    summary = runtime.train(cfg)
    wandb.teardown()
    output = tmp_path / "out"
    journals = list(output.glob("wandb/offline-run-*/run-*.wandb"))
    assert len(journals) == 1  # no extra runs in CPU workers
    records = _offline_records(journals[0])
    history = [
        {
            item.key or ".".join(item.nested_key): json.loads(item.value_json)
            for item in record.history.item
        }
        for record in records
        if record.HasField("history")
    ]
    with (output / "metrics.csv").open() as f:
        local = list(csv.DictReader(f))
    assert [x["train/step"] for x in history] == [1, 2, 3]
    assert [x["train/loss"] for x in history] == [float(x["loss"]) for x in local]
    assert [x["train/images_seen"] for x in history] == [2, 4, 6]
    assert [x["train/data_wait_seconds"] for x in history] == [
        float(x["data_wait_seconds"]) for x in local
    ]
    assert [r.exit.exit_code for r in records if r.HasField("exit")] == [0]
    assert summary["optimizer_updates"] == 3
    assert not torch.cuda.is_initialized()
    (tmp_path / "train-offline-evidence.json").write_text(
        json.dumps(
            {
                "scope": "toy CPU train, real spawn DataLoader and W&B offline SDK; no GPU or pretrained weights",
                "num_workers": 2,
                "history": history,
                "matches_csv": True,
                "journal": str(journals[0]),
            },
            indent=2,
        )
    )
