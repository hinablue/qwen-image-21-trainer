"""Native YAML -> real tiny training/sampling -> real W&B offline readback.

TE, processor and VAE are explicit fixtures, not pretrained image generation.
"""

import csv
import json
import os
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image
from safetensors.torch import save_file
from test_tracking_media import history_rows, read_run
from test_training_sampling import TinyFixtureEncoder, TinyFixtureProcessor, TinyFixtureVAE

from qwen21_trainer import data, runtime
from qwen21_trainer.config import load_config
from qwen21_trainer.config_snapshot import credential_key


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_yaml_train_sample_and_wandb_together(tmp_path, monkeypatch, dtype):
    from diffsynth.models.qwen_image_21_dit import QwenImage21DiT
    from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline

    for key in tuple(os.environ):
        if credential_key(key):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WANDB_BASE_URL", "http://127.0.0.1:9")
    raw = {
        "model": {"device": "cpu"},
        "training": {
            "cache_dir": "cache",
            "output_dir": "out",
            "rank": 32,
            "alpha": 16,
            "lr_scheduler": "cosine",
            "num_warmup": 1,
            "max_steps": 3,
            "save_every": 2,
            "num_workers": 2,
        },
        "wandb": {
            "enabled": True,
            "mode": "offline",
            "log_every": 3,
            "log_config": True,
            "log_samples": True,
        },
        "sample": {
            "enabled": True,
            "every": 2,
            "prompts": ["tiny fixture A", "tiny fixture B"],
            "width": 64,
            "height": 64,
            "steps": 2,
            "cfg_scale": 1.0,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path)
    folder = tmp_path / "cache"
    folder.mkdir()
    save_file(
        {
            "input_latents": torch.randn(1, 64, 4, 4, dtype=dtype),
            "prompt_embeds": torch.randn(1, 5, 64, dtype=dtype),
            "edit_image_pad_mask": torch.zeros(1, 5, dtype=torch.bool),
        },
        str(folder / "one.safetensors"),
    )
    monkeypatch.setattr(data, "load_manifest", lambda cfg: {"dataset_fingerprint": "tiny-fixture"})
    monkeypatch.setattr(runtime, "model_assets", lambda cfg: {"fixture_only": True})
    monkeypatch.setattr(
        runtime,
        "load_cache_index",
        lambda *args: {
            "identity": {"model_signature": "tiny-fixture"},
            "items": [{"file": "one.safetensors"}],
        },
    )
    roles_seen = []

    def build(cfg, assets, roles, **kwargs):
        roles_seen.append(list(roles))
        pipe = QwenImage21Pipeline(device="cpu", torch_dtype=dtype)
        if roles == ["transformer"]:
            pipe.dit = QwenImage21DiT(
                in_channels=64,
                out_channels=64,
                num_layers=1,
                num_attention_heads=2,
                attention_head_dim=32,
                context_in_dim=64,
                axes_dims_rope=(8, 12, 12),
            ).to(dtype=dtype)
        else:
            assert roles == ["text_encoder", "vae"]
            pipe.text_encoder = TinyFixtureEncoder().to(dtype=dtype)
            pipe.processor = TinyFixtureProcessor()
            pipe.vae = TinyFixtureVAE()
        return pipe

    original_make = runtime.make_training_module
    monkeypatch.setattr(runtime, "build_pipeline", build)
    monkeypatch.setattr(
        runtime,
        "make_training_module",
        lambda pipe, settings, **kwargs: original_make(
            pipe, settings, targets="to_q,to_k,to_v,to_out.0", **kwargs
        ),
    )
    summary = runtime.train(cfg)
    output = tmp_path / "out"
    journal, records = read_run(output)
    rows = history_rows(records)
    assert [row["train/step"] for row in rows] == [1, 2, 3]
    assert "samples/images" not in rows[0]
    media_rows = rows[1:]
    assert all(row["samples/images"]["count"] == 2 for row in media_rows)
    for row in media_rows:
        for index, filename in enumerate(row["samples/images"]["filenames"]):
            local = output / "samples" / f"step-{row['train/step']:06d}-{index:02d}.png"
            remote_staged = journal.parent / "files" / filename
            with Image.open(local) as a, Image.open(remote_staged) as b:
                assert a.size == b.size == (64, 64)
                assert a.convert("RGBA").tobytes() == b.convert("RGBA").tobytes()
    with (output / "metrics.csv").open() as f:
        csv_rows = list(csv.DictReader(f))
    assert [row["train/loss"] for row in rows] == [float(row["loss"]) for row in csv_rows]
    assert [row["lr/dit"] for row in rows] == pytest.approx(
        [0.0, cfg.training["learning_rate"], cfg.training["learning_rate"] / 2]
    )
    assert [row["loss/average"] for row in rows] == [float(row["loss_average"]) for row in csv_rows]
    assert all(f"sample_{index}" in row for row in media_rows for index in (0, 1))
    from qwen21_trainer.lora_io import read_checkpoint

    saved_lora = read_checkpoint(summary["checkpoint"])
    assert saved_lora.rank == 32 and saved_lora.alpha == 16
    assert saved_lora.metadata["lr_scheduler"] == "cosine"
    assert saved_lora.metadata["num_warmup"] == "1"
    artifacts = [
        r.artifact
        for r in records
        if r.HasField("artifact") and r.artifact.type == "training-config"
    ]
    assert len(artifacts) == 1 and artifacts[0].type == "training-config"
    assert (
        len([r for r in records if r.HasField("artifact") and r.artifact.type == "training-log"])
        == 1
    )
    assert yaml.safe_load((output / "wandb-config/source-config.yaml").read_text()) == raw
    assert roles_seen == [["transformer"], ["text_encoder", "vae"], ["text_encoder", "vae"]]
    assert Path(summary["checkpoint"]).is_file()
    assert [r.exit.exit_code for r in records if r.HasField("exit")] == [0]
    assert not torch.cuda.is_initialized()
    (tmp_path / "end-to-end-evidence.json").write_text(
        json.dumps(
            {
                "scope": "real tiny DiT/PEFT and upstream inference; fixture TE/processor/VAE; offline only",
                "dtype": str(dtype),
                "rank": saved_lora.rank,
                "alpha": saved_lora.alpha,
                "lr_used": [row["lr/dit"] for row in rows],
                "history_steps": [r["train/step"] for r in rows],
                "sample_steps": [r["train/step"] for r in media_rows],
                "png_count": sum(r["samples/images"]["count"] for r in media_rows),
                "config_artifact_count": len(artifacts),
                "csv_matches": True,
                "png_pixels_match": True,
                "roles_seen": roles_seen,
                "journal": str(journal),
                "checkpoint": summary["checkpoint"],
            },
            indent=2,
        )
    )
