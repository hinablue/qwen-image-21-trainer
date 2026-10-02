"""Real tqdm rendering with CPU fixtures; no production cache/training runs."""

import csv
import io
import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file
from tqdm import tqdm as real_tqdm

from qwen21_trainer import runtime
from qwen21_trainer.config import Config
from qwen21_trainer.data import prepare_dataset


class TTYBuffer(io.StringIO):
    def isatty(self):
        return True


@pytest.fixture
def progress_capture(monkeypatch):
    stream = TTYBuffer()
    bars = []

    def create(*args, **kwargs):
        kwargs.update(file=stream, disable=False, mininterval=0, miniters=1, dynamic_ncols=False)
        bar = real_tqdm(*args, **kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(runtime, "tqdm", create)
    return stream, bars


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.tensor([1.0]))

    def trainable_modules(self):
        return self.parameters()

    def forward(self, sample):
        return (self.lora_A - sample["target"]).square().mean()


def test_train_single_progress_bar_and_csv_preserved(
    tmp_path, monkeypatch, capsys, progress_capture
):
    from qwen21_trainer import data

    cfg = Config(
        path=tmp_path / "config.toml",
        model={"device": "cpu"},
        training={
            "cache_dir": str(tmp_path / "cache"),
            "output_dir": str(tmp_path / "out"),
            "max_steps": 3,
            "gradient_accumulation_steps": 2,
            "save_every": 1,
        },
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    save_file({"target": torch.zeros(1)}, str(cache / "item.safetensors"))
    monkeypatch.setattr(data, "load_manifest", lambda cfg: {"dataset_fingerprint": "fixture"})
    monkeypatch.setattr(runtime, "model_assets", lambda cfg: {})
    monkeypatch.setattr(
        runtime,
        "load_cache_index",
        lambda *args: {
            "identity": {"model_signature": "fixture"},
            "items": [{"file": "item.safetensors"}],
        },
    )
    monkeypatch.setattr(runtime, "build_pipeline", lambda *args, **kwargs: object())
    model = ScalarModel()
    monkeypatch.setattr(runtime, "make_training_module", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        runtime,
        "save_lora",
        lambda model, path, meta: save_file({"lora_A": model.lora_A.detach()}, str(path)),
    )
    summary = runtime.train(cfg)
    stream, bars = progress_capture
    assert len(bars) == 1 and bars[0].total == 3 and bars[0].n == 3
    assert bars[0].disable  # closed, including cursor/newline cleanup
    text = stream.getvalue()
    assert "Train" in text and "100%" in text and "loss=" in text and "lr=" in text
    assert text.count("\n") == 1 and text.count("\r") > 1
    assert capsys.readouterr().out == ""  # no per-step print or entire run.json dump
    with (tmp_path / "out/metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [int(row["step"]) for row in rows] == [1, 2, 3]
    assert [int(row["images_seen"]) for row in rows] == [2, 4, 6]
    assert summary["optimizer_updates"] == 3
    assert (
        json.loads((tmp_path / "out/run.json").read_text())["optimizer_runtime"]["name"] == "adamw"
    )


def test_training_failure_closes_partial_bar(tmp_path, progress_capture):
    cfg = Config(path=tmp_path / "config.toml", training={"max_steps": 3})
    with pytest.raises(FloatingPointError):
        runtime.optimization_loop(
            ScalarModel(), [{"target": torch.tensor([float("nan")])}], cfg.training
        )
    stream, bars = progress_capture
    assert bars[0].n == 0 and bars[0].disable
    assert "100%" not in stream.getvalue() and stream.getvalue().count("\n") == 1


def test_redirected_progress_is_quiet(tmp_path, monkeypatch, capsys):
    stream = io.StringIO()  # isatty() is false, e.g. redirected logs.

    def create(*args, **kwargs):
        return real_tqdm(*args, file=stream, **kwargs)

    monkeypatch.setattr(runtime, "tqdm", create)
    cfg = Config(path=tmp_path / "config.toml", training={"max_steps": 2})
    rows = runtime.optimization_loop(ScalarModel(), [{"target": torch.zeros(1)}], cfg.training)
    assert len(rows) == 2
    assert stream.getvalue() == ""
    assert capsys.readouterr().out == ""


class EncoderDouble(torch.nn.Module):
    def __init__(self, fail_at=None):
        super().__init__()
        from diffsynth.diffusion.flow_match import FlowMatchScheduler

        self.scheduler = FlowMatchScheduler("Qwen-Image")
        self.norm = torch.nn.Identity()
        self.text_encoder = SimpleNamespace(
            model=SimpleNamespace(
                model=SimpleNamespace(language_model=SimpleNamespace(norm=self.norm))
            )
        )
        self.units = [object()]
        self.calls = 0
        self.fail_at = fail_at

    def unit_runner(self, unit, pipe, shared, positive, negative):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("encoder fixture failure")
        shared["input_latents"] = torch.ones(1, 64, 4, 4, dtype=torch.bfloat16)
        positive.update(
            prompt_embeds=torch.ones(1, 5, 4096, dtype=torch.bfloat16),
            edit_image_pad_mask=torch.zeros(1, 5, dtype=torch.bool),
            prompt_embeds_mask=None,
        )
        return shared, positive, negative


def cache_fixture(tmp_path, monkeypatch, fail_at=None):
    images = tmp_path / "images"
    images.mkdir()
    for i in range(2):
        Image.new("RGB", (64, 64), "red").save(images / f"{i}.png")
        (images / f"{i}.txt").write_text("a red fixture")
    cfg = Config(
        path=tmp_path / "config.toml",
        dataset={"path": str(images)},
        training={
            "cache_dir": str(tmp_path / "cache"),
            "output_dir": str(tmp_path / "out"),
        },
    )
    prepare_dataset(cfg)
    monkeypatch.setattr(runtime, "model_assets", lambda cfg: {})
    monkeypatch.setattr(runtime, "source_signature", lambda *args: "fixture")
    double = EncoderDouble(fail_at=fail_at)
    monkeypatch.setattr(runtime, "build_pipeline", lambda *args: double)
    return cfg, double


def test_cache_progress_includes_resumed_items(tmp_path, monkeypatch, capsys, progress_capture):
    cfg, encoder = cache_fixture(tmp_path, monkeypatch)
    runtime.cache_dataset(cfg)
    runtime.cache_dataset(cfg)
    stream, bars = progress_capture
    assert len(bars) == 2
    assert all(bar.n == 2 and bar.total == 2 and bar.disable for bar in bars)
    assert "encoded=2" in bars[0].postfix and "reused=0" in bars[0].postfix
    assert "encoded=0" in bars[1].postfix and "reused=2" in bars[1].postfix
    assert encoder.calls == 2
    assert capsys.readouterr().out == ""
    assert stream.getvalue().count("\n") == 2


def test_cache_failure_closes_without_counting_failed_item(tmp_path, monkeypatch, progress_capture):
    cfg, _ = cache_fixture(tmp_path, monkeypatch, fail_at=2)
    with pytest.raises(RuntimeError, match="fixture failure"):
        runtime.cache_dataset(cfg)
    stream, bars = progress_capture
    assert bars[0].n == 1 and bars[0].disable
    assert "100%" not in stream.getvalue()
    assert not (tmp_path / "cache/cache-index.json").exists()
    # Same-config restart reuses the completed first image after interruption.
    runtime.cache_dataset(cfg)
    assert len(bars) == 2 and bars[1].n == 2
    assert "encoded=1" in bars[1].postfix and "reused=1" in bars[1].postfix
