"""Native YAML/TOML share one schema and capture the as-loaded source."""

import json

import pytest
import yaml

from qwen21_trainer.config import Config, load_config


def test_yaml_and_toml_resolve_equally_and_snapshot_source(tmp_path):
    toml = tmp_path / "config.toml"
    yml = tmp_path / "config.yaml"
    raw = {
        "training": {"rank": 8},
        "sample": {"enabled": True, "every": 2, "prompts": ["a cup", "a fox"]},
    }
    toml.write_text(
        '[training]\nrank=8\n[sample]\nenabled=true\nevery=2\nprompts=["a cup", "a fox"]\n'
    )
    yml.write_text(yaml.safe_dump(raw))
    a, b = load_config(toml), load_config(yml)
    da, db = a.to_dict(), b.to_dict()
    da.pop("path")
    db.pop("path")
    assert da == db
    assert a.source_document == b.source_document == raw
    assert a.source_format == "toml" and b.source_format == "yaml"
    yml.write_text("training: {rank: 64}\n")
    assert b.source_document == raw and b.training["rank"] == 8
    json.dumps(b.to_dict())


def test_direct_config_has_no_fabricated_source(tmp_path):
    cfg = Config(path=tmp_path / "missing.toml")
    assert cfg.source_document is None and cfg.source_format is None
    assert cfg.wandb["log_config"] is True and cfg.wandb["log_samples"] is True
    assert cfg.sample["enabled"] is False


@pytest.mark.parametrize(
    "text",
    [
        "training: {rank: 8, rank: 16}",
        'training: !!python/object/apply:os.system ["false"]',
        "sample: &x {prompts: [*x]}",
        "sample: {prompts: [12]}",
        "sample: {enabled: yes, every: 0}",
        "wandb: {log_config: 1}",
        "[1, 2, 3]",
        "job: extension\nconfig: {process: []}",
    ],
)
def test_bad_yaml_rejected(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)


def test_oversized_input_rejected_before_parse(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_bytes(b"#" * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="1 MiB"):
        load_config(path)


@pytest.mark.parametrize(
    "settings",
    [
        {"every": 0},
        {"every": True},
        {"prompts": "one"},
        {"prompts": [""]},
        {"prompts": ["x"] * 17},
        {"enabled": 1},
    ],
)
def test_bad_sample_settings_rejected(tmp_path, settings):
    with pytest.raises(ValueError):
        Config(path=tmp_path / "c.toml", sample=settings)


def test_sampling_with_cpu_offload_not_silently_accepted(tmp_path):
    with pytest.raises(ValueError, match="cpu_offload"):
        Config(path=tmp_path / "c.toml", model={"cpu_offload": True}, sample={"enabled": True})


def test_yaml_format_does_not_invalidate_same_dataset_cache(tmp_path):
    from PIL import Image

    from qwen21_trainer.data import load_manifest, prepare_dataset

    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (64, 64), "white").save(images / "one.png")
    (images / "one.txt").write_text("a fixture")
    toml = tmp_path / "config.toml"
    toml.write_text(
        '[dataset]\npath="images"\n[training]\ncache_dir="cache"\noutput_dir="output"\n'
    )
    raw = {
        "dataset": {"path": "images"},
        "training": {"cache_dir": "cache", "output_dir": "output"},
    }
    yml = tmp_path / "config.yaml"
    yml.write_text(yaml.safe_dump(raw))
    first = prepare_dataset(load_config(toml))
    second = load_manifest(load_config(yml))
    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]
