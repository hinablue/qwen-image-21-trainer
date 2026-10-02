"""Portable public presets and source-import output isolation (no real data)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from qwen21_trainer.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["small.toml", "wavelet-adopt.example.toml"])
def test_public_presets_are_portable_and_do_not_upload(name):
    cfg = load_config(ROOT / "configs" / name)
    assert cfg.wandb["enabled"] is False
    assert cfg.sample["enabled"] is False
    assert cfg.training["include_blocks"] == []
    assert cfg.training["exclude_blocks"] == []
    for section, field in (
        ("model", "root"),
        ("dataset", "path"),
        ("training", "cache_dir"),
        ("training", "output_dir"),
    ):
        assert not Path(cfg.source_document[section][field]).is_absolute()


def test_import_uses_destination_stem_and_preserves_source(tmp_path):
    images = tmp_path / "images" / "fixture"
    images.mkdir(parents=True)
    source = tmp_path / "source.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "config": {
                    "process": [
                        {
                            "datasets": [
                                {
                                    "folder_path": "/container/datasets/fixture",
                                    "resolution": [512],
                                    "num_repeats": 3,
                                }
                            ],
                            "network": {"linear": 32},
                            "train": {"lr": 0.0001},
                            "sample": {
                                "samples": [{"prompt": "a ceramic cup"}],
                                "width": 512,
                                "height": 512,
                            },
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    original = source.read_bytes()
    outputs = []
    for name in ("experiment-a", "experiment-b"):
        destination = tmp_path / "configs" / f"{name}.toml"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/import_aitk_config.py"),
                "--source", str(source),
                "--output", str(destination),
                "--host-datasets-root", str(images.parent),
                "--container-datasets-root", "/container/datasets",
                "--steps", "2",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(result.stdout)["config"] == str(destination)
        cfg = load_config(destination)
        assert cfg.training["output_dir"] == str(tmp_path / "output" / name)
        assert cfg.training["cache_dir"] == str(tmp_path / "cache" / name)
        assert cfg.training["max_steps"] == 2
        assert cfg.datasets[0]["path"] == str(images)
        assert cfg.datasets[0]["repeats"] == 3
        assert destination.with_suffix(".import.json").is_file()
        outputs.append(cfg.training["output_dir"])
    assert len(set(outputs)) == 2
    assert source.read_bytes() == original
