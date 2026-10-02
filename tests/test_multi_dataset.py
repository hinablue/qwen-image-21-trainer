from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from qwen21_trainer.config import Config, load_config
from qwen21_trainer.data import load_manifest, prepare_dataset


def group(root: Path):
    root.mkdir()
    Image.new("RGB", (64, 64)).save(root / "same.png")
    (root / "same.txt").write_text("a test image", encoding="utf-8")
    return root


def test_multi_dataset_repeats_and_same_stem(tmp_path):
    a, b = group(tmp_path / "a"), group(tmp_path / "b")
    cfg = Config(
        path=tmp_path / "config.toml",
        training={"cache_dir": str(tmp_path / "cache"), "output_dir": str(tmp_path / "output")},
        datasets=[
            {"path": str(a), "repeats": 5, "max_pixels": 1048576},
            {"path": str(b), "repeats": 1, "max_pixels": 262144},
        ],
    )
    result = prepare_dataset(cfg)
    assert len(result["items"]) == 2
    assert len({i["id"] for i in result["items"]}) == 2
    assert [i["repeats"] for i in result["items"]] == [5, 1]
    assert [i["max_pixels"] for i in result["items"]] == [1048576, 262144]
    assert load_manifest(cfg) == result
    altered = replace(
        cfg, datasets=[{"path": str(a), "repeats": 1}, {"path": str(b), "repeats": 1}]
    )
    with pytest.raises(ValueError, match="過期"):
        load_manifest(altered)


def test_duplicate_root_rejected(tmp_path):
    root = group(tmp_path / "a")
    cfg = Config(
        path=tmp_path / "cfg.toml",
        datasets=[{"path": str(root)}, {"path": str(root)}],
        training={"cache_dir": str(tmp_path / "cache"), "output_dir": str(tmp_path / "output")},
    )
    with pytest.raises(ValueError, match="重複列入"):
        prepare_dataset(cfg)


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "5"])
def test_invalid_repeat(value, tmp_path):
    with pytest.raises(ValueError):
        Config(
            path=tmp_path / "cfg.toml",
            datasets=[{"path": str(tmp_path / "images"), "repeats": value}],
        )


def test_empty_datasets_toml_rejected(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("datasets = []")
    with pytest.raises(ValueError):
        load_config(p)
