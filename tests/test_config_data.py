"""CPU-only config/manifest fixtures; no model downloads or torch imports."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from qwen21_trainer.config import Config, load_config
from qwen21_trainer.data import load_manifest, prepare_dataset


class Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="qwen21-config-data-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config_path = self.root / "configs" / "訓練 設定.toml"
        self.config_path.parent.mkdir()
        self.dataset = self.root / "資料 有空白"
        self.dataset.mkdir()
        self.cache = self.root / "cache"

    def config(self, updates: dict | None = None) -> Config:
        sections = {
            "model": {"root": "../models"},
            "dataset": {"path": "../資料 有空白"},
            "training": {"cache_dir": "../cache", "output_dir": "../output"},
        }
        for name, values in (updates or {}).items():
            if isinstance(values, dict):
                sections.setdefault(name, {}).update(values)
            else:
                sections[name] = values
        lines = []
        for name, values in sections.items():
            if not isinstance(values, dict):
                lines.insert(0, f"{name} = {json.dumps(values)}")
                continue
            lines.append(f"[{name}]")
            for key, value in values.items():
                if isinstance(value, float):
                    encoded = repr(value)
                else:
                    encoded = json.dumps(value, ensure_ascii=False)
                lines.append(f"{key} = {encoded}")
        self.config_path.write_text("\n".join(lines), encoding="utf-8")
        return load_config(self.config_path)

    def image(
        self, name: str = "小 茶杯.png", caption: str | None = "一只陶瓷茶杯", *, color: str = "red"
    ) -> Path:
        path = self.dataset / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), color=color).save(path)
        if caption is not None:
            path.with_suffix(".txt").write_text(caption, encoding="utf-8")
        return path

    @property
    def manifest_path(self) -> Path:
        return self.cache / "dataset.json"

    def save_manifest(self, value: dict) -> None:
        self.manifest_path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class ConfigTests(Fixture):
    def test_defaults_and_jsonable_resolved_snapshot(self):
        self.config_path.write_text("", encoding="utf-8")
        cfg = load_config(self.config_path)
        self.assertIsInstance(cfg.path, Path)
        self.assertEqual(cfg.path, self.config_path.resolve())
        self.assertEqual(
            cfg.model,
            {
                "root": str(self.root / "models/Qwen-Image-2.1"),
                "format": "official",
                "processor_path": "",
                "device": "cuda",
                "attention": "segmented",
                "cpu_offload": False,
            },
        )
        self.assertEqual(
            cfg.dataset,
            {
                "path": str(self.root / "data/train"),
                "caption_extension": ".txt",
                "max_pixels": 262144,
                "recursive": True,
            },
        )
        self.assertEqual(
            cfg.training,
            {
                "output_dir": str(self.root / "output/small"),
                "cache_dir": str(self.root / "cache/small"),
                "seed": 42,
                "rank": 32,
                "alpha": 32,
                "include_blocks": [],
                "exclude_blocks": [],
                "learning_rate": 1e-4,
                "lr_scheduler": "diffsynth",
                "num_warmup": 0,
                "weight_decay": 0.01,
                "loss_type": "mse",
                "loss_weighting": "diffsynth",
                "optimizer": "adamw",
                "optimizer_params": {},
                "max_steps": 300,
                "gradient_accumulation_steps": 1,
                "num_workers": 0,
                "prefetch_factor": 2,
                "save_every": 100,
                "gradient_checkpointing": True,
                "checkpointing_offload": False,
            },
        )
        self.assertEqual(
            cfg.sample,
            {
                "enabled": False,
                "every": 100,
                "prompts": [],
                "prompt": "a photo of a small ceramic cup on a wooden table",
                "negative_prompt": "",
                "width": 512,
                "height": 512,
                "steps": 30,
                "cfg_scale": 3.0,
                "seed": 42,
            },
        )
        snapshot = cfg.to_dict()
        self.assertEqual(json.loads(json.dumps(snapshot, allow_nan=False)), snapshot)
        snapshot["training"]["rank"] = 1
        self.assertEqual(cfg.training["rank"], 32)
        self.assertFalse((self.root / "models").exists())
        self.assertFalse(self.cache.exists())

    def test_relative_paths_use_config_directory_not_cwd(self):
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        cfg = self.config()
        relative = load_config(Path("configs") / self.config_path.name)
        self.assertEqual(relative.to_dict(), cfg.to_dict())
        self.assertEqual(cfg.dataset["path"], str(self.dataset))
        self.assertEqual(cfg.model["root"], str(self.root / "models"))

    def test_unknown_sections_and_keys_are_rejected(self):
        for updates in (
            {"unknown": {"x": 1}},
            {"model": {"unknown": 1}},
            {"training": {"unsupported_alpha_alias": 16}},
            {"training": {"batch_size": 1}},
            {"dataset": {"edit_image": "a.png"}},
            {"sample": {"unknown": 1}},
        ):
            with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, "不支援"):
                self.config(updates)

    def test_booleans_are_not_integers_or_numbers(self):
        fields = {
            "dataset": ("max_pixels",),
            "training": (
                "rank",
                "max_steps",
                "gradient_accumulation_steps",
                "save_every",
                "seed",
                "learning_rate",
                "weight_decay",
            ),
            "sample": ("width", "height", "steps", "seed", "cfg_scale"),
        }
        for section, keys in fields.items():
            for key in keys:
                with self.subTest(key=f"{section}.{key}"), self.assertRaises(ValueError):
                    self.config({section: {key: True}})

    def test_bool_fields_reject_integer_values(self):
        for section, key in (
            ("model", "cpu_offload"),
            ("dataset", "recursive"),
            ("training", "gradient_checkpointing"),
            ("training", "checkpointing_offload"),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.config({section: {key: 1}})

    def test_nonfinite_and_out_of_range_numbers_rejected(self):
        for section, key in (
            ("training", "learning_rate"),
            ("training", "weight_decay"),
            ("sample", "cfg_scale"),
        ):
            for value in (float("nan"), float("inf"), float("-inf"), -0.01, "1.0"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.config({section: {key: value}})
        for section, key, value in (
            ("dataset", "max_pixels", 1023),
            ("training", "rank", 0),
            ("training", "rank", 1.0),
            ("training", "learning_rate", 0),
            ("training", "max_steps", 0),
            ("training", "seed", -1),
            ("training", "gradient_accumulation_steps", 0),
            ("training", "save_every", 0),
            ("sample", "steps", 0),
            ("sample", "width", 31),
            ("sample", "height", 513),
            ("sample", "seed", -1),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.config({section: {key: value}})

    def test_valid_zero_and_minimum_values(self):
        cfg = self.config(
            {
                "model": {"device": "cpu", "attention": "flex", "cpu_offload": True},
                "dataset": {"max_pixels": 1024},
                "training": {"rank": 1, "seed": 0, "weight_decay": 0},
                "sample": {"width": 32, "height": 32, "cfg_scale": 0, "seed": 0},
            }
        )
        self.assertEqual(cfg.training["weight_decay"], 0.0)
        self.assertEqual(cfg.sample["cfg_scale"], 0.0)

    def test_invalid_enum_string_and_section_types(self):
        for updates in (
            {"model": {"device": "cuda:0"}},
            {"model": {"attention": "flash"}},
            {"model": {"root": " "}},
            {"dataset": {"path": 1}},
            {"sample": {"prompt": ["a"]}},
            {"dataset": 1},
        ):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                self.config(updates)
        for extension in ("txt", "../x", ".x/y", ".x\\y", ".", ".jpg", ".a.txt"):
            with self.subTest(extension=extension), self.assertRaises(ValueError):
                self.config({"dataset": {"caption_extension": extension}})

    def test_output_and_cache_boundaries(self):
        for key, path in (
            ("output_dir", "../資料 有空白"),
            ("output_dir", ".."),
            ("cache_dir", "../資料 有空白"),
            ("cache_dir", ".."),
            ("cache_dir", "../資料 有空白/cache"),
        ):
            with self.subTest(key=key, path=path), self.assertRaises(ValueError):
                self.config({"training": {key: path}})
        # Only cache descendants are prohibited by this contract.
        cfg = self.config({"training": {"output_dir": "../資料 有空白/output"}})
        self.assertTrue(Path(cfg.training["output_dir"]).is_relative_to(self.dataset))

    def test_boundary_checks_follow_aliases(self):
        alias = self.root / "alias"
        alias.symlink_to(self.dataset, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.config({"training": {"cache_dir": "../alias/cache"}})

    def test_invalid_toml_and_missing_file(self):
        with self.assertRaisesRegex(FileNotFoundError, "找不到設定檔"):
            load_config(self.config_path)
        for content in ("[broken", "[model]\nroot=1\nroot=2", "training = []"):
            self.config_path.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaises(ValueError):
                load_config(self.config_path)
        with self.assertRaises(ValueError):
            load_config("\x00")
        with self.assertRaises(ValueError):
            load_config(self.root)

    def test_dataclass_copies_and_validates_inputs(self):
        model = {"device": "cpu"}
        cfg = Config(self.config_path, model=model)
        model["device"] = "bogus"
        self.assertEqual(cfg.model["device"], "cpu")
        with self.assertRaises(ValueError):
            Config(self.config_path, training={"rank": True})


class DatasetTests(Fixture):
    def test_one_image_unicode_spaces_hashes_and_dimensions(self):
        image = self.image(caption="\ufeff  陶瓷杯\n放在木桌上  \n")
        cfg = self.config()
        manifest = prepare_dataset(cfg)
        item = manifest["items"][0]
        self.assertEqual(len(manifest["items"]), 1)
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(
            manifest["preprocessing"], {"max_pixels": 262144, "rgba": True, "division_factor": 32}
        )
        self.assertEqual(
            item,
            {
                "id": hashlib.sha256(image.name.encode("utf-8")).hexdigest()[:20],
                "image": str(image),
                "caption_file": str(image.with_suffix(".txt")),
                "prompt": "陶瓷杯\n放在木桌上",
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "caption_sha256": hashlib.sha256(
                    image.with_suffix(".txt").read_bytes()
                ).hexdigest(),
                "width": 64,
                "height": 48,
            },
        )
        self.assertRegex(manifest["dataset_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertEqual(load_manifest(cfg), manifest)
        self.assertEqual(json.loads(self.manifest_path.read_text()), manifest)

    def test_all_supported_extensions_and_deterministic_order(self):
        for name in ("z.bmp", "b.jpeg", "A.JPG", "小杯.webp", "a.png"):
            self.image(name)
        manifest = prepare_dataset(self.config())
        names = [Path(item["image"]).name for item in manifest["items"]]
        self.assertEqual(names, sorted(names))
        self.assertEqual(len(names), 5)

    def test_root_symlink_is_resolved_and_allowed(self):
        image = self.image()
        alias = self.root / "資料連結"
        alias.symlink_to(self.dataset, target_is_directory=True)
        cfg = self.config({"dataset": {"path": "../資料連結"}})
        self.assertEqual(prepare_dataset(cfg)["items"][0]["image"], str(image))

    def test_recursive_and_nonrecursive(self):
        self.image("a.png")
        self.image("子目錄/a.png")
        cfg = self.config()
        manifest = prepare_dataset(cfg)
        self.assertEqual(len(manifest["items"]), 2)
        self.assertEqual(len({item["id"] for item in manifest["items"]}), 2)
        cfg = self.config({"dataset": {"recursive": False}})
        with self.assertRaises(ValueError):
            load_manifest(cfg)
        self.assertEqual(len(prepare_dataset(cfg, overwrite=True)["items"]), 1)

    def test_recursive_setting_change_stales_even_same_items(self):
        self.image()
        prepare_dataset(self.config())
        cfg = self.config({"dataset": {"recursive": False}})
        with self.assertRaisesRegex(ValueError, "過期"):
            load_manifest(cfg)

    def test_missing_or_empty_caption_creates_no_cache(self):
        path = self.image(caption=None)
        cfg = self.config()
        with self.assertRaisesRegex(FileNotFoundError, "caption"):
            prepare_dataset(cfg)
        self.assertFalse(self.cache.exists())
        for caption in ("", " \n\t\r ", "\ufeff"):
            path.with_suffix(".txt").write_text(caption, encoding="utf-8")
            with self.subTest(caption=caption), self.assertRaisesRegex(ValueError, "空白"):
                prepare_dataset(cfg)
            self.assertFalse(self.cache.exists())

    def test_non_utf8_caption_rejected(self):
        path = self.image()
        path.with_suffix(".txt").write_bytes(b"\xff\xfe")
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            prepare_dataset(self.config())
        self.assertFalse(self.cache.exists())

    def test_corrupt_and_truncated_images_rejected(self):
        path = self.image()
        original = path.read_bytes()
        cfg = self.config()
        for content in (b"not an image", original[:30]):
            path.write_bytes(content)
            with self.subTest(length=len(content)), self.assertRaisesRegex(ValueError, "圖片"):
                prepare_dataset(cfg)
            self.assertFalse(self.cache.exists())

    def test_empty_or_missing_dataset_rejected(self):
        cfg = self.config()
        with self.assertRaisesRegex(ValueError, "沒有支援的圖片"):
            prepare_dataset(cfg)
        self.dataset.rmdir()
        with self.assertRaisesRegex(FileNotFoundError, "資料集"):
            prepare_dataset(cfg)

    def test_multiple_image_same_stem_rejected(self):
        self.image("same.jpg")
        self.image("same.png")
        with self.assertRaisesRegex(ValueError, "stem 重複"):
            prepare_dataset(self.config())
        self.assertFalse(self.cache.exists())

    def test_identical_prepare_is_idempotent_without_source_changes(self):
        image = self.image()
        cfg = self.config()
        source = {path: path.read_bytes() for path in (image, image.with_suffix(".txt"))}
        first = prepare_dataset(cfg)
        before = self.manifest_path.stat()
        self.assertEqual(prepare_dataset(cfg), first)
        self.assertEqual(prepare_dataset(cfg, overwrite=True), first)
        self.assertEqual(self.manifest_path.stat().st_mtime_ns, before.st_mtime_ns)
        self.assertEqual(self.manifest_path.stat().st_ino, before.st_ino)
        self.assertEqual({path: path.read_bytes() for path in source}, source)

    def test_changed_image_bytes_require_explicit_overwrite(self):
        path = self.image()
        cfg = self.config()
        previous = prepare_dataset(cfg)
        old_manifest = self.manifest_path.read_bytes()
        Image.new("RGB", (64, 48), "blue").save(path)
        changed_bytes = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "過期"):
            load_manifest(cfg)
        with self.assertRaisesRegex(ValueError, "--overwrite"):
            prepare_dataset(cfg)
        self.assertEqual(self.manifest_path.read_bytes(), old_manifest)
        updated = prepare_dataset(cfg, overwrite=True)
        self.assertNotEqual(previous["dataset_fingerprint"], updated["dataset_fingerprint"])
        self.assertEqual(previous["items"][0]["id"], updated["items"][0]["id"])
        self.assertEqual(path.read_bytes(), changed_bytes)
        self.assertEqual(load_manifest(cfg), updated)

    def test_caption_hash_changes_even_when_stripped_prompt_unchanged(self):
        path = self.image(caption="茶杯")
        cfg = self.config()
        previous = prepare_dataset(cfg)
        path.with_suffix(".txt").write_text(" 茶杯\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_manifest(cfg)
        with self.assertRaises(ValueError):
            prepare_dataset(cfg)
        updated = prepare_dataset(cfg, overwrite=True)
        self.assertEqual(previous["items"][0]["prompt"], updated["items"][0]["prompt"])
        self.assertNotEqual(
            previous["items"][0]["caption_sha256"], updated["items"][0]["caption_sha256"]
        )
        self.assertNotEqual(previous["dataset_fingerprint"], updated["dataset_fingerprint"])

    def test_added_and_removed_images_stale_manifest(self):
        self.image("a.png")
        cfg = self.config()
        prepare_dataset(cfg)
        added = self.image("b.png")
        with self.assertRaises(ValueError):
            load_manifest(cfg)
        prepare_dataset(cfg, overwrite=True)
        added.unlink()
        with self.assertRaises(ValueError):
            load_manifest(cfg)

    def test_preprocessing_change_stales_but_training_change_does_not(self):
        self.image()
        original = prepare_dataset(self.config())
        cfg = self.config({"training": {"learning_rate": 0.002, "rank": 16}})
        self.assertEqual(load_manifest(cfg), original)
        cfg = self.config({"dataset": {"max_pixels": 65536}})
        with self.assertRaisesRegex(ValueError, "過期"):
            load_manifest(cfg)
        with self.assertRaises(ValueError):
            prepare_dataset(cfg)
        updated = prepare_dataset(cfg, overwrite=True)
        self.assertEqual(updated["preprocessing"]["max_pixels"], 65536)

    def test_caption_extension_change_stales_manifest(self):
        image = self.image()
        image.with_suffix(".caption").write_text("一只陶瓷茶杯", encoding="utf-8")
        prepare_dataset(self.config())
        cfg = self.config({"dataset": {"caption_extension": ".caption"}})
        with self.assertRaises(ValueError):
            load_manifest(cfg)
        manifest = prepare_dataset(cfg, overwrite=True)
        self.assertEqual(manifest["items"][0]["caption_file"], str(image.with_suffix(".caption")))

    def test_validation_failure_preserves_old_manifest_even_overwrite(self):
        self.image("a.png")
        cfg = self.config()
        prepare_dataset(cfg)
        original = self.manifest_path.read_bytes()
        self.image("b.png", caption="")
        with self.assertRaises(ValueError):
            prepare_dataset(cfg, overwrite=True)
        self.assertEqual(self.manifest_path.read_bytes(), original)
        self.assertEqual(list(self.cache.iterdir()), [self.manifest_path])

    def test_image_caption_directory_and_dangling_symlinks_rejected(self):
        original = self.image()
        cfg = self.config()
        cases = [
            (self.dataset / "alias.png", original),
            (self.dataset / "outside", self.root),
            (self.dataset / "internal", self.dataset),
            (self.dataset / "dangling.png", self.root / "missing"),
        ]
        for link, target in cases:
            with self.subTest(link=link):
                link.symlink_to(target, target_is_directory=target.is_dir())
                try:
                    with self.assertRaisesRegex(ValueError, "符號連結"):
                        prepare_dataset(cfg)
                finally:
                    link.unlink()
        caption = original.with_suffix(".txt")
        outside = self.root / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        caption.unlink()
        caption.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "符號連結"):
            prepare_dataset(cfg)
        self.assertFalse(self.cache.exists())

    def test_symlink_added_after_prepare_rejected(self):
        path = self.image()
        cfg = self.config()
        prepare_dataset(cfg)
        outside = self.root / "outside.png"
        path.replace(outside)
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "符號連結"):
            load_manifest(cfg)

    def test_manifest_and_cache_symlinks_rejected(self):
        self.image()
        cfg = self.config()
        prepare_dataset(cfg)
        outside = self.root / "outside.json"
        self.manifest_path.replace(outside)
        self.manifest_path.symlink_to(outside)
        for action in (load_manifest, prepare_dataset):
            with (
                self.subTest(action=action.__name__),
                self.assertRaisesRegex(ValueError, "符號連結"),
            ):
                action(cfg)
        self.manifest_path.unlink()
        self.cache.rmdir()
        self.cache.symlink_to(self.dataset, target_is_directory=True)
        with self.assertRaises(ValueError):
            prepare_dataset(cfg)
        self.assertFalse((self.dataset / "dataset.json").exists())

    def test_manifest_path_escape_and_schema_tampering_rejected(self):
        self.image()
        cfg = self.config()
        original = prepare_dataset(cfg)
        mutations = [
            lambda m: m["items"][0].update(image="/etc/passwd"),
            lambda m: m["items"][0].update(caption_file="../../outside.txt"),
            lambda m: m["items"][0].update(width=True),
            lambda m: m["items"][0].update(width=64.0),
            lambda m: m["items"][0].update(prompt="changed"),
            lambda m: m["items"][0].update(id="0" * 20),
            lambda m: m["items"][0].update(sha256="0" * 64),
            lambda m: m["preprocessing"].update(rgba=1),
            lambda m: m.update(schema_version=True),
            lambda m: m.update(schema_version=2),
            lambda m: m.update(items=[]),
            lambda m: m.update(extra="unknown"),
            lambda m: m.pop("dataset_fingerprint"),
        ]
        for mutate in mutations:
            modified = json.loads(json.dumps(original))
            mutate(modified)
            self.save_manifest(modified)
            before = self.manifest_path.read_bytes()
            with self.subTest(modified=modified), self.assertRaises(ValueError):
                load_manifest(cfg)
            self.assertEqual(self.manifest_path.read_bytes(), before)

    def test_nonfinite_duplicate_and_invalid_manifest_json_rejected(self):
        self.image()
        cfg = self.config()
        prepare_dataset(cfg)
        for content in (
            '{"x": NaN}',
            '{"x": Infinity}',
            '{"x": -Infinity}',
            '{"x": 1e9999}',
            '{"schema_version": 1, "schema_version": 1}',
            "[]",
            "{",
            '{"x": ' + "9" * 5000 + "}",
        ):
            self.manifest_path.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaises(ValueError):
                load_manifest(cfg)
        self.manifest_path.write_bytes(b"\xff")
        with self.assertRaises(ValueError):
            load_manifest(cfg)
        # Explicit overwrite can repair an invalid manifest, not bad sources.
        prepared = prepare_dataset(cfg, overwrite=True)
        self.assertEqual(load_manifest(cfg), prepared)

    def test_missing_manifest_load_is_read_only(self):
        self.image()
        cfg = self.config()
        with self.assertRaisesRegex(FileNotFoundError, "manifest"):
            load_manifest(cfg)
        self.assertFalse(self.cache.exists())

    def test_load_success_has_no_writes(self):
        self.image()
        cfg = self.config()
        prepare_dataset(cfg)
        before = {
            str(path): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in self.root.rglob("*")
            if path.is_file()
        }
        load_manifest(cfg)
        after = {
            str(path): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in self.root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(before, after)

    def test_atomic_write_failure_preserves_existing_manifest(self):
        image = self.image()
        cfg = self.config()
        prepare_dataset(cfg)
        original = self.manifest_path.read_bytes()
        image.with_suffix(".txt").write_text("新 caption", encoding="utf-8")
        with patch("qwen21_trainer.data.os.replace", side_effect=OSError("simulated failure")):
            with self.assertRaisesRegex(ValueError, "原子寫入"):
                prepare_dataset(cfg, overwrite=True)
        self.assertEqual(self.manifest_path.read_bytes(), original)
        self.assertEqual(list(self.cache.iterdir()), [self.manifest_path])

    def test_atomic_no_clobber_when_manifest_appears_during_write(self):
        self.image()
        cfg = self.config()

        def concurrent_creator(source, destination):
            Path(destination).write_text("concurrent writer", encoding="utf-8")
            raise FileExistsError("concurrent file")

        with patch("qwen21_trainer.data.os.link", side_effect=concurrent_creator):
            with self.assertRaisesRegex(ValueError, "未覆寫"):
                prepare_dataset(cfg)
        self.assertEqual(self.manifest_path.read_text(), "concurrent writer")
        self.assertEqual(list(self.cache.iterdir()), [self.manifest_path])

    def test_mutated_config_and_non_boolean_overwrite_rejected(self):
        self.image()
        cfg = self.config()
        with self.assertRaises(ValueError):
            prepare_dataset(cfg, overwrite=1)
        cfg.dataset["max_pixels"] = float("nan")
        with self.assertRaises(ValueError):
            prepare_dataset(cfg)
        self.assertFalse(self.cache.exists())


if __name__ == "__main__":
    unittest.main()
