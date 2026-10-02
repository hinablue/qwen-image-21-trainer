"""Real offline SDK config/artifact/media proofs using explicitly synthetic PNGs.

No model is loaded; these fixture pixels must never be called generated model
samples. The tracker only receives explicitly selected files, never a scan.
"""

import base64
import copy
import hashlib
import json
import os
import traceback
from collections import Counter
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image, PngImagePlugin
from test_tracking import _offline_records, fixture_config, fixture_plan, row
from test_tracking import sdk as sdk_fixture

from qwen21_trainer.config import Config, load_config
from qwen21_trainer.config_snapshot import REDACTED, ConfigSanitizer, credential_key
from qwen21_trainer.tracking import TrainingTracker

sdk = sdk_fixture

FAKE_ENV_SECRET = "fixture-environment-token-not-a-real-credential"
FAKE_KEY_SECRET = "fixture-nested-api-key-not-a-real-credential"
FAKE_URL_SECRET = "fixture-url-password-not-a-real-credential"
FAKE_QUERY_SECRET = "fixture-query-token-not-a-real-credential"
COMMENT = "fixture-comment-only-must-not-be-uploaded"
PROMPT = "A hand-painted ceramic cup, explicit fixture prompt"


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    # No actual credentials enter these test runs. Never print environment values.
    for key in tuple(os.environ):
        if credential_key(key):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WANDB_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("HF_TOKEN", FAKE_ENV_SECRET)


def media_config(tmp_path, **options):
    cfg = fixture_config(**{"log_config": True, "log_samples": True, "log_every": 3, **options})
    cfg.path = tmp_path / "source.toml"
    # Parsed source snapshot, intentionally distinct from resolved paths/defaults.
    cfg.source_document = {
        "model": {"root": "../models/selected-model", "tokenizer": "preserved-tokenizer"},
        "dataset": {"path": "../data/selected-images"},
        "sample": {"enabled": True, "prompt": PROMPT},
        "credentials": {"opaque": FAKE_KEY_SECRET},
        "nested": [{"accessToken": FAKE_KEY_SECRET, "max_tokens": 8192}],
        "remote": f"https://fixture-user:{FAKE_URL_SECRET}@example.invalid/model?access_token={FAKE_QUERY_SECRET}&max_tokens=8192",
        "note": f"known environment value: {FAKE_ENV_SECRET}",
    }
    cfg.source_format = "toml"
    cfg.path.write_text(f"# {COMMENT}\n[model]\nroot = '../models/selected-model'\n")
    effective = {
        "path": str(cfg.path),
        "model": {"root": "/models/selected-model", "tokenizer": "preserved-tokenizer"},
        "dataset": {"path": "/data/selected-images"},
        "sample": {"enabled": True, "prompt": PROMPT, "max_tokens": 8192},
        "training": cfg.training,
        "wandb": cfg.wandb,
        "source_details": copy.deepcopy(cfg.source_document),
    }
    cfg.to_dict = lambda: copy.deepcopy(effective)
    return cfg


def sample_fixture(output, step=2, index=0):
    path = output / "samples" / f"fixture-step-{step}-{index}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = PngImagePlugin.PngInfo()
    text.add_text("unapproved_metadata", FAKE_ENV_SECRET)
    Image.new("RGB", (16, 12), (17, 41, 73)).save(
        path,
        pnginfo=text,
        icc_profile=FAKE_ENV_SECRET.encode(),
        exif=b"Exif\x00\x00" + FAKE_ENV_SECRET.encode(),
    )
    return {
        "path": str(path),
        "prompt": f"{PROMPT}; {FAKE_ENV_SECRET}; https://u:{FAKE_URL_SECRET}@example.invalid/?token={FAKE_QUERY_SECRET}",
        "seed": 42,
        "step": step,
        "index": index,
    }


def read_run(output):
    import wandb

    wandb.teardown()
    journals = list(output.glob("wandb/offline-run-*/run-*.wandb"))
    assert len(journals) == 1
    return journals[0], _offline_records(journals[0])


def history_rows(records):
    rows = []
    for record in records:
        if not record.HasField("history"):
            continue
        row = {}
        for item in record.history.item:
            keys = [item.key] if item.key else list(item.nested_key)
            target = row
            for key in keys[:-1]:
                target = target.setdefault(key, {})
            target[keys[-1]] = json.loads(item.value_json)
        assert record.history.step.num == row["_step"] == row["train/step"]
        rows.append(row)
    return rows


def assert_no_secrets(data):
    if isinstance(data, str):
        data = data.encode()
    for sentinel in (FAKE_ENV_SECRET, FAKE_KEY_SECRET, FAKE_URL_SECRET, FAKE_QUERY_SECRET, COMMENT):
        # Boolean assertion avoids exposing actual payloads in pytest diffs.
        assert sentinel.encode() not in data


def test_sanitizer_nested_credentials_urls_and_noncredential_tokens():
    sanitizer = ConfigSanitizer()
    raw = {
        "apiKey": "sensitive-value",
        "db_password": "sensitive-value",
        "nested": [{"client_secret": "sensitive-value", "HF_TOKEN": "sensitive-value"}],
        "tokenizer": "qwen-tokenizer",
        "max_tokens": 256,
        "token_count": 123,
        "endpoint": "https://user:pass@example.invalid/path?api_key=secret&max_tokens=9#access_token=bad",
        "text": FAKE_ENV_SECRET,
    }
    clean = sanitizer.sanitize(raw)
    assert clean["apiKey"] == clean["db_password"] == REDACTED
    assert clean["nested"] == [{"client_secret": REDACTED, "HF_TOKEN": REDACTED}]
    assert clean["tokenizer"] == "qwen-tokenizer" and clean["max_tokens"] == 256
    assert clean["token_count"] == 123
    assert "user:pass" not in clean["endpoint"] and "=secret" not in clean["endpoint"]
    assert "=bad" not in clean["endpoint"] and "max_tokens=9" in clean["endpoint"]
    assert clean["text"] == REDACTED
    assert raw["apiKey"] == "sensitive-value"  # no mutation of live config


def test_real_config_artifact_and_image_journal_readback(tmp_path, capsys):
    import wandb
    import yaml

    assert wandb.__version__ == "0.30.0"
    cfg = media_config(tmp_path)
    # Tracker must use parsed-at-load snapshot, never re-open this changed file.
    cfg.path.write_text("THIS FILE CHANGED AFTER LOAD: must not appear")
    output = tmp_path / "offline-media-proof"
    sample = sample_fixture(output)
    tracker = TrainingTracker(cfg, fixture_plan(), output)
    with tracker:
        tracker.log(row(1))
        tracker.log(row(2), samples=[sample])  # non-cadence step must survive
        tracker.log({"step": 4}, samples=[sample_fixture(output, step=4, index=1)])
    journal, records = read_run(output)
    rows = history_rows(records)
    assert [entry["train/step"] for entry in rows] == [1, 2, 4]
    assert [entry["_step"] for entry in rows] == [1, 2, 4]
    assert rows[1]["train/loss"] == 0.5
    assert "train/loss" not in rows[2]  # images-only call, no invented metrics
    assert "samples/images" not in rows[0]
    config = next(record.run.config for record in records if record.HasField("run"))
    values = {item.key: json.loads(item.value_json) for item in config.update}
    assert values["rank"] == 4  # old flat hyperparameters retained
    resolved = values["resolved_config"]
    assert resolved["model"]["root"] == "/models/selected-model"
    assert resolved["dataset"]["path"] == "/data/selected-images"
    assert resolved["sample"]["prompt"] == PROMPT
    assert values["source_config"]["model"]["root"] == "../models/selected-model"
    assert values["source_config"]["nested"][0]["max_tokens"] == 8192
    assert values["source_config"]["nested"][0]["accessToken"] == REDACTED
    artifacts = [record.artifact for record in records if record.HasField("artifact")]
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.type == "training-config"
    assert artifact.name == f"config-{tracker.info['run_id']}"
    entries = list(artifact.manifest.contents)
    assert {entry.path for entry in entries} == {
        "effective-config.json",
        "source-config.yaml",
        "snapshot-metadata.json",
    }
    artifact_files = []
    for entry in entries:
        staged = Path(entry.local_path)
        local = output / "wandb-config" / entry.path
        actual = staged.read_bytes()
        assert actual == local.read_bytes()
        assert base64.b64encode(hashlib.md5(actual).digest()).decode() == entry.digest
        assert len(actual) == entry.size
        assert_no_secrets(actual)
        artifact_files.append(
            {
                "name": entry.path,
                "staged": str(staged),
                "bytes": len(actual),
                "digest_verified": True,
            }
        )
    source = yaml.safe_load((output / "wandb-config/source-config.yaml").read_text())
    assert source == values["source_config"]
    assert json.loads((output / "wandb-config/effective-config.json").read_text()) == resolved
    metadata = json.loads((output / "wandb-config/snapshot-metadata.json").read_text())
    assert metadata["source_format"] == "toml" and metadata["comments_preserved"] is False
    image_proofs = []
    for entry in rows[1:]:
        media = entry["samples/images"]
        assert media["_type"] == "images/separated"
        assert media["count"] == 1
        assert str(entry["train/step"]) in media["captions"][0]
        assert "seed=42" in media["captions"][0] and PROMPT in media["captions"][0]
        assert_no_secrets(json.dumps(media))
        image_path = journal.parent / "files" / media["filenames"][0]
        assert image_path.is_file()
        image_bytes = image_path.read_bytes()
        assert_no_secrets(image_bytes)
        with Image.open(image_path) as image:
            assert image.format == "PNG" and image.size == (16, 12)
            assert image.convert("RGB").getpixel((0, 0)) == (17, 41, 73)
            assert "unapproved_metadata" not in image.info
            assert "icc_profile" not in image.info and "exif" not in image.info
        image_proofs.append(
            {"path": str(image_path), "step": entry["_step"], "png_pixel_readback": True}
        )
    serialized = b"".join(record.SerializeToString() for record in records)
    assert_no_secrets(serialized)
    assert b"THIS FILE CHANGED AFTER LOAD" not in serialized
    for path in (output / "wandb").rglob("*"):
        if path.is_file():
            assert_no_secrets(path.read_bytes())
    assert_no_secrets(str(capsys.readouterr()))
    kinds = Counter(record.WhichOneof("record_type") for record in records)
    assert kinds["stats"] == kinds["output"] == kinds["output_raw"] == 0
    assert not list(output.rglob("wandb-metadata.json"))
    assert not list(output.rglob("requirements.txt"))
    assert [record.exit.exit_code for record in records if record.HasField("exit")] == [0]
    report = {
        "real_sdk_version": wandb.__version__,
        "mode": "offline",
        "base_url_loopback": True,
        "synthetic_fixture_png_not_model_generated": True,
        "journal": str(journal),
        "run_id": tracker.info["run_id"],
        "record_counts": dict(kinds),
        "history_steps": [entry["_step"] for entry in rows],
        "same_row_metrics_images": True,
        "images_only_cadence_override": True,
        "config_artifact_files": artifact_files,
        "images": image_proofs,
        "source_snapshot_not_reread": True,
        "credentials_and_comments_absent": True,
    }
    (tmp_path / "config-media-evidence.json").write_text(json.dumps(report, indent=2))


def test_real_optout_no_config_artifact_no_image_read(tmp_path, monkeypatch):
    import wandb

    cfg = fixture_config(log_config=False, log_samples=False, log_every=3)
    cfg.to_dict = Mock(side_effect=AssertionError("config must not be inspected"))
    monkeypatch.setattr(
        "qwen21_trainer.tracking._sample_image",
        Mock(side_effect=AssertionError("must not read images")),
    )
    output = tmp_path / "optout"
    with TrainingTracker(cfg, fixture_plan(), output) as tracker:
        tracker.log(row(1), samples=[{"path": "/not-allowed/external.png"}])
        tracker.log(row(2), samples=[{"path": "/not-allowed/external.png"}])
    _, records = read_run(output)
    cfg.to_dict.assert_not_called()
    assert [entry["_step"] for entry in history_rows(records)] == [1]
    assert all("samples/images" not in entry for entry in history_rows(records))
    assert not any(record.HasField("artifact") for record in records)
    assert not (output / "wandb-config").exists()
    assert not list(output.rglob("*.png"))
    assert wandb.run is None


@pytest.mark.parametrize(
    "kind",
    [
        "outside",
        "file_symlink",
        "directory_symlink",
        "samples_symlink",
        "traversal",
        "not_png",
        "fifo",
        "wrong_step",
    ],
)
def test_real_rejects_unapproved_image_and_finishes_failed(tmp_path, kind):
    output = tmp_path / "rejected"
    sample = sample_fixture(output)
    outside = tmp_path / "external.png"
    Image.new("RGB", (2, 2)).save(outside)
    if kind == "outside":
        sample["path"] = str(outside)
    elif kind == "file_symlink":
        path = output / "samples/link.png"
        path.symlink_to(outside)
        sample["path"] = str(path)
    elif kind == "directory_symlink":
        path = output / "samples/link"
        path.symlink_to(tmp_path, target_is_directory=True)
        sample["path"] = str(path / "external.png")
    elif kind == "samples_symlink":
        Path(sample["path"]).unlink()
        (output / "samples").rmdir()
        (output / "samples").symlink_to(tmp_path, target_is_directory=True)
        sample["path"] = str(output / "samples/external.png")
    elif kind == "traversal":
        sample["path"] = str(output / "samples/../../external.png")
    elif kind == "not_png":
        Path(sample["path"]).write_text("not PNG")
    elif kind == "fifo":
        Path(sample["path"]).unlink()
        os.mkfifo(sample["path"])
    else:
        sample["step"] = 999
    cfg = fixture_config(log_config=False, log_samples=True)
    with pytest.raises(RuntimeError, match="記錄訓練指標失敗") as caught:
        with TrainingTracker(cfg, fixture_plan(), output) as tracker:
            tracker.log(row(2), samples=[sample])
    assert_no_secrets("".join(traceback.format_exception(caught.value)))
    _, records = read_run(output)
    assert history_rows(records) == []
    assert [record.exit.exit_code for record in records if record.HasField("exit")] == [1]
    assert not list((output / "wandb").rglob("*.png"))
    assert tracker.info["exit_code"] == 1


def test_disabled_config_and_samples_are_not_inspected(tmp_path, monkeypatch):
    cfg = fixture_config(enabled=False, log_config=True, log_samples=True)
    cfg.to_dict = Mock(side_effect=AssertionError("must not inspect config"))
    monkeypatch.setattr(
        "qwen21_trainer.tracking.importlib.import_module",
        Mock(side_effect=AssertionError("must not import SDK")),
    )
    output = tmp_path / "not-created"
    with TrainingTracker(cfg, fixture_plan(), output) as tracker:
        tracker.log({}, samples=[{"path": "/do-not-read.png"}])
    cfg.to_dict.assert_not_called()
    assert not output.exists()


def test_no_source_snapshot_never_fabricates_original(tmp_path):
    cfg = Config(
        path=tmp_path / "not-a-real-config.toml", wandb={"enabled": True, "mode": "offline"}
    )
    output = tmp_path / "effective-only"
    with TrainingTracker(cfg, fixture_plan(), output):
        pass
    _, records = read_run(output)
    artifact = next(record.artifact for record in records if record.HasField("artifact"))
    assert {entry.path for entry in artifact.manifest.contents} == {
        "effective-config.json",
        "snapshot-metadata.json",
    }
    metadata = json.loads((output / "wandb-config/snapshot-metadata.json").read_text())
    assert metadata["source_available"] is False
    assert not (output / "wandb-config/source-config.yaml").exists()


@pytest.mark.parametrize("source_format", ["toml", "yaml"])
def test_real_loader_source_snapshot_is_immutable_after_file_change(tmp_path, source_format):
    import yaml

    path = tmp_path / f"source.{source_format}"
    if source_format == "toml":
        document = f'[model]\nroot = "models/picked"\n[sample]\nprompt = {json.dumps(PROMPT)}\n[wandb]\nenabled = true\nmode = "offline"\n'
    else:
        document = yaml.safe_dump(
            {
                "model": {"root": "models/picked"},
                "sample": {"prompt": PROMPT},
                "wandb": {"enabled": True, "mode": "offline"},
            }
        )
    path.write_text(f"# {COMMENT}\n{document}")
    cfg = load_config(path)
    path.write_text("changed since config was loaded")
    output = tmp_path / "loaded-snapshot"
    with TrainingTracker(cfg, fixture_plan(), output):
        pass
    _, records = read_run(output)
    source = yaml.safe_load((output / "wandb-config/source-config.yaml").read_text())
    effective = json.loads((output / "wandb-config/effective-config.json").read_text())
    metadata = json.loads((output / "wandb-config/snapshot-metadata.json").read_text())
    assert source["model"]["root"] == "models/picked"
    assert effective["model"]["root"] == str(tmp_path / "models/picked")
    assert source["sample"]["prompt"] == effective["sample"]["prompt"] == PROMPT
    assert metadata["source_format"] == source_format
    serialized = b"".join(record.SerializeToString() for record in records)
    assert b"changed since config was loaded" not in serialized
    assert_no_secrets(serialized)


def test_artifact_failure_is_private_and_finishes(tmp_path, sdk, capsys):
    fake, run = sdk
    fake.Artifact = Mock()
    run.log_artifact.side_effect = RuntimeError(FAKE_ENV_SECRET)
    with pytest.raises(RuntimeError, match="初始化失敗") as caught:
        with TrainingTracker(media_config(tmp_path), fixture_plan(), tmp_path / "failure"):
            pytest.fail("must fail closed")
    run.finish.assert_called_once_with(exit_code=1)
    assert_no_secrets("".join(traceback.format_exception(caught.value)) + str(capsys.readouterr()))
