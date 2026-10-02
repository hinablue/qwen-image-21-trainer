"""Krea-style telemetry with real W&B 0.30 offline readback, CPU fixtures only.

Fixture metrics and PNG pixels are synthetic, not model training or samples.
No original console logs, private dataset, credentials or model are collected.
"""

import base64
import csv
import hashlib
import json
import os
import traceback
from collections import Counter
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image
from test_tracking import fixture_config, fixture_plan, row
from test_tracking import sdk as sdk_fixture
from test_tracking_media import (
    FAKE_ENV_SECRET,
    assert_no_secrets,
    history_rows,
    read_run,
    sample_fixture,
)
from test_tracking_media import offline_environment as offline_environment_fixture

import qwen21_trainer.tracking as tracking
from qwen21_trainer.tracking import TrainingTracker

sdk = sdk_fixture
offline_environment = offline_environment_fixture
FIELDS = (
    "step",
    "loss",
    "loss_average",
    "learning_rate",
    "elapsed_seconds",
    "images_seen",
    "data_wait_seconds",
)


def fixture_rows():
    losses = [8.0, 2.0, 5.0, 1.0, 9.0]
    used_lrs = [0.00001, 0.00003, 0.00002, 0.00001, 0.0]
    result = []
    for step, (loss, lr) in enumerate(zip(losses, used_lrs), 1):
        event = {key: value for key, value in row(step).items() if key in FIELDS}
        event.update(loss=loss, loss_average=sum(losses[:step]) / step, learning_rate=lr)
        result.append(event)
    return result


def write_csv(output, rows=None):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(fixture_rows() if rows is None else rows)


def artifacts(records):
    return [record.artifact for record in records if record.HasField("artifact")]


def assert_private_records(records):
    assert_no_secrets(b"".join(record.SerializeToString() for record in records))
    kinds = Counter(record.WhichOneof("record_type") for record in records)
    assert kinds["stats"] == kinds["output"] == kinds["output_raw"] == 0
    return kinds


def test_real_offline_aliases_media_and_numeric_csv_readback(tmp_path, capsys):
    import wandb

    assert wandb.__version__ == "0.30.0"
    output = tmp_path / "offline-krea-style"
    cfg = fixture_config(log_samples=True, log_training_log=True, log_every=3)
    samples = [sample_fixture(output, index=i) for i in (0, 2)]
    # Distinct synthetic pixels; identical contents are SDK-deduplicated too.
    Image.new("RGB", (16, 12), (71, 41, 73)).save(samples[1]["path"])
    rows = fixture_rows()
    with TrainingTracker(cfg, fixture_plan(), output) as tracker:
        # This writer is deliberately buffered; tracker.__exit__ runs after
        # close(), exactly like the training caller's required nesting contract.
        with (output / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for event in rows:
                writer.writerow(event)
                tracker.log(
                    {**event, "next_learning_rate": 999.0},
                    samples=samples if event["step"] == 2 else None,
                )
    journal, records = read_run(output)
    histories = history_rows(records)
    assert [entry["_step"] for entry in histories] == [1, 2, 3, 5]
    for entry in histories:
        expected = rows[entry["_step"] - 1]
        assert entry["loss/current"] == entry["train/loss"] == expected["loss"]
        assert entry["lr/dit"] == entry["train/learning_rate"] == expected["learning_rate"]
        assert entry["loss/average"] == expected["loss_average"]
        assert "next_learning_rate" not in entry
    # Last cumulative average includes the omitted step 4: not average of logs.
    assert histories[-1]["loss/average"] == sum(event["loss"] for event in rows) / len(rows)
    assert histories[-1]["loss/average"] != sum(entry["loss/current"] for entry in histories) / len(
        histories
    )
    media = histories[1]["samples/images"]
    assert media["count"] == len(samples)
    paths = []
    for position, sample in enumerate(samples):
        alias = histories[1][f"sample_{sample['index']}"]
        assert alias["_type"] == "image-file"
        assert alias["path"] == media["filenames"][position]
        path = journal.parent / "files" / alias["path"]
        data = path.read_bytes()
        assert_no_secrets(data)
        assert hashlib.sha256(data).hexdigest() == alias["sha256"]
        with Image.open(path) as image:
            assert image.size == (16, 12)
            assert image.convert("RGB").getpixel((0, 0)) == (
                (17, 41, 73) if position == 0 else (71, 41, 73)
            )
            assert not {"exif", "icc_profile", "unapproved_metadata"}.intersection(image.info)
        assert_no_secrets(alias["caption"])
        paths.append(path)
    assert set((journal.parent / "files/media/images").rglob("*.png")) == set(paths)
    assert len(paths) == len(set(paths)) == len(samples)
    metrics = [record.metric for record in records if record.HasField("metric")]
    by_glob = {metric.glob_name: metric for metric in metrics if metric.glob_name}
    for name in ("train/*", "loss/*", "lr/*", "samples/*", "sample_*"):
        assert by_glob[name].step_metric == "train/step"
    logged = artifacts(records)
    assert len(logged) == 1
    artifact = logged[0]
    assert artifact.type == "training-log"
    assert artifact.name == f"training-log-{tracker.info['run_id']}"
    assert "cumulative optimizer-update mean" in artifact.description
    assert "not Krea's epoch moving average" in artifact.description
    entries = list(artifact.manifest.contents)
    assert len(entries) == 1 and entries[0].path == "metrics.csv"
    entry = entries[0]
    staged = Path(entry.local_path)
    local = output / "wandb-logs/metrics.csv"
    data = staged.read_bytes()
    assert data == local.read_bytes()
    assert entry.size == len(data)
    assert entry.digest == base64.b64encode(hashlib.md5(data).digest()).decode()
    with local.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(FIELDS)
        saved = list(reader)
    assert len(saved) == len(rows)
    for stored, expected in zip(saved, rows):
        assert {key: float(value) for key, value in stored.items()} == expected
    assert_no_secrets(data)
    kinds = assert_private_records(records)
    assert [record.exit.exit_code for record in records if record.HasField("exit")] == [0]
    assert tracker.info["training_log_status"] == "recorded"
    for path in (output / "wandb").rglob("*"):
        if path.is_file():
            assert_no_secrets(path.read_bytes())
    assert not list(output.rglob("wandb-metadata.json"))
    assert not list(output.rglob("requirements.txt"))
    assert_no_secrets(str(capsys.readouterr()))
    report = {
        "sdk_version": wandb.__version__,
        "mode": "offline",
        "loopback_only": True,
        "synthetic_fixtures_not_training": True,
        "run_id": tracker.info["run_id"],
        "journal": str(journal),
        "journal_crc_verified": True,
        "history_steps": [entry["_step"] for entry in histories],
        "loss_average_semantics": "run cumulative optimizer-update average, not epoch moving average",
        "history_rows": histories,
        "record_counts": dict(kinds),
        "image_files": [str(path) for path in paths],
        "image_aliases_reuse_files": True,
        "training_log_staged": str(staged),
        "training_log_snapshot": str(local),
        "training_log_rows": len(saved),
        "manifest_digest_verified": True,
        "manifest_bytes_verified": True,
        "private_sentinels_absent": True,
    }
    (tmp_path / "krea-style-evidence.json").write_text(json.dumps(report, indent=2))


def test_average_not_fabricated_and_legacy_keys_preserved(tmp_path, sdk):
    with TrainingTracker(fixture_config(), fixture_plan(), tmp_path) as tracker:
        tracker.log(row(1))
    payload = sdk[1].log.call_args.args[0]
    assert set(payload) == {
        "train/step",
        "train/loss",
        "train/learning_rate",
        "train/elapsed_seconds",
        "train/images_seen",
        "train/data_wait_seconds",
        "train/steps_per_second",
        "loss/current",
        "lr/dit",
    }
    assert "loss/average" not in payload
    sdk[1].log_artifact.assert_not_called()


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), FAKE_ENV_SECRET])
def test_invalid_average_is_not_uploaded(tmp_path, sdk, bad):
    with pytest.raises(ValueError):
        with TrainingTracker(fixture_config(), fixture_plan(), tmp_path) as tracker:
            tracker.log({**row(1), "loss_average": bad})
    sdk[1].log.assert_not_called()
    sdk[1].finish.assert_called_once_with(exit_code=1)


def test_real_sample_duplicate_index_is_rejected_before_encoding(tmp_path):
    output = tmp_path / "duplicate"
    sample = sample_fixture(output)
    with pytest.raises(RuntimeError, match="記錄訓練指標失敗"):
        with TrainingTracker(fixture_config(log_samples=True), fixture_plan(), output) as tracker:
            tracker.log(row(2), samples=[sample, dict(sample)])
    _, records = read_run(output)
    assert history_rows(records) == []
    assert not list((output / "wandb").rglob("*.png"))
    assert [record.exit.exit_code for record in records if record.HasField("exit")] == [1]
    assert_private_records(records)


def test_sample_aliases_reuse_identical_image_objects(tmp_path, sdk, monkeypatch):
    images = [object(), object()]
    encoder = Mock(side_effect=images)
    monkeypatch.setattr(tracking, "_sample_image", encoder)
    with TrainingTracker(fixture_config(log_samples=True), fixture_plan(), tmp_path) as tracker:
        tracker.log(row(2), samples=[{"index": 0}, {"index": 4}])
    assert encoder.call_count == 2
    sdk[1].log.assert_called_once()
    payload = sdk[1].log.call_args.args[0]
    assert payload["sample_0"] is payload["samples/images"][0] is images[0]
    assert payload["sample_4"] is payload["samples/images"][1] is images[1]
    assert payload["train/step"] == sdk[1].log.call_args.kwargs["step"] == 2


def test_disabled_training_log_never_reads_or_creates_files(tmp_path, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("must not read metrics"))
    monkeypatch.setattr(tracking, "_training_log_snapshot", forbidden)
    output = tmp_path / "not-created"
    with TrainingTracker(
        fixture_config(enabled=False, log_training_log=True), fixture_plan(), output
    ) as tracker:
        tracker.log({})
    forbidden.assert_not_called()
    assert not output.exists()


def test_real_optout_never_reads_csv(tmp_path, monkeypatch):
    output = tmp_path / "optout"
    output.mkdir()
    (output / "metrics.csv").symlink_to(tmp_path / "never-read-secret")
    forbidden = Mock(side_effect=AssertionError("must not read metrics"))
    monkeypatch.setattr(tracking, "_training_log_snapshot", forbidden)
    with TrainingTracker(fixture_config(log_training_log=False), fixture_plan(), output) as tracker:
        tracker.log(row(1))
    _, records = read_run(output)
    forbidden.assert_not_called()
    assert artifacts(records) == []
    assert not (output / "wandb-logs").exists()
    assert_private_records(records)


def test_real_missing_csv_never_fabricates_artifact(tmp_path):
    output = tmp_path / "missing"
    with TrainingTracker(fixture_config(log_training_log=True), fixture_plan(), output) as tracker:
        tracker.log(row(1))
    _, records = read_run(output)
    assert artifacts(records) == []
    assert tracker.info["training_log_status"] == "missing"
    assert not (output / "wandb-logs").exists()


@pytest.mark.parametrize(
    "kind",
    [
        "symlink",
        "broken_symlink",
        "output_symlink",
        "snapshot_dir_symlink",
        "snapshot_file_symlink",
        "directory",
        "fifo",
        "extra_header",
        "duplicate_header",
        "no_step",
        "nan",
        "inf",
        "overflow",
        "secret",
        "zero_step",
        "negative_step",
        "negative_images",
        "fractional_images",
        "missing_cell",
        "extra_cell",
        "empty",
        "blank_row",
        "bool",
        "invalid_utf8",
    ],
)
def test_real_invalid_training_log_is_private_and_finishes(tmp_path, capsys, kind):
    output = tmp_path / "rejected"
    write_csv(output)
    source = output / "metrics.csv"
    outside = tmp_path / "private.csv"
    outside.write_text(FAKE_ENV_SECRET)
    if kind in {"symlink", "broken_symlink", "directory", "fifo"}:
        source.unlink()
        if kind == "symlink":
            source.symlink_to(outside)
        elif kind == "broken_symlink":
            source.symlink_to(tmp_path / "missing-private.csv")
        elif kind == "directory":
            source.mkdir()
        else:
            os.mkfifo(source)
    elif kind == "output_symlink":
        linked = tmp_path / "linked"
        linked.symlink_to(output, target_is_directory=True)
        output = linked
    elif kind == "snapshot_dir_symlink":
        (output / "wandb-logs").symlink_to(tmp_path, target_is_directory=True)
    elif kind == "snapshot_file_symlink":
        (output / "wandb-logs").mkdir()
        (output / "wandb-logs/metrics.csv").symlink_to(outside)
    else:
        content = {
            "extra_header": f"step,loss,api_key\n1,2,{FAKE_ENV_SECRET}\n",
            "duplicate_header": "step,loss,loss\n1,2,3\n",
            "no_step": "loss\n2\n",
            "nan": "step,loss\n1,NaN\n",
            "inf": "step,loss\n1,inf\n",
            "overflow": "step,loss\n1,1e999\n",
            "secret": f"step,loss\n1,{FAKE_ENV_SECRET}\n",
            "zero_step": "step,loss\n0,2\n",
            "negative_step": "step,loss\n-1,2\n",
            "negative_images": "step,images_seen\n1,-1\n",
            "fractional_images": "step,images_seen\n1,1.5\n",
            "missing_cell": "step,loss\n1\n",
            "extra_cell": "step,loss\n1,2,3\n",
            "empty": "",
            "blank_row": "step,loss\n\n",
            "bool": "step,loss\n1,true\n",
            "invalid_utf8": "step,loss\n1,2\n",
        }[kind]
        source.write_bytes(content.encode() + (b"\xff" if kind == "invalid_utf8" else b""))
    with pytest.raises(RuntimeError, match="training-log 保存失敗") as caught:
        with TrainingTracker(
            fixture_config(log_training_log=True), fixture_plan(), output
        ) as tracker:
            tracker.log(row(1))
    assert_no_secrets("".join(traceback.format_exception(caught.value)) + str(capsys.readouterr()))
    _, records = read_run(output)
    assert artifacts(records) == []
    assert [record.exit.exit_code for record in records if record.HasField("exit")] == [1]
    assert tracker.info["training_log_status"] == "failed"
    assert tracker.info["exit_code"] == 1
    assert_private_records(records)
    for path in (output / "wandb").rglob("*"):
        if path.is_file():
            assert_no_secrets(path.read_bytes())
    assert outside.read_text() == FAKE_ENV_SECRET


@pytest.mark.parametrize("error", [RuntimeError("fixture training failure"), KeyboardInterrupt()])
def test_real_failed_training_still_flushes_numeric_csv(tmp_path, error):
    output = tmp_path / "interrupted"
    with pytest.raises(type(error)) as caught:
        with TrainingTracker(
            fixture_config(log_training_log=True), fixture_plan(), output
        ) as tracker:
            with (output / "metrics.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(fixture_rows()[0])
                tracker.log(fixture_rows()[0])
                raise error
    assert caught.value is error
    _, records = read_run(output)
    assert len(artifacts(records)) == 1
    entry = artifacts(records)[0].manifest.contents[0]
    actual = Path(entry.local_path).read_bytes()
    assert actual == (output / "wandb-logs/metrics.csv").read_bytes()
    assert base64.b64encode(hashlib.md5(actual).digest()).decode() == entry.digest
    assert [record.exit.exit_code for record in records if record.HasField("exit")] == [1]
    assert tracker.info["training_log_status"] == "recorded"
    assert_private_records(records)


@pytest.mark.parametrize("operation", ["snapshot", "artifact", "add_file", "log_artifact"])
@pytest.mark.parametrize("training_error", [False, True])
@pytest.mark.parametrize("finish_error", [False, True])
def test_artifact_error_always_finishes_and_preserves_original(
    tmp_path,
    sdk,
    monkeypatch,
    capsys,
    operation,
    training_error,
    finish_error,
):
    fake, run = sdk
    fake.Artifact = Mock()
    write_csv(tmp_path)

    def fail(*args, **kwargs):
        print(FAKE_ENV_SECRET)
        raise RuntimeError(FAKE_ENV_SECRET)

    if operation == "snapshot":
        monkeypatch.setattr(tracking, "_training_log_snapshot", fail)
    elif operation == "artifact":
        fake.Artifact.side_effect = fail
    elif operation == "add_file":
        fake.Artifact.return_value.add_file.side_effect = fail
    else:
        run.log_artifact.side_effect = fail
    if finish_error:
        run.finish.side_effect = fail
    original = ValueError("original fixture training failure")
    with pytest.raises(ValueError if training_error else RuntimeError) as caught:
        with TrainingTracker(
            fixture_config(log_training_log=True), fixture_plan(), tmp_path
        ) as tracker:
            if training_error:
                raise original
    if training_error:
        assert caught.value is original
    run.finish.assert_called_once_with(exit_code=1)
    assert tracker.info["training_log_status"] == "failed"
    assert "training_log_error" in tracker.info
    assert_no_secrets("".join(traceback.format_exception(caught.value)) + str(capsys.readouterr()))
    assert_no_secrets(json.dumps(tracker.info))


def test_sdk_init_failure_does_not_try_stale_csv(tmp_path, sdk, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("must not read stale metrics"))
    monkeypatch.setattr(tracking, "_training_log_snapshot", forbidden)
    sdk[1].define_metric.side_effect = RuntimeError(FAKE_ENV_SECRET)
    with pytest.raises(RuntimeError, match="初始化失敗"):
        with TrainingTracker(fixture_config(log_training_log=True), fixture_plan(), tmp_path):
            pytest.fail("body must not run")
    forbidden.assert_not_called()
    sdk[1].finish.assert_called_once_with(exit_code=1)
