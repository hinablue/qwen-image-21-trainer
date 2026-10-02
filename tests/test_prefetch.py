"""CPU-only, real spawn workers; order/RNG/cleanup, not GPU speed claims."""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import time

import pytest
import torch
from safetensors.torch import save_file
from torch.utils.data import Dataset

from qwen21_trainer.config import Config
from qwen21_trainer.prefetch import CachedTensorDataset, sample_stream, shuffled_indices


class ProbeDataset(Dataset):
    def __init__(self, length=7, counter=None, ready=None):
        self.length = length
        self.counter = counter
        self.ready = ready

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.counter is not None:
            with self.counter.get_lock():
                self.counter.value += 1
                if self.counter.value >= 4:
                    self.ready.set()
        return {
            "index": index,
            "pid": os.getpid(),
            "threads": torch.get_num_threads(),
            "cuda_initialized": torch.cuda.is_initialized(),
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "value": torch.tensor([index], dtype=torch.bfloat16),
        }


def settings(workers, **overrides):
    return {
        "num_workers": workers,
        "prefetch_factor": 2,
        "seed": 17415,
        "max_steps": 8,
        "gradient_accumulation_steps": 3,
        **overrides,
    }


def original_order(length, count, seed):
    rng = random.Random(seed)
    indices, result = [], []
    for _ in range(count):
        if not indices:
            indices = list(range(length))
            rng.shuffle(indices)
        result.append(indices.pop())
    return result


def test_sampler_matches_previous_shuffle_pop_across_cycles():
    assert list(shuffled_indices(7, 24, 17415)) == original_order(7, 24, 17415)


@pytest.mark.parametrize("workers", [0, 2])
def test_order_cpu_and_main_rng_are_preserved(workers):
    torch.manual_seed(42)
    random.seed(99)
    before_torch, before_python = torch.get_rng_state().clone(), random.getstate()
    with sample_stream(ProbeDataset(), settings(workers)) as stream:
        results = list(stream)
    assert [x["index"] for x in results] == original_order(7, 24, 17415)
    assert torch.equal(before_torch, torch.get_rng_state())
    assert before_python == random.getstate()
    assert all(
        x["value"].device.type == "cpu" and x["value"].dtype == torch.bfloat16 for x in results
    )
    if workers:
        assert len({x["pid"] for x in results}) == workers
        assert all(x["pid"] != os.getpid() and x["threads"] == 1 for x in results)
        assert all(not x["cuda_initialized"] and x["cuda_visible"] == "" for x in results)
    else:
        assert {x["pid"] for x in results} == {os.getpid()}


def test_prefetch_window_bounded_and_workers_closed_on_early_exit():
    ctx = mp.get_context("spawn")
    counter, ready = ctx.Value("i", 0), ctx.Event()
    with sample_stream(ProbeDataset(50, counter, ready), settings(2, max_steps=20)) as stream:
        workers = list(stream._workers)
        assert ready.wait(30), "spawn workers did not prefetch without a consumer"
        time.sleep(0.1)
        assert counter.value == 4  # two workers, two outstanding tasks each
        first = next(stream)
        assert first["index"] == original_order(50, 1, 17415)[0]
    for process in workers:
        process.join(timeout=5)
        assert not process.is_alive()


def test_cached_dataset_preserves_repeats_shapes_and_bool_masks(tmp_path):
    for i in range(2):
        save_file(
            {
                "input_latents": torch.full((1, 64, 4, 6), i, dtype=torch.bfloat16),
                "prompt_embeds": torch.ones(1, i + 3, 8, dtype=torch.bfloat16),
                "edit_image_pad_mask": torch.zeros(1, i + 3, dtype=torch.bool),
            },
            str(tmp_path / f"{i}.safetensors"),
        )
    dataset = CachedTensorDataset(
        tmp_path, [{"file": "0.safetensors", "repeats": 2}, {"file": "1.safetensors", "repeats": 1}]
    )
    assert len(dataset) == 3
    assert [int(dataset[i]["input_latents"][0, 0, 0, 0]) for i in range(3)] == [0, 0, 1]
    with sample_stream(dataset, settings(2, max_steps=2, gradient_accumulation_steps=2)) as stream:
        rows = list(stream)
    expected = original_order(3, 4, 17415)
    assert [int(x["input_latents"][0, 0, 0, 0]) for x in rows] == [[0, 0, 1][i] for i in expected]
    assert all(x["input_latents"].shape == (1, 64, 4, 6) for x in rows)
    assert all(x["edit_image_pad_mask"].dtype == torch.bool for x in rows)


def test_worker_read_failure_propagates_and_closes(tmp_path):
    dataset = CachedTensorDataset(tmp_path, [{"file": "missing.safetensors"}])
    workers = []
    with pytest.raises(FileNotFoundError):
        with sample_stream(dataset, settings(2)) as stream:
            workers = list(stream._workers)
            next(stream)
    for process in workers:
        process.join(timeout=5)
        assert not process.is_alive()


@pytest.mark.parametrize(
    "update",
    [
        {"num_workers": -1},
        {"num_workers": True},
        {"num_workers": 1.5},
        {"prefetch_factor": 0},
        {"prefetch_factor": True},
        {"prefetch_factor": "2"},
    ],
)
def test_bad_config_rejected(tmp_path, update):
    with pytest.raises(ValueError):
        Config(path=tmp_path / "config.toml", training=update)


def test_default_prefetch_is_opt_in(tmp_path):
    cfg = Config(path=tmp_path / "config.toml")
    assert cfg.training["num_workers"] == 0
    assert cfg.training["prefetch_factor"] == 2


@pytest.mark.parametrize("name", ["../escape.safetensors", "/tmp/escape.safetensors"])
def test_cached_dataset_rejects_path_escape(tmp_path, name):
    with pytest.raises(ValueError):
        CachedTensorDataset(tmp_path, [{"file": name}])
