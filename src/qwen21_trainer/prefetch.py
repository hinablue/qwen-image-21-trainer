"""Bounded, ordered CPU cache prefetch with spawn-only DataLoader workers."""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


def shuffled_indices(length, count, seed):
    """Exactly the previous private Random(seed), shuffle(list), pop() order."""
    if length < 1:
        raise ValueError("沒有可訓練樣本。")
    rng = random.Random(seed)
    pending = []
    for _ in range(count):
        if not pending:
            pending = list(range(length))
            rng.shuffle(pending)
        yield pending.pop()


class TrainingOrder(Sampler):
    def __init__(self, length, count, seed):
        self.length, self.count, self.seed = length, count, seed

    def __iter__(self):
        return shuffled_indices(self.length, self.count, self.seed)

    def __len__(self):
        return self.count


class CachedTensorDataset(Dataset):
    """Picklable file descriptors only: no GPU tensors, model or pipeline refs."""

    def __init__(self, cache_dir, entries):
        self.cache_dir = str(Path(cache_dir).resolve())
        self.files = []
        for entry in entries:
            name = entry["file"]
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or not name.endswith(".safetensors")
            ):
                raise ValueError("cache file 必須是 safetensors basename，不能含路徑。")
            repeats = entry.get("repeats", 1)
            if type(repeats) is not int or repeats < 1:
                raise ValueError("cache repeats 必須是正整數。")
            self.files.extend([name] * repeats)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        from safetensors.torch import load_file

        # Device is explicit. Workers must never move data to CUDA.
        return load_file(str(Path(self.cache_dir) / self.files[index]), device="cpu")


def _identity(sample):
    # No automatic batching, padding, dtype conversion or mask conversion.
    return sample


def _cpu_worker_init(worker_id):
    # Runs in spawned children only; parent CUDA visibility/RNG are untouched.
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU prefetch worker 不應初始化 CUDA。")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)


@contextmanager
def sample_stream(dataset, settings):
    """Yield one ordered sample stream for the complete finite training budget.

    0 workers follows the old synchronous read path. With workers, a dedicated
    CPU Generator prevents DataLoader base-seed creation from consuming the main
    torch RNG. Only the main process samples diffusion noise/timesteps.
    """
    workers = settings.get("num_workers", 0)
    factor = settings.get("prefetch_factor", 2)
    if type(workers) is not int or workers < 0:
        raise ValueError("training.num_workers 必須是非負整數。")
    if type(factor) is not int or factor < 1:
        raise ValueError("training.prefetch_factor 必須是正整數。")
    if len(dataset) < 1:
        raise ValueError("沒有可訓練樣本。")
    count = settings["max_steps"] * settings["gradient_accumulation_steps"]
    order = TrainingOrder(len(dataset), count, settings["seed"])
    if workers == 0:
        stream = (dataset[index] for index in order)
        try:
            yield stream
        finally:
            stream.close()
        return
    generator = torch.Generator(device="cpu").manual_seed(settings["seed"])
    loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        sampler=order,
        num_workers=workers,
        prefetch_factor=factor,
        collate_fn=_identity,
        worker_init_fn=_cpu_worker_init,
        multiprocessing_context="spawn",
        pin_memory=False,
        persistent_workers=False,
        generator=generator,
        in_order=True,
    )
    iterator = None
    try:
        iterator = iter(loader)
        yield iterator
    finally:
        if iterator is not None:
            # PyTorch 2.12 has no public close API. Explicit shutdown is needed
            # on exceptions/early exits, not only on normal iterator exhaustion.
            # This version is pinned; real-spawn regression tests cover cleanup.
            iterator._shutdown_workers()
