"""Opt-in training samples, sharing only the live (already adapted) DiT.

TE/VAE are loaded per sample group; this deliberately costs additional memory.
CPU-offload/VRAM-managed training is not supported. The training pipeline/scheduler
never runs inference, and this module never freezes or moves the shared DiT.
"""

from __future__ import annotations

import gc
import hashlib
import io
import json
import os
import random
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from itertools import chain
from pathlib import Path


@contextmanager
def _training_state(model, device):
    """Include construction and cleanup in RNG/mode isolation, also on failure."""
    import numpy as np
    import torch

    modes = [(module, module.training) for module in model.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    devices = {
        tensor.device.index
        for tensor in chain(model.parameters(), model.buffers())
        if tensor.device.type == "cuda"
    }
    device = torch.device(device)
    if device.type == "cuda":
        devices.add(device.index if device.index is not None else torch.cuda.current_device())
    # The pinned DiT moves this non-buffer list during forward. Restore its
    # original reference, too; KV caches are explicitly disabled below.
    rope = getattr(model.pipe.dit, "pos_embed", None)
    freqs = getattr(rope, "freqs", None)
    try:
        with torch.random.fork_rng(devices=sorted(devices)), torch.no_grad():
            model.eval()
            yield devices
    finally:
        for module, training in modes:
            # Calling train() recursively would destroy mixed child modes.
            module.training = training
        if freqs is not None:
            rope.freqs = freqs
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


@contextmanager
def _capture_hooks(pipe):
    """Remove only new upstream TE norm hooks, never pre-existing hooks."""
    norm = getattr(pipe, "text_encoder", None)
    for name in ("model", "model", "language_model", "norm"):
        norm = getattr(norm, name, None)
    hooks = getattr(norm, "_forward_hooks", None)
    existing = set(hooks) if hooks is not None else set()
    try:
        yield
    finally:
        if hooks is not None:
            for key in set(hooks) - existing:
                del hooks[key]
                # Keep PyTorch's ancillary hook registries consistent.
                for name in ("_forward_hooks_with_kwargs", "_forward_hooks_always_called"):
                    getattr(norm, name, {}).pop(key, None)


@contextmanager
def _sample_directory(output_dir):
    """Walk with directory FDs: no symlinks, traversal, or check/open race."""
    root = Path(output_dir).expanduser().absolute()
    if ".." in root.parts:
        raise ValueError("Sample output path must not contain '..'.")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(root.anchor, flags)
    try:
        for part in (*root.parts[1:], "samples"):
            try:
                os.mkdir(part, dir_fd=fd)
            except FileExistsError:
                pass
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        yield root / "samples", fd
    finally:
        os.close(fd)


@contextmanager
def _step_files(directory_fd, names):
    """Reserve every destination exclusively; manifest commits the group last.

    On a Python exception remove only files created by this invocation. A crash
    may leave a partial group without a manifest; subsequent runs fail closed
    rather than overwriting those files.
    """
    files = {}
    success = False
    try:
        for name in names:
            fd = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory_fd,
            )
            files[name] = os.fdopen(fd, "wb")
        yield files
        for handle in files.values():
            handle.flush()
            os.fsync(handle.fileno())
        os.fsync(directory_fd)
        success = True
    finally:
        for name, handle in files.items():
            try:
                if not success:
                    # Do not delete a different entry substituted by another process.
                    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    created = os.fstat(handle.fileno())
                    if (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino):
                        os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            finally:
                handle.close()


class TrainingSampler:
    """Generate once per eligible optimizer step, never once per microstep."""

    def __init__(self, cfg, assets, model, output_dir):
        self.cfg = cfg
        self.assets = assets
        self.model = model
        self.output_dir = Path(output_dir)
        self.enabled = cfg.sample.get("enabled", False)
        self._completed = set()
        if self.enabled and cfg.model.get("cpu_offload", False):
            raise ValueError("Training sampling does not support model.cpu_offload=True.")

    def should_sample(self, step) -> bool:
        return (
            self.enabled
            and type(step) is int
            and 0 < step <= self.cfg.training["max_steps"]
            and step not in self._completed
            and (
                step % self.cfg.sample.get("every", 100) == 0
                or step == self.cfg.training["max_steps"]
            )
        )

    def generate(self, step) -> list[dict]:
        if not self.should_sample(step):
            return []

        import torch
        from PIL import Image
        from tqdm import tqdm

        from .runtime import build_pipeline

        if getattr(self.model, "_offload_manager", None) is not None:
            raise ValueError("Training sampling does not support an offload manager.")
        train_pipe = self.model.pipe
        if getattr(train_pipe, "vram_management_enabled", False):
            raise ValueError("Training sampling does not support managed VRAM pipelines.")
        sample = self.cfg.sample
        prompts = list(sample.get("prompts") or [sample["prompt"]])
        stem = f"step-{step:06d}"
        image_names = [f"{stem}-{index:02d}.png" for index in range(len(prompts))]
        manifest_name = f"{stem}.json"
        records = []
        with _sample_directory(self.output_dir) as (folder, directory_fd):
            with _step_files(directory_fd, [*image_names, manifest_name]) as files:
                # A cumulative diagnostic log is append-only, never followed through
                # a symlink. PNGs and manifests are always exclusive/no-overwrite.
                log_fd = os.open(
                    "loading.log",
                    os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory_fd,
                )
                with os.fdopen(log_fd, "a", encoding="utf-8") as log:
                    with _training_state(self.model, self.cfg.model["device"]) as devices:
                        pipe, bars = None, []
                        try:
                            with redirect_stdout(log), redirect_stderr(log):
                                print(f"\n[{stem}] loading text_encoder and vae only")
                                pipe = build_pipeline(
                                    self.cfg, self.assets, ["text_encoder", "vae"]
                                )
                            if pipe is train_pipe:
                                pipe = None  # Never detach or clean up the training pipeline.
                                raise RuntimeError("Sampling requires an independent pipeline.")
                            if pipe.scheduler is train_pipe.scheduler:
                                raise RuntimeError("Sampling requires an independent scheduler.")
                            if pipe.dit is not None:
                                raise RuntimeError(
                                    "Sampling loader unexpectedly loaded a second DiT."
                                )
                            if getattr(pipe, "vram_management_enabled", False):
                                raise ValueError(
                                    "Sampling pipeline VRAM management is unsupported."
                                )
                            pipe.dit = train_pipe.dit
                            pipe.eval()  # Do NOT call requires_grad_(False) on this shared object.

                            def progress_bar_cmd(items):
                                bar = tqdm(
                                    items,
                                    desc=f"Sample {step}",
                                    unit="denoise",
                                    leave=False,
                                    dynamic_ncols=True,
                                    mininterval=0.5,
                                    disable=None,
                                )
                                bars.append(bar)
                                return bar

                            for index, (prompt, name) in enumerate(zip(prompts, image_names)):
                                with _capture_hooks(pipe):
                                    image = pipe(
                                        prompt=prompt,
                                        negative_prompt=sample["negative_prompt"],
                                        width=sample["width"],
                                        height=sample["height"],
                                        num_inference_steps=sample["steps"],
                                        cfg_scale=sample["cfg_scale"],
                                        seed=sample["seed"],
                                        rand_device="cpu",
                                        use_kv_cache=False,
                                        tiled=True,
                                        use_flex_attention=self.cfg.model["attention"] == "flex",
                                        progress_bar_cmd=progress_bar_cmd,
                                    )
                                if not isinstance(image, Image.Image):
                                    raise TypeError("Sampling pipeline must return a PIL image.")
                                try:
                                    if image.size != (sample["width"], sample["height"]):
                                        raise ValueError(
                                            "Sample image dimensions do not match config."
                                        )
                                    buffer = io.BytesIO()
                                    image.save(buffer, format="PNG")
                                    encoded = buffer.getvalue()
                                finally:
                                    image.close()
                                files[name].write(encoded)
                                files[name].flush()
                                os.fsync(files[name].fileno())
                                records.append(
                                    {
                                        "path": str(folder / name),
                                        "prompt": prompt,
                                        "seed": sample["seed"],
                                        "step": step,
                                        "index": index,
                                        "width": sample["width"],
                                        "height": sample["height"],
                                        "sha256": hashlib.sha256(encoded).hexdigest(),
                                    }
                                )
                            manifest = {
                                "schema_version": 1,
                                "step": step,
                                "negative_prompt": sample["negative_prompt"],
                                "num_inference_steps": sample["steps"],
                                "cfg_scale": sample["cfg_scale"],
                                "vae_tiled": True,
                                "samples": records,
                            }
                            files[manifest_name].write(
                                json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
                            )
                        finally:
                            for bar in bars:
                                bar.close()
                            if pipe is not None:
                                # Never .to('cpu'), .offload(), or destroy the shared DiT.
                                # Break its reference before releasing temporary TE/VAE.
                                pipe.dit = None
                                pipe.text_encoder = None
                                pipe.vae = None
                                pipe.processor = None
                            del pipe
                            gc.collect()
                            for device in devices:
                                with torch.cuda.device(device):
                                    torch.cuda.empty_cache()
        self._completed.add(step)
        return records
