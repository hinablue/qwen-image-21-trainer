"""Pinned DiffSynth runtime; one training process with optional CPU prefetch."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path

from tqdm import tqdm

from . import UPSTREAM_REVISION


def atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def model_assets(cfg):
    from .model_io import model_assets as discover

    return discover(cfg)


def source_signature(cfg, assets):
    from .model_io import source_signature as signature

    return signature(cfg, assets)


def check_device(cfg):
    import torch

    device = cfg.model["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("設定為 cuda，但此環境看不到 CUDA GPU；可先執行 smoke-test（CPU）。")
    if device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("此 GPU 不支援本訓練器要求的 BF16。")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "只支援單一訓練 process（可另開 CPU 預取 workers）；請直接用 run.sh，不要用多程序 accelerate launch。"
        )
    return torch.device(device)


def build_pipeline(cfg, assets, roles, *, training=False):
    from .model_io import build_pipeline as load

    return load(cfg, assets, roles, training=training)


def _cache_identity(manifest, model_signature):
    return {
        "schema_version": 1,
        "upstream_revision": UPSTREAM_REVISION,
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "model_signature": model_signature,
        "latent_mode": "posterior_mean",
        "dtype": "bfloat16",
        "task": "text-to-image",
    }


def cache_dataset(cfg, *, overwrite=False):
    import torch
    from diffsynth.core import UnifiedDataset
    from safetensors.torch import load_file, save_file

    from .data import load_manifest

    manifest = load_manifest(cfg)
    assets = model_assets(cfg)
    identity = _cache_identity(manifest, source_signature(cfg, assets))
    cache_dir = Path(cfg.training["cache_dir"])
    index_path = cache_dir / "cache-index.json"
    previous = json.loads(index_path.read_text()) if index_path.exists() else None
    if previous and previous["identity"] != identity and not overwrite:
        raise ValueError("cache 身分已改變；請執行 cache --overwrite 重新編碼。")
    # Record identity before starting; a failed run can safely resume matching items.
    pending = cache_dir / "cache-pending.json"
    previous_pending = json.loads(pending.read_text()) if pending.exists() else None
    reusable = not overwrite and previous is not None and previous.get("identity") == identity
    reusable = reusable or (not overwrite and previous_pending == identity)
    if previous_pending and previous_pending != identity and not overwrite:
        raise ValueError("存在另一份資料／模型的未完成 cache；請使用 cache --overwrite。")
    atomic_json(pending, identity)
    pipe = None
    entries = []
    operators = {
        max_pixels: UnifiedDataset.default_image_operator(
            max_pixels=max_pixels,
            height_division_factor=32,
            width_division_factor=32,
            convert_RGB=False,
            convert_RGBA=True,
        )
        for max_pixels in {
            item.get("max_pixels", cfg.dataset["max_pixels"]) for item in manifest["items"]
        }
    }
    progress = tqdm(
        total=len(manifest["items"]),
        desc="Cache TE/VAE",
        unit="img",
        dynamic_ncols=True,
        mininterval=0.5,
        disable=None,
    )
    encoded = reused = 0
    try:
        for item in manifest["items"]:
            dest = cache_dir / f"{item['id']}.safetensors"
            same_identity = False
            if reusable and dest.is_file():
                from safetensors import safe_open

                with safe_open(str(dest), framework="pt") as handle:
                    metadata = handle.metadata() or {}
                same_identity = (
                    json.loads(metadata.get("identity", "{}")) == identity
                    and metadata.get("item_id") == item["id"]
                )
            if same_identity:
                # Interrupted rebuilds may leave old files: verify each file's identity.
                tensors = load_file(str(dest))
                validate_tensors(tensors)
            else:
                if pipe is None:
                    pipe = build_pipeline(cfg, assets, ["text_encoder", "vae"])
                    pipe.scheduler.set_timesteps(1000, training=True)
                    pipe.requires_grad_(False)
                    pipe.eval()
                image = operators[item.get("max_pixels", cfg.dataset["max_pixels"])](item["image"])
                inputs_shared = {
                    "cfg_scale": 1,
                    "edit_image": None,
                    "input_image": image,
                    "height": image.height,
                    "width": image.width,
                    "seed": cfg.training["seed"],
                    "rand_device": "cpu",
                    "tiled": False,
                    "tile_size": 256,
                    "tile_stride": 192,
                    "use_kv_cache": False,
                    "use_flex_attention": False,
                }
                inputs = (inputs_shared, {"prompt": item["prompt"]}, {"negative_prompt": ""})
                # The pinned upstream TE adds a capture hook per call without removing it.
                # Remove ONLY newly-added hooks after each encode, preserving existing ones.
                norm = pipe.text_encoder.model.model.language_model.norm
                existing_hooks = set(norm._forward_hooks)
                try:
                    with torch.no_grad():
                        for unit in pipe.units:
                            inputs = pipe.unit_runner(unit, pipe, *inputs)
                finally:
                    for key in set(norm._forward_hooks) - existing_hooks:
                        del norm._forward_hooks[key]
                shared, positive, _ = inputs
                tensors = {
                    "input_latents": shared["input_latents"],
                    "prompt_embeds": positive["prompt_embeds"],
                    "edit_image_pad_mask": positive["edit_image_pad_mask"],
                }
                if positive["prompt_embeds_mask"] is not None:
                    tensors["prompt_embeds_mask"] = positive["prompt_embeds_mask"]
                tensors = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
                validate_tensors(tensors)
                tmp = dest.with_suffix(".safetensors.tmp")
                save_file(
                    tensors,
                    str(tmp),
                    metadata={"identity": json.dumps(identity), "item_id": item["id"]},
                )
                tmp.replace(dest)
            entries.append(
                {
                    "id": item["id"],
                    "repeats": item.get("repeats", 1),
                    "file": dest.name,
                    "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                    "latent_shape": list(tensors["input_latents"].shape),
                }
            )
            if same_identity:
                reused += 1
            else:
                encoded += 1
            progress.set_postfix(encoded=encoded, reused=reused, refresh=False)
            progress.update(1)
        index = {"identity": identity, "items": entries}
        atomic_json(index_path, index)
        pending.unlink(missing_ok=True)
        return index
    finally:
        progress.close()
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def validate_tensors(tensors, *, channels=64):
    import torch

    required = {"input_latents", "prompt_embeds", "edit_image_pad_mask"}
    if not required <= tensors.keys() or tensors.keys() - required - {"prompt_embeds_mask"}:
        raise ValueError("cache tensor keys 不正確。")
    latent, prompt, mask = (
        tensors[k] for k in ("input_latents", "prompt_embeds", "edit_image_pad_mask")
    )
    if (
        latent.ndim != 4
        or latent.shape[:2] != (1, channels)
        or min(latent.shape[2:]) < 2
        or any(v % 2 for v in latent.shape[2:])
    ):
        raise ValueError("cache latent shape 必須是 (1,64,偶數H,偶數W)。")
    if prompt.ndim != 3 or prompt.shape[0] != 1 or prompt.shape[1] < 1:
        raise ValueError("cache prompt shape 不正確。")
    if mask.shape != prompt.shape[:2] or mask.dtype != torch.bool or mask.any():
        raise ValueError("第一版只接受 T2I cache，不接受 reference-image slots。")
    if "prompt_embeds_mask" in tensors and tensors["prompt_embeds_mask"].shape != mask.shape:
        raise ValueError("cache attention mask shape 不一致。")
    if any(v.is_floating_point() and not torch.isfinite(v).all() for v in tensors.values()):
        raise ValueError("cache 包含 NaN／Inf。")


def load_cache_index(cfg, manifest, assets):
    from safetensors.torch import load_file

    folder = Path(cfg.training["cache_dir"])
    index_file = folder / "cache-index.json"
    if not index_file.exists():
        raise FileNotFoundError("還沒有完成 cache。請依序執行 prepare、cache、train。")
    index = json.loads(index_file.read_text())
    expected = _cache_identity(manifest, source_signature(cfg, assets))
    if index["identity"] != expected:
        raise ValueError("資料／模型／編碼設定與 cache 不一致；請重新 prepare／cache。")
    if [x["id"] for x in index["items"]] != [x["id"] for x in manifest["items"]]:
        raise ValueError("cache 未完整涵蓋 dataset。")
    for entry, item in zip(index["items"], manifest["items"]):
        if type(entry.get("repeats", 1)) is not int or entry.get("repeats", 1) != item.get(
            "repeats", 1
        ):
            raise ValueError("cache repeats 與資料設定不符。")
        if entry["file"] != f"{entry['id']}.safetensors":
            raise ValueError("cache 檔名格式錯誤。")
        p = folder / entry["file"]
        if (
            p.is_symlink()
            or not p.is_file()
            or hashlib.sha256(p.read_bytes()).hexdigest() != entry["sha256"]
        ):
            raise ValueError(f"cache 缺失或損壞：{p}")
        tensors = load_file(str(p))
        validate_tensors(tensors)
        if tensors["prompt_embeds"].shape[-1] != 4096:
            raise ValueError("cache 文字特徵不是 Qwen3-VL-8B 的 4096 維。")
    return index


def make_training_module(pipe, training, *, targets=None, attention="segmented"):
    from diffsynth.diffusion.training_module import DiffusionTrainingModule

    from .block_filter import filter_lora_targets
    from .losses import HaarWaveletLoss, flow_matching_loss
    from .training_options import normalize_training_options

    training = normalize_training_options(training)

    class TrainingModule(DiffusionTrainingModule):
        def __init__(self):
            super().__init__()
            self.pipe = pipe
            self.wavelet_loss = HaarWaveletLoss() if training["loss_type"] == "wavelet" else None
            self.switch_pipe_to_training_mode(
                self.pipe,
                lora_base_model="dit",
                lora_target_modules=targets or "",
                lora_rank=training["rank"],
            )

        def parse_lora_target_modules(self, model, lora_target_modules):
            import re

            import torch

            if lora_target_modules:
                # Internal smoke/test callers use exact names or dotted suffixes.
                requested = lora_target_modules.split(",")
                candidates = [
                    name
                    for name, module in model.named_modules()
                    if isinstance(module, torch.nn.Linear)
                    and any(name == target or name.endswith("." + target) for target in requested)
                ]
            else:
                candidates = self.auto_detect_lora_target_modules(model)
            self.lora_target_filter = filter_lora_targets(candidates, training)
            selected = self.lora_target_filter["selected_targets"]
            print(
                f"LoRA targets: {len(selected)}/{len(candidates)} selected "
                f"(glob; exclude wins); include_blocks={training['include_blocks']!r}, "
                f"exclude_blocks={training['exclude_blocks']!r}",
                flush=True,
            )
            # A PEFT list permits suffix matches; the pinned helper also turns
            # singleton lists into regex strings. Always pass an escaped exact
            # fullmatch regex so neither path can re-enable an excluded target.
            return "(?:" + "|".join(re.escape(name) for name in selected) + ")"

        def add_lora_to_model(
            self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None
        ):
            # The pinned switch_pipe helper does not expose alpha, but calls
            # this virtual method. Inject the real PEFT config without changing
            # vendor code or baking the scaling into A/B parameters.
            result = super().add_lora_to_model(
                model,
                target_modules,
                lora_rank,
                lora_alpha=training["alpha"],
                upcast_dtype=upcast_dtype,
            )
            from peft.tuners.lora.layer import LoraLayer

            actual = sorted(
                name for name, module in result.named_modules() if isinstance(module, LoraLayer)
            )
            if actual != self.lora_target_filter["selected_targets"]:
                raise RuntimeError("PEFT 實際掛載的 LoRA targets 與 include/exclude 篩選結果不一致。")
            return result

        def forward(self, tensors):
            tensors = self.transfer_data_to_device(tensors, self.pipe.device, self.pipe.torch_dtype)
            return self.loss(tensors)

        def loss(self, tensors):
            inputs = dict(tensors)
            inputs.setdefault("prompt_embeds_mask", None)
            inputs.update(
                use_gradient_checkpointing=training["gradient_checkpointing"],
                use_gradient_checkpointing_offload=training["checkpointing_offload"],
                use_flex_attention=attention == "flex",
                kv_cache=None,
            )
            return flow_matching_loss(
                self.pipe,
                inputs,
                loss_type=training["loss_type"],
                weighting=training["loss_weighting"],
                wavelet=self.wavelet_loss,
            )

    return TrainingModule()


def save_lora(model, path, metadata):
    from .lora_io import save_checkpoint

    return save_checkpoint(model, path, metadata)


def optimization_loop(
    model, samples, settings, *, on_step=None, on_optimizer=None, lr_settings=None
):
    """Exactly max_steps optimizer updates; no epoch-derived step override."""
    import torch

    from .optimizers import build_optimizer, optimizer_report
    from .prefetch import sample_stream

    optimizer = build_optimizer(model.trainable_modules(), settings)
    report = optimizer_report(optimizer, settings)
    if on_optimizer is not None:
        on_optimizer(report)
    from .lr_schedule import build_lr_scheduler

    # Smoke tests can exercise the first few updates of a longer configured
    # schedule without extending their actual optimizer-update budget.
    schedule_options = dict(settings)
    if lr_settings is not None:
        schedule_options.update(
            {
                key: lr_settings[key]
                for key in ("lr_scheduler", "num_warmup", "max_steps")
                if key in lr_settings
            }
        )
    scheduler = build_lr_scheduler(optimizer, schedule_options)
    updates = []
    loss_total = 0.0
    accumulation = settings["gradient_accumulation_steps"]
    if len(samples) == 0:
        raise ValueError("沒有可訓練樣本。")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start = time.monotonic()
    with sample_stream(samples, settings) as sample_iterator:
        with tqdm(
            total=settings["max_steps"],
            desc="Train",
            unit="step",
            dynamic_ncols=True,
            mininterval=0.5,
            disable=None,
        ) as progress:
            for step in range(1, settings["max_steps"] + 1):
                total_loss = 0.0
                data_wait = 0.0
                for _ in range(accumulation):
                    wait_start = time.monotonic()
                    sample = next(sample_iterator)
                    data_wait += time.monotonic() - wait_start
                    loss = model(sample)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"第 {step} 步 loss 不是有限值，已停止。")
                    (loss / accumulation).backward()
                    manager = getattr(model, "_offload_manager", None)
                    if manager is not None:
                        manager.after_backward()
                    total_loss += float(loss.detach()) / accumulation
                finite = [
                    torch.isfinite(p.grad).all()
                    for p in model.trainable_modules()
                    if p.grad is not None
                ]
                if finite and not torch.stack(finite).all():
                    raise FloatingPointError(f"第 {step} 步梯度包含 NaN／Inf，已停止。")
                used_lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                loss_total += total_loss
                row = {
                    "step": step,
                    "loss": total_loss,
                    "loss_average": loss_total / step,
                    "learning_rate": used_lr,
                    "elapsed_seconds": time.monotonic() - start,
                    "images_seen": step * accumulation,
                    "data_wait_seconds": data_wait,
                }
                updates.append(row)
                if on_step is not None:
                    on_step(row)
                progress.set_postfix(loss=f"{total_loss:.6f}", lr=f"{used_lr:.2e}", refresh=False)
                progress.update(1)
    return updates


def train(cfg, *, dry_run=False, warm_start=None):
    import torch

    from .data import load_manifest
    from .lr_schedule import lr_scheduler_report

    manifest = load_manifest(cfg)
    assets = model_assets(cfg)
    index = load_cache_index(cfg, manifest, assets)
    out = Path(cfg.training["output_dir"])
    if out.exists() and any(out.iterdir()):
        raise ValueError("output_dir 不是空目錄；請換一個實驗目錄，避免覆蓋訓練結果。")
    plan = {
        "upstream_revision": UPSTREAM_REVISION,
        "config": cfg.to_dict(),
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "model_signature": index["identity"]["model_signature"],
        "num_images": len(index["items"]),
        "weighted_images_per_pass": sum(x.get("repeats", 1) for x in index["items"]),
        "optimizer_updates": cfg.training["max_steps"],
        "effective_batch_size": cfg.training["gradient_accumulation_steps"],
        "cpu_prefetch": {
            "num_workers": cfg.training["num_workers"],
            "prefetch_factor": cfg.training["prefetch_factor"],
            "max_inflight_tasks": cfg.training["num_workers"] * cfg.training["prefetch_factor"],
            "backend": "spawn" if cfg.training["num_workers"] else "synchronous",
            "pin_memory": False,
        },
        "rank": cfg.training["rank"],
        "alpha": cfg.training["alpha"],
        "lora_scale": cfg.training["alpha"] / cfg.training["rank"],
        "latent_mode": "posterior_mean",
        "schedule_mu": 0.8,
        "loss_type": cfg.training["loss_type"],
        "loss_weighting": cfg.training["loss_weighting"],
        "optimizer": cfg.training["optimizer"],
        "lr_schedule": lr_scheduler_report(cfg.training),
        "optimizer_params": cfg.training["optimizer_params"],
        "warm_start": str(Path(warm_start).resolve()) if warm_start else None,
    }
    if warm_start:
        from .lora_io import read_checkpoint

        checkpoint = read_checkpoint(warm_start)
        if checkpoint.rank != cfg.training["rank"] or checkpoint.alpha != cfg.training["alpha"]:
            raise ValueError("warm-start 的 rank/alpha 與設定不符；不會暗中重新縮放權重。")
        plan["warm_start_lora"] = {
            "rank": checkpoint.rank,
            "alpha": checkpoint.alpha,
            "legacy": checkpoint.legacy,
        }
        del checkpoint
    if dry_run:
        return plan
    device = check_device(cfg)
    from .tracking import TrainingTracker

    out.mkdir(parents=True, exist_ok=True)
    # Start tracking before seeding/model initialization so SDK setup cannot
    # consume the training RNG stream. Only the parent process enters this scope.
    with TrainingTracker(cfg, plan, out) as tracker:
        plan["wandb_runtime"] = tracker.info
        torch.manual_seed(cfg.training["seed"])
        random.seed(cfg.training["seed"])
        pipe = build_pipeline(cfg, assets, ["transformer"], training=True)
        model = make_training_module(pipe, cfg.training, attention=cfg.model["attention"])
        target_filter = getattr(model, "lora_target_filter", None)
        if target_filter is not None:
            plan["lora_target_filter"] = target_filter
        if warm_start:
            from .lora_io import load_training_checkpoint

            # Raw A/B go into PEFT: never call the inference alpha converter here.
            plan["warm_start_lora"] = load_training_checkpoint(model, warm_start)
            print("warm start：僅載入 LoRA 權重；optimizer、步數、隨機狀態重新開始。", flush=True)
        if cfg.model["cpu_offload"]:
            if device.type != "cuda":
                raise ValueError("cpu_offload 需要 CUDA，CPU 測試請關閉。")
            from diffsynth.core.offload_training import OffloadTrainingManager

            pipe.device = device
            model._offload_manager = OffloadTrainingManager(model, device)
        else:
            model.to(device=device)
        trainables = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        plan["trainable_parameters"] = sum(p.numel() for _, p in trainables)
        plan["adapter_count"] = sum("lora_A" in n for n, _ in trainables)
        plan["trainable_dtypes"] = sorted({str(p.dtype) for _, p in trainables})
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / "run.json", plan)

        from .prefetch import CachedTensorDataset

        samples = CachedTensorDataset(cfg.training["cache_dir"], index["items"])
        sampler = None
        if cfg.sample["enabled"]:
            from .sampling import TrainingSampler

            sampler = TrainingSampler(cfg, assets, model, out)

        with (out / "metrics.csv").open("x", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "loss",
                    "loss_average",
                    "learning_rate",
                    "elapsed_seconds",
                    "images_seen",
                    "data_wait_seconds",
                ],
            )
            writer.writeheader()

            def on_step(row):
                writer.writerow(row)
                f.flush()
                if row["step"] % cfg.training["save_every"] == 0:
                    save_lora(
                        model,
                        out / f"step-{row['step']:06d}.safetensors",
                        {
                            "upstream_revision": UPSTREAM_REVISION,
                            "rank": cfg.training["rank"],
                            "alpha": cfg.training["alpha"],
                            "step": row["step"],
                            "lr_scheduler": cfg.training["lr_scheduler"],
                            "num_warmup": cfg.training["num_warmup"],
                        },
                    )
                # Sampling is opt-in and happens after a completed optimizer
                # update/checkpoint. One W&B commit holds this step's metrics
                # and images, avoiding late same-step media being discarded.
                previews = (
                    sampler.generate(row["step"])
                    if sampler is not None and sampler.should_sample(row["step"])
                    else []
                )
                if previews:
                    tracker.log(row, samples=previews)
                else:
                    tracker.log(row)

            def on_optimizer(report):
                plan["optimizer_runtime"] = report
                atomic_json(out / "run.json", plan)

            updates = optimization_loop(
                model, samples, cfg.training, on_step=on_step, on_optimizer=on_optimizer
            )
        final_path = out / "final.safetensors"
        save_lora(
            model,
            final_path,
            {
                "upstream_revision": UPSTREAM_REVISION,
                "rank": cfg.training["rank"],
                "alpha": cfg.training["alpha"],
                "step": len(updates),
                "lr_scheduler": cfg.training["lr_scheduler"],
                "num_warmup": cfg.training["num_warmup"],
            },
        )
        summary = {
            "optimizer_updates": len(updates),
            "final_loss": updates[-1]["loss"],
            "checkpoint": str(final_path),
            "sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
        }
        atomic_json(out / "completed.json", summary)
        return summary


def sample(cfg, *, lora, output, prompt=None):
    output = Path(output)
    if output.exists():
        raise ValueError(f"不覆蓋既有圖片：{output}")
    if output.suffix.lower() != ".png":
        raise ValueError("RGBA sample 請使用 .png 輸出。")
    checkpoint = None
    if lora:
        from .lora_io import read_checkpoint

        checkpoint = read_checkpoint(lora)
    pipe = build_pipeline(cfg, model_assets(cfg), ["transformer", "text_encoder", "vae"])
    if checkpoint is not None:
        import warnings

        # Apply the file's alpha/r exactly once, before the generic loader can
        # cast an alpha scalar to the pipeline's BF16 dtype. Subsequent loader
        # passes see A/B only; strength is independent and explicitly one.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Alpha detected in the LoRA file.*", category=UserWarning
            )
            state = pipe.lora_loader().convert_state_dict(checkpoint.inference_state)
        modules = dict(pipe.dit.named_modules())
        if not state:
            raise ValueError("LoRA 沒有可載入的 A/B tensors。")
        for key, value in state.items():
            target = key.rsplit(".lora_", 1)[0]
            if target not in modules:
                raise ValueError(f"LoRA target 不相容：{target}；fused gate_up 不能直接載入。")
            module = modules[target]
            expected = module.in_features if ".lora_A." in key else module.out_features
            axis = 1 if ".lora_A." in key else 0
            if value.ndim != 2 or value.shape[axis] != expected:
                raise ValueError(f"LoRA tensor shape 不相容：{key}")
        pipe.load_lora(pipe.dit, state_dict=state, alpha=1.0)
    settings = cfg.sample
    image = pipe(
        prompt=prompt or settings["prompt"],
        negative_prompt=settings["negative_prompt"],
        height=settings["height"],
        width=settings["width"],
        seed=settings["seed"],
        num_inference_steps=settings["steps"],
        cfg_scale=settings["cfg_scale"],
        use_flex_attention=cfg.model["attention"] == "flex",
        tiled=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return {"image": str(output.resolve())}
