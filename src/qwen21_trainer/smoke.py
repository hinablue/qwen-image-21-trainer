"""Offline smoke: real tiny upstream DiT, PEFT, loss, updates and LoRA reload."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path


def run_smoke(output_dir, *, training_options=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from diffsynth.models.qwen_image_21_dit import QwenImage21DiT
    from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline

    from .lora_io import load_training_checkpoint, read_checkpoint
    from .lr_schedule import lr_scheduler_report
    from .runtime import atomic_json, make_training_module, optimization_loop, save_lora
    from .training_options import normalize_training_options

    torch.set_num_threads(1)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("smoke-test 輸出目錄非空，請使用新目錄。")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    params = dict(
        in_channels=4,
        out_channels=4,
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=32,
        context_in_dim=64,
        axes_dims_rope=(8, 12, 12),
    )
    settings = dict(
        rank=2,
        learning_rate=1e-3,
        weight_decay=0.01,
        max_steps=3,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        checkpointing_offload=False,
        seed=42,
    )
    if training_options:
        for key in (
            "rank",
            "alpha",
            "include_blocks",
            "exclude_blocks",
            "learning_rate",
            "weight_decay",
            "loss_type",
            "loss_weighting",
            "optimizer",
            "optimizer_params",
            "num_workers",
            "prefetch_factor",
        ):
            if key in training_options:
                settings[key] = training_options[key]
    settings = normalize_training_options(settings)
    lr_options = {
        key: training_options[key]
        for key in ("lr_scheduler", "num_warmup", "max_steps")
        if training_options and key in training_options
    }
    schedule_report = lr_scheduler_report({**settings, **lr_options})
    targets = "to_q,to_k,to_v,to_out.0,gate_layer,proj,out"
    reports = []
    for dtype in (torch.float32, torch.bfloat16):
        pipe = QwenImage21Pipeline(device="cpu", torch_dtype=dtype)
        pipe.dit = QwenImage21DiT(**params).to(dtype=dtype)
        pristine = copy.deepcopy(pipe.dit.state_dict())
        model = make_training_module(pipe, settings, targets=targets)
        samples = [
            {
                "input_latents": torch.randn(1, 4, 4, 4, dtype=dtype),
                "prompt_embeds": torch.randn(1, 5, 64, dtype=dtype),
                "edit_image_pad_mask": torch.zeros(1, 5, dtype=torch.bool),
            }
            for _ in range(2)
        ]
        before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        optimizer_reports = []
        updates = optimization_loop(
            model, samples, settings, on_optimizer=optimizer_reports.append, lr_settings=lr_options
        )
        changed = sum(
            not torch.equal(before[n], p) for n, p in model.named_parameters() if n in before
        )
        assert changed > 0, "LoRA parameters did not update"
        assert len(updates) == 3 and updates[-1]["images_seen"] == 6
        checkpoint = output / f"lora-{str(dtype).split('.')[-1]}.safetensors"
        save_lora(
            model,
            checkpoint,
            {"test_only": "true", "rank": settings["rank"], "alpha": settings["alpha"]},
        )
        reloaded = QwenImage21Pipeline(device="cpu", torch_dtype=dtype)
        reloaded.dit = QwenImage21DiT(**params).to(dtype=dtype)
        reloaded.dit.load_state_dict(pristine)
        model2 = make_training_module(reloaded, settings, targets=targets)
        saved = read_checkpoint(checkpoint)
        state = saved.training_state
        reload_report = load_training_checkpoint(model2, checkpoint)
        model.eval()
        model2.eval()
        kwargs = dict(
            latents=samples[0]["input_latents"],
            timestep=torch.tensor([500.0], dtype=dtype),
            prompt_embeds=samples[0]["prompt_embeds"],
            prompt_embeds_mask=None,
            edit_image_pad_mask=samples[0]["edit_image_pad_mask"],
            use_flex_attention=False,
        )
        with torch.no_grad():
            original = pipe.model_fn(dit=pipe.dit, **kwargs)
            recovered = reloaded.model_fn(dit=reloaded.dit, **kwargs)
        error = float((original.float() - recovered.float()).abs().max())
        assert torch.equal(original, recovered), f"reload differs: {error}"
        reports.append(
            {
                "dtype": str(dtype),
                "rank": settings["rank"],
                "alpha": settings["alpha"],
                "lora_scale": settings["alpha"] / settings["rank"],
                "lora_target_filter": model.lora_target_filter,
                "reload_report": reload_report,
                "lr_schedule": schedule_report,
                "learning_rates": [r["learning_rate"] for r in updates],
                "saved_alpha_tensors": sum(key.endswith(".alpha") for key in saved.inference_state),
                "loss_type": settings.get("loss_type", "mse"),
                "loss_weighting": settings.get("loss_weighting", "diffsynth"),
                "optimizer": optimizer_reports[0],
                "num_workers": settings.get("num_workers", 0),
                "prefetch_factor": settings.get("prefetch_factor", 2),
                "data_wait_seconds": [r["data_wait_seconds"] for r in updates],
                "optimizer_updates": len(updates),
                "microsteps": updates[-1]["images_seen"],
                "losses": [r["loss"] for r in updates],
                "changed_lora_tensors": changed,
                "saved_lora_tensors": len(state),
                "reload_max_abs_error": error,
                "checkpoint": str(checkpoint.resolve()),
            }
        )
    # Prove production automatic target detection without allocating full weights.
    from diffsynth.diffusion.training_module import DiffusionTrainingModule

    with torch.device("meta"):
        production_shape_model = QwenImage21DiT()
    detector = DiffusionTrainingModule()
    detected = detector.auto_detect_lora_target_modules(production_shape_model)
    assert len(detected) == 224, len(detected)
    summary = {
        "status": "passed",
        "cuda_visible": torch.cuda.is_available(),
        "scope": "random tiny real upstream DiT + PEFT + selected Flow Matching objective; not pretrained training or image quality",
        "production_auto_targets": len(detected),
        "runs": reports,
    }
    atomic_json(output / "smoke-report.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output_dir), ensure_ascii=False, indent=2))
