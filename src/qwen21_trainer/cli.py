"""CLI: lightweight validation first; model work is always explicit."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description="DiffSynth Qwen-Image 2.1 小資料集 LoRA 訓練器")
    sub = p.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("prepare", "檢查圖片與 caption，建立資料 manifest"),
        ("cache", "使用官方 VAE／TE 預先編碼，不載入 DiT"),
        ("train", "使用 cache 訓練 LoRA；啟用 sample 時另載 TE/VAE"),
        ("sample", "用 DiffSynth pipeline 產生驗證圖片"),
        ("doctor", "離線檢查依賴、設定、模型及 cache"),
        ("download", "明確下載官方 Qwen-Image-2.1 模型"),
    ]:
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument(
            "--config",
            default="configs/small.toml",
            help="TOML/YAML 路徑；相對路徑以設定檔所在目錄為基準",
        )
        if name in ("prepare", "cache"):
            cmd.add_argument(
                "--overwrite",
                action="store_true",
                help="允許重建衍生的 manifest／cache，不修改原圖",
            )
        if name == "train":
            cmd.add_argument(
                "--dry-run", action="store_true", help="驗證資料／模型／cache，不載入 GPU 或訓練"
            )
            cmd.add_argument(
                "--lora", help="DiffSynth 同 rank/targets LoRA warm start；不是完整狀態 resume"
            )
            for key, description in (
                ("include_blocks", "僅訓練符合 glob 的 LoRA layers"),
                ("exclude_blocks", "排除符合 glob 的 LoRA layers（排除優先）"),
            ):
                cmd.add_argument(
                    f"--{key}",
                    f"--{key.replace('_', '-')}",
                    nargs="+",
                    metavar="PATTERN",
                    help=f"{description}；覆寫設定檔，可傳多個加引號的 pattern 或陣列字串；[] 清空",
                )
        if name == "sample":
            cmd.add_argument("--lora", help="DiffSynth LoRA；省略時使用 base model")
            cmd.add_argument("--output", required=True, help="輸出 PNG 路徑；拒絕覆蓋")
            cmd.add_argument("--prompt", help="覆寫設定檔的 sample prompt")
        if name == "download":
            cmd.add_argument(
                "--confirm-large-download",
                action="store_true",
                help="允許下載完整官方模型（數十 GB）",
            )
            cmd.add_argument(
                "--revision", default="main", help="Hugging Face 權重 revision，可指定 commit"
            )
    smoke = sub.add_parser("smoke-test", help="CPU 離線 tiny 真實 DiT 訓練／儲存／重載，不下載權重")
    smoke.add_argument("--output-dir", default="verification/smoke", help="必須是空目錄或新目錄")
    smoke.add_argument(
        "--config", help="取 rank/alpha/loss/optimizer/LR 排程前綴；CPU tiny model 固定 3 updates"
    )
    sub.add_parser("verify-upstream", help="比對已安裝 DiffSynth 與 bundled source 的 SHA-256")
    return p


def verify_upstream():
    import hashlib
    import importlib.util

    manifest = json.loads((Path(__file__).parent / "upstream-manifest.json").read_text())
    spec = importlib.util.find_spec("diffsynth")
    if spec is None or not spec.origin:
        raise RuntimeError("找不到 DiffSynth，請先執行 uv sync --locked。")
    folder = Path(spec.origin).parent
    checked = 0
    for name, digest in manifest["files"].items():
        if name.startswith("diffsynth/") and name.endswith(".py"):
            file = folder / name.removeprefix("diffsynth/")
            if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"DiffSynth 原始碼不符合固定版本：{file}；請重新 uv sync。")
            checked += 1
    return {
        "revision": manifest["revision"],
        "verified_python_files": checked,
        "installed_path": str(folder),
    }


def doctor(cfg):
    from .data import load_manifest
    from .runtime import load_cache_index, model_assets

    report = {
        "config": cfg.to_dict(),
        "upstream": verify_upstream(),
        "checks": {},
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "peft",
                "accelerate",
                "diffsynth",
                "adv_optm",
                "pytorch-wavelets",
                "PyWavelets",
            )
        },
    }
    import torch

    report["cuda_available"] = torch.cuda.is_available()
    errors = []
    assets = None
    manifest = None
    for name, action in [
        ("dataset", lambda: load_manifest(cfg)),
        ("models", lambda: model_assets(cfg)),
    ]:
        try:
            value = action()
            if name == "dataset":
                manifest = value
            else:
                assets = value
            report["checks"][name] = "ok"
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            report["checks"][name] = str(exc)
            errors.append(name)
    if assets and manifest:
        try:
            index = load_cache_index(cfg, manifest, assets)
            report["checks"]["cache"] = f"ok: {len(index['items'])} images"
        except (ValueError, FileNotFoundError) as exc:
            report["checks"]["cache"] = str(exc)
            errors.append("cache")
    if cfg.model["device"] == "cuda" and not report["cuda_available"]:
        errors.append("cuda")
    if cfg.wandb["enabled"]:
        if importlib.util.find_spec("wandb") is None:
            report["checks"]["wandb"] = "缺少 wandb SDK，請同步專案依賴。"
            errors.append("wandb")
        elif cfg.wandb["mode"] == "online" and not os.environ.get("WANDB_API_KEY"):
            report["checks"]["wandb"] = "環境中缺少 WANDB_API_KEY（不會互動登入）。"
            errors.append("wandb")
        else:
            report["checks"]["wandb"] = (
                "configured; credentials/network not validated"
                if cfg.wandb["mode"] == "online"
                else "offline"
            )
    report["ready_to_train"] = not errors
    return report


def download(cfg, args):
    if not args.confirm_large_download:
        raise ValueError(
            "完整權重為數十 GB。請加 --confirm-large-download 明確允許下載；此命令不會啟動訓練。"
        )
    if cfg.model["format"] != "official":
        raise ValueError(
            "comfy_bf16 使用既有本機權重，不需要 download；下載官方格式請使用另一份 official 設定。"
        )
    from huggingface_hub import HfApi, snapshot_download

    from .runtime import atomic_json

    repo = "Qwen/Qwen-Image-2.1"
    info = HfApi().model_info(repo, revision=args.revision)
    # Resolve mutable revision to an exact commit before fetching any weights.
    dest = Path(cfg.model["root"])
    snapshot_download(
        repo_id=repo,
        revision=info.sha,
        local_dir=str(dest),
        allow_patterns=[
            "transformer/*",
            "text_encoder/*",
            "vae/*",
            "processor/*",
            "scheduler/*",
            "model_index.json",
        ],
    )
    record = {"repository": repo, "revision": info.sha, "local_dir": str(dest)}
    atomic_json(dest / "download-provenance.json", record)
    return record


def _apply_block_filter_overrides(cfg, args):
    from copy import deepcopy
    from dataclasses import replace

    from .block_filter import parse_cli_block_patterns

    overrides = {
        key: parse_cli_block_patterns(getattr(args, key), f"--{key}")
        for key in ("include_blocks", "exclude_blocks")
        if getattr(args, key, None) is not None
    }
    if not overrides:
        return cfg
    result = replace(cfg, training={**cfg.training, **overrides})
    # Keep file provenance distinct from effective CLI-overridden settings.
    object.__setattr__(result, "source_document", deepcopy(cfg.source_document))
    object.__setattr__(result, "source_format", cfg.source_format)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "smoke-test":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["HF_HUB_OFFLINE"] = "1"
            verify_upstream()
            from .smoke import run_smoke

            options = None
            if args.config:
                from .config import load_config

                options = load_config(args.config).training
            result = run_smoke(args.output_dir, training_options=options)
        elif args.command == "verify-upstream":
            result = verify_upstream()
        else:
            from .config import load_config

            cfg = load_config(args.config)
            if args.command == "train":
                cfg = _apply_block_filter_overrides(cfg, args)
            if args.command == "prepare":
                from .data import prepare_dataset

                result = prepare_dataset(cfg, overwrite=args.overwrite)
                result = {
                    "num_images": len(result["items"]),
                    "dataset_fingerprint": result["dataset_fingerprint"],
                    "manifest": str(Path(cfg.training["cache_dir"]) / "dataset.json"),
                }
            elif args.command == "doctor":
                result = doctor(cfg)
            elif args.command == "download":
                result = download(cfg, args)
            else:
                verify_upstream()
                from . import runtime

                if args.command == "cache":
                    index = runtime.cache_dataset(cfg, overwrite=args.overwrite)
                    result = {
                        "cached_images": len(index["items"]),
                        "cache_dir": str(cfg.training["cache_dir"]),
                    }
                elif args.command == "train":
                    result = runtime.train(cfg, dry_run=args.dry_run, warm_start=args.lora)
                else:
                    result = runtime.sample(
                        cfg, lora=args.lora, output=args.output, prompt=args.prompt
                    )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if args.command == "doctor" and not result["ready_to_train"] else 0
    except (ValueError, FileNotFoundError, RuntimeError, FloatingPointError, OSError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
