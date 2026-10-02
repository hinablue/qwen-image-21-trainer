"""Import dataset paths/repeats/resolution from ai-toolkit, not its optimizer recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--host-datasets-root", required=True)
    p.add_argument("--container-datasets-root", default="/app/ai-toolkit/datasets")
    p.add_argument("--steps", type=int, default=300)
    a = p.parse_args()
    source, output = Path(a.source).resolve(), Path(a.output).resolve()
    if output.exists() or output.with_suffix(".import.json").exists():
        raise ValueError("不覆蓋既有設定或 import provenance。")
    raw = source.read_bytes()
    process = yaml.safe_load(raw)["config"]["process"][0]
    datasets = process["datasets"]
    if not datasets or a.steps <= 0:
        raise ValueError("datasets 或 steps 不合法。")

    def quote(value):
        return json.dumps(str(value), ensure_ascii=False)

    lines = [
        "# 由 ai-toolkit config 匯入資料設定；不複製其 optimizer/loss/timestep。",
        "# 全資料短測試：max_steps 是 optimizer updates，並非讀完全部資料。",
        "[model]",
        'root = "../models/Qwen-Image-2.1"',
        'device = "cuda"',
        'attention = "segmented"',
        "cpu_offload = false",
        "",
        "[dataset]",
        "max_pixels = 1048576",
        'caption_extension = ".txt"',
        "recursive = true",
        "",
    ]
    mappings = []
    for ds in datasets:
        for unsupported in (
            "mask_path",
            "control_path",
            "control_path_1",
            "control_path_2",
            "control_path_3",
        ):
            if ds.get(unsupported):
                raise ValueError(f"不支援 dataset {unsupported}，不可默默忽略。")
        if (
            ds.get("is_reg")
            or ds.get("controls")
            or ds.get("flip_x")
            or ds.get("flip_y")
            or ds.get("network_weight", 1) != 1
        ):
            raise ValueError("此匯入器不支援 regularization/control/flip/非1 network_weight。")
        relative = Path(ds["folder_path"]).relative_to(a.container_datasets_root)
        host = (Path(a.host_datasets_root).resolve() / relative).resolve()
        if not host.is_relative_to(Path(a.host_datasets_root).resolve()) or not host.is_dir():
            raise ValueError(f"資料集 host 路徑不存在或逃逸：{host}")
        resolutions = ds["resolution"]
        if (
            not isinstance(resolutions, list)
            or len(resolutions) != 1
            or type(resolutions[0]) is not int
        ):
            raise ValueError("第一版匯入只支援每組單一 resolution。")
        repeats = ds.get("num_repeats", 1)
        if type(repeats) is not int or repeats < 1:
            raise ValueError("num_repeats 必須為正整數。")
        extension = ds.get("caption_ext", "txt")
        extension = extension if extension.startswith(".") else "." + extension
        lines += [
            "[[datasets]]",
            f"path = {quote(host)}",
            f"repeats = {repeats}",
            f"max_pixels = {resolutions[0] ** 2}",
            f"caption_extension = {quote(extension)}",
            "recursive = true",
            "",
        ]
        mappings.append(
            {
                "source_path": ds["folder_path"],
                "host_path": str(host),
                "repeats": repeats,
                "max_pixels": resolutions[0] ** 2,
                "source_caption_dropout_rate": ds.get("caption_dropout_rate", 0),
                "target_caption_dropout_rate": 0,
            }
        )
    network, train, sample = (process[name] for name in ("network", "train", "sample"))
    lines += [
        "[training]",
        f"output_dir = {quote('../output/' + output.stem)}",
        f"cache_dir = {quote('../cache/' + output.stem)}",
        f"seed = {sample.get('seed', 42)}",
        f"rank = {network['linear']}",
        f"learning_rate = {train['lr']}",
        "weight_decay = 0.01",
        f"max_steps = {a.steps}",
        "gradient_accumulation_steps = 1",
        "save_every = 100",
        "gradient_checkpointing = true",
        "checkpointing_offload = false",
        "",
        "[sample]",
        f"prompt = {quote(sample['samples'][0]['prompt'])}",
        f"negative_prompt = {quote(sample.get('neg', ''))}",
        f"width = {sample['width']}",
        f"height = {sample['height']}",
        f"steps = {sample.get('sample_steps', 30)}",
        f"cfg_scale = {float(sample.get('guidance_scale', 3))}",
        f"seed = {sample.get('seed', 42)}",
        "",
    ]
    text = "\n".join(lines)
    # Parse and validate before publishing the imported config.
    import tomllib

    from qwen21_trainer.config import Config

    cfg = Config(path=output, **tomllib.loads(text))
    provenance = {
        "source": str(source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "output": str(output),
        "datasets": mappings,
        "source_recipe": {
            "optimizer": train.get("optimizer"),
            "loss": train.get("loss_type"),
            "timestep": train.get("timestep_type"),
            "steps": train.get("steps"),
        },
        "target_recipe": {
            "optimizer": "AdamW",
            "loss": "FlowMatchSFTLoss + bell weighting",
            "schedule": "fixed mu=0.8 shifted",
            "steps": a.steps,
            "alpha": cfg.training["rank"],
            "rgba": True,
            "caption_dropout_rate": 0,
            "ema": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    output.with_suffix(".import.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"config": str(output), "datasets": len(mappings), "steps": a.steps},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        raise SystemExit(2)
