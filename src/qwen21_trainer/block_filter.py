"""Torch-free glob filtering over original dotted LoRA target module paths."""

from __future__ import annotations

import ast
import warnings
from fnmatch import fnmatchcase


def validate_block_patterns(value, label):
    """Require a real list; never interpret config strings as executable code."""
    if not isinstance(value, list) or any(
        not isinstance(pattern, str)
        or not pattern.strip()
        or any(char in pattern for char in "\x00\r\n")
        for pattern in value
    ):
        raise ValueError(f"{label} 必須是非空白字串的陣列（可用 [] 表示不限制）。")
    return list(value)


def parse_cli_block_patterns(values, label):
    """Accept quoted CLI globs or one quoted Python/JSON-style list literal."""
    if len(values) == 1 and values[0].lstrip().startswith("["):
        try:
            values = ast.literal_eval(values[0])
        except (ValueError, SyntaxError) as exc:
            # A raw glob may start with a character class, e.g. '[tT]*'.
            # Quoted/comma-containing list syntax must still fail strictly.
            bracket_end = values[0].find("]")
            if bracket_end > 1 and not any(char in values[0] for char in "'\","):
                return validate_block_patterns(values, label)
            raise ValueError(f"{label} 陣列格式錯誤；例如 \"['*mlp*', '*.attn.to_q']\"。") from exc
    return validate_block_patterns(values, label)


def filter_lora_targets(candidates, training):
    """Select (default candidates intersect include) minus exclude; exclude wins.

    Empty include/exclude lists impose no restriction. Matching is case-sensitive
    fnmatch (glob), NOT regex, against the entire original dotted module path.
    '*' crosses dots; whole-block selection needs e.g. 'transformer_blocks.0.*'.
    Filtering never expands the trainer's existing eligible Linear target set.
    """
    candidates = sorted(set(candidates))
    include = validate_block_patterns(training.get("include_blocks", []), "training.include_blocks")
    exclude = validate_block_patterns(training.get("exclude_blocks", []), "training.exclude_blocks")
    if not candidates:
        raise ValueError("找不到可掛載 LoRA 的 Linear targets；請檢查模型與 target 設定。")
    unmatched = {
        key: [
            pattern for pattern in patterns if not any(fnmatchcase(n, pattern) for n in candidates)
        ]
        for key, patterns in (("include_blocks", include), ("exclude_blocks", exclude))
    }
    selected = [
        name
        for name in candidates
        if (not include or any(fnmatchcase(name, pattern) for pattern in include))
        and not any(fnmatchcase(name, pattern) for pattern in exclude)
    ]
    if not selected:
        raise ValueError(
            "include_blocks/exclude_blocks 篩選後沒有可訓練的 LoRA targets；"
            f"include_blocks={include!r}，exclude_blocks={exclude!r}。"
            "使用 glob 比對完整 module path（不是 regex／parameter key）；"
            f"可用名稱例如：{', '.join(candidates[:7])}"
        )
    for key, patterns in unmatched.items():
        if patterns:
            warnings.warn(
                f"training.{key} 未命中任何候選 LoRA target：{patterns!r}；"
                "請使用原始點分隔 module path，例如 *.attn.to_q。",
                UserWarning,
                stacklevel=2,
            )
    return {
        "match_mode": "glob",
        "precedence": "exclude_over_include",
        "include_blocks": include,
        "exclude_blocks": exclude,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "selected_targets": selected,
        "unmatched_patterns": unmatched,
    }
