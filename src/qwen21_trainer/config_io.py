"""Bounded, safe config parsing; no secret-bearing snippets in parser errors."""

import tomllib
from pathlib import Path

MAX_BYTES = 1024 * 1024


def _validate_tree(value, active=None, budget=None, depth=0):
    active = set() if active is None else active
    budget = [10000] if budget is None else budget
    budget[0] -= 1
    if budget[0] < 0 or depth > 64:
        raise ValueError("設定檔的結構過大或過深。")
    if isinstance(value, (dict, list)):
        if id(value) in active:
            raise ValueError("設定檔不允許循環 alias。")
        active.add(id(value))
        try:
            if isinstance(value, dict):
                if any(type(key) is not str for key in value):
                    raise ValueError("設定欄位名稱必須是字串。")
                children = value.values()
            else:
                children = value
            for item in children:
                _validate_tree(item, active, budget, depth + 1)
        finally:
            active.remove(id(value))
    elif type(value) not in (str, int, float, bool, type(None)):
        raise ValueError("設定只允許字串、數值、布林值、陣列及表格。")


def read_config_document(path: Path):
    with path.open("rb") as f:
        content = f.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise ValueError("設定檔上限為 1 MiB。")
    fmt = "yaml" if path.suffix.lower() in {".yaml", ".yml"} else "toml"
    if fmt == "yaml":
        import yaml
    try:
        text = content.decode("utf-8")
        if fmt == "yaml":

            class UniqueSafeLoader(yaml.SafeLoader):
                def construct_mapping(self, node, deep=False):
                    seen = set()
                    for key_node, _ in node.value:
                        key = self.construct_object(key_node, deep=deep)
                        if not isinstance(key, str) or key in seen:
                            raise ValueError("YAML 不允許重複或非字串欄位名稱。")
                        seen.add(key)
                    return super().construct_mapping(node, deep=deep)

            raw = yaml.load(text, Loader=UniqueSafeLoader)
        else:
            raw = tomllib.loads(text)
    except (UnicodeError, ValueError, RecursionError):
        raise ValueError(f"無法解析 {fmt.upper()} 設定檔；請檢查語法與欄位。") from None
    except Exception as exc:
        # PyYAML errors may quote raw credentials from the input. Do not include
        # their exception text in a CLI traceback or remote diagnostic log.
        if fmt == "yaml" and isinstance(exc, yaml.YAMLError):
            raise ValueError("無法解析 YAML 設定檔；不支援不安全的標籤或格式。") from None
        raise
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("設定檔頂層必須是表格／mapping。")
    _validate_tree(raw)
    return raw, fmt
