"""Read-only source validation and atomic, content-addressed dataset manifests.

The source images/captions are never rewritten. The manifest describes original
image dimensions; resizing/RGBA conversion belongs to the downstream cache
stage. Fingerprints cover the canonical manifest payload (including absolute
paths, dataset settings, preprocessing, prompts and both source-byte hashes).
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
import tempfile
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from .config import Config

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
_MANIFEST_NAME = "dataset.json"


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("manifest 必須是有效 JSON，且不可包含非有限數值。") from exc


def _reject_symlink_chain(path: Path) -> None:
    for component in (*reversed(path.parents), path):
        if component.is_symlink():
            raise ValueError(f"快取路徑不可包含符號連結：{component}")


def _checked_config(cfg: Config) -> Config:
    if not isinstance(cfg, Config):
        raise ValueError("必須使用 load_config() 取得 Config 設定。")
    # Revalidate mutable section dictionaries and filesystem boundaries on use.
    checked = Config(
        path=cfg.path,
        model=cfg.model,
        dataset=cfg.dataset,
        training=cfg.training,
        sample=cfg.sample,
        wandb=cfg.wandb,
        datasets=cfg.datasets,
    )
    _reject_symlink_chain(Path(cfg.training["cache_dir"]))
    return checked


def _inside(path: Path, root: Path) -> None:
    try:
        relative = path.relative_to(root)
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"資料集內不可使用符號連結：{current}")
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"資料集路徑不可逃逸根目錄：{path}")
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"無法驗證資料集路徑：{path}") from exc
    except ValueError as exc:
        raise ValueError(f"不安全的資料集路徑 {path}：{exc}") from exc


def _read_regular(path: Path, label: str) -> bytes:
    if path.is_symlink():
        raise ValueError(f"{label} 不可使用符號連結：{path}")
    try:
        # Do not follow a final-component symlink swapped in after validation;
        # nonblocking open also prevents a replaced FIFO from hanging a caller.
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError(f"{label} 必須是一般檔案：{path}")
            return stream.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"找不到{label}：{path}") from exc
    except OSError as exc:
        raise ValueError(f"無法讀取{label} {path}：{exc}") from exc


def _images(root: Path, recursive: bool) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"找不到資料集目錄：{root}")
    if not root.is_dir():
        raise ValueError(f"資料集路徑必須是目錄：{root}")
    result: list[Path] = []
    directories = [root]
    try:
        while directories:
            directory = directories.pop()
            _inside(directory, root)
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
            for entry in entries:
                path = Path(entry.path)
                _inside(path, root)
                if entry.is_symlink():
                    raise ValueError(f"資料集內不可使用符號連結：{path}")
                if entry.is_dir(follow_symlinks=False):
                    if recursive:
                        directories.append(path)
                elif path.suffix.lower() in _IMAGE_EXTENSIONS:
                    if not entry.is_file(follow_symlinks=False):
                        raise ValueError(f"圖片必須是一般檔案：{path}")
                    result.append(path)
    except OSError as exc:
        raise ValueError(f"無法掃描資料集 {root}：{exc}") from exc
    if not result:
        raise ValueError(f"資料集沒有支援的圖片（jpg/jpeg/png/webp/bmp）：{root}")
    return sorted(result, key=lambda path: path.relative_to(root).as_posix())


def _dimensions(content: bytes, path: Path) -> tuple[int, int]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                image.verify()
            # verify() checks container integrity; load() additionally decodes
            # pixels and rejects truncated images that pass the first check.
            with Image.open(BytesIO(content)) as image:
                image.load()
                width, height = image.size
        if width < 1 or height < 1:
            raise ValueError("圖片尺寸必須大於零")
        return width, height
    except (
        OSError,
        ValueError,
        SyntaxError,
        EOFError,
        OverflowError,
        struct.error,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError(f"圖片損壞或無法安全解碼：{path}（{exc}）") from exc


def _build_manifest(cfg: Config) -> dict[str, Any]:
    if len(cfg.datasets) == 1 and cfg.datasets[0] == {**cfg.dataset, "repeats": 1}:
        return _build_single_manifest(cfg)
    from dataclasses import replace

    items, groups, seen = [], [], set()
    for index, source in enumerate(cfg.datasets):
        spec = {k: v for k, v in source.items() if k != "repeats"}
        part = _build_single_manifest(replace(cfg, dataset=spec, datasets=[]))
        groups.append({**source, "count": len(part["items"])})
        for item in part["items"]:
            if item["image"] in seen:
                raise ValueError(f"同一張圖片被多個 datasets 重複列入：{item['image']}")
            seen.add(item["image"])
            item["id"] = hashlib.sha256(f"{index}:{item['image']}".encode()).hexdigest()[:20]
            item.update(
                dataset_index=index, repeats=source["repeats"], max_pixels=source["max_pixels"]
            )
            items.append(item)
    payload = {
        "schema_version": 1,
        "datasets": groups,
        "preprocessing": {"rgba": True, "division_factor": 32},
        "items": items,
    }
    return {
        **payload,
        "dataset_fingerprint": hashlib.sha256(_canonical(payload).encode()).hexdigest(),
    }


def _build_single_manifest(cfg: Config) -> dict[str, Any]:
    root = Path(cfg.dataset["path"])
    paths = _images(root, cfg.dataset["recursive"])
    extension = cfg.dataset["caption_extension"]
    captions: dict[Path, Path] = {}
    for path in paths:
        caption_path = path.with_suffix(extension)
        if caption_path in captions:
            raise ValueError(
                f"圖片 stem 重複、共用同一份 caption：{captions[caption_path]} 與 {path}"
            )
        captions[caption_path] = path

    items = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        caption_path = path.with_suffix(extension)
        _inside(path, root)
        _inside(caption_path, root)
        image_bytes = _read_regular(path, "圖片")
        caption_bytes = _read_regular(caption_path, "caption 檔案")
        try:
            prompt = caption_bytes.decode("utf-8-sig").strip()
        except UnicodeError as exc:
            raise ValueError(f"caption 必須是 UTF-8 文字：{caption_path}") from exc
        if not prompt:
            raise ValueError(f"caption 不可為空白：{caption_path}")
        width, height = _dimensions(image_bytes, path)
        items.append(
            {
                "id": hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20],
                "image": str(path),
                "caption_file": str(caption_path),
                "prompt": prompt,
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
                "caption_sha256": hashlib.sha256(caption_bytes).hexdigest(),
                "width": width,
                "height": height,
            }
        )
    payload = {
        "schema_version": 1,
        "dataset": {
            "path": str(root),
            "caption_extension": extension,
            "recursive": cfg.dataset["recursive"],
        },
        "preprocessing": {
            "max_pixels": cfg.dataset["max_pixels"],
            "rgba": True,
            "division_factor": 32,
        },
        "items": items,
    }
    return {
        **payload,
        "dataset_fingerprint": hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest(),
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"manifest JSON 欄位重複：{key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"manifest 不可包含非有限數值：{value}")


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(
            _read_regular(path, "dataset manifest").decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
        if not isinstance(result, dict):
            raise ValueError("manifest 最外層必須是 JSON 物件。")
        _canonical(result)  # Also rejects overflowed JSON numbers, e.g. 1e9999.
        return result
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"manifest 不是有效的 UTF-8 JSON：{path}（{exc}）") from exc


def _atomic_write(path: Path, manifest: dict[str, Any], *, overwrite: bool) -> None:
    temporary: Path | None = None
    try:
        _reject_symlink_chain(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink_chain(path)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".dataset-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                manifest, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Atomic no-clobber publication: an intervening creator must never
            # be silently overwritten when overwrite=False.
            os.link(temporary, path)
        temporary.unlink(missing_ok=True)
        temporary = None
    except FileExistsError as exc:
        raise ValueError(f"manifest 已存在，未覆寫；請重新驗證或指定 --overwrite：{path}") from exc
    except OSError as exc:
        raise ValueError(f"無法原子寫入 dataset manifest {path}：{exc}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                raise ValueError(f"無法清除 manifest 暫存檔：{temporary}") from exc


def prepare_dataset(cfg: Config, *, overwrite: bool = False) -> dict[str, Any]:
    """Validate every source, then publish ``cache_dir/dataset.json`` atomically.

    Identical content returns the existing manifest without rewriting it. Any
    changed content/settings requires explicit ``overwrite=True``. Invalid
    datasets never replace an existing manifest or create a cache directory.
    """
    if type(overwrite) is not bool:
        raise ValueError("overwrite 必須是布林值。")
    checked = _checked_config(cfg)
    path = Path(checked.training["cache_dir"]) / _MANIFEST_NAME
    _reject_symlink_chain(path)
    current = _build_manifest(checked)
    if path.exists():
        try:
            previous = _read_manifest(path)
        except ValueError:
            if not overwrite:
                raise
        else:
            if _canonical(previous) == _canonical(current):
                return previous
            if not overwrite:
                raise ValueError(
                    "資料或設定已變更，現有 manifest 已過期；請使用 prepare --overwrite。"
                )
    _atomic_write(path, current, overwrite=overwrite)
    return current


def load_manifest(cfg: Config) -> dict[str, Any]:
    """Read and fully revalidate a manifest without writing anything.

    Exact canonical comparison catches schema/type changes (including bool/int
    confusion), path escapes, altered metadata, added/removed sources, source
    byte changes, and changed dataset/preprocessing settings. Untrusted paths
    from the manifest are never opened; only configured source paths are read.
    """
    checked = _checked_config(cfg)
    path = Path(checked.training["cache_dir"]) / _MANIFEST_NAME
    _reject_symlink_chain(path)
    stored = _read_manifest(path)
    current = _build_manifest(checked)
    if _canonical(stored) != _canonical(current):
        raise ValueError("manifest 無效或資料／設定已過期；請重新執行 prepare --overwrite。")
    return stored
