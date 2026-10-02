"""Credential-safe, reserialized snapshots; never re-open the source config.

Paths and prompts are intentional payloads. Credentials, URL authentication and
known credential environment values are not. Source comments are never retained.
This module does not import the W&B SDK or inspect training data/model files.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import parse_qsl, quote, quote_plus, urlencode, urlsplit, urlunsplit

REDACTED = "[REDACTED]"
_URL = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s<>\"']+")
_CREDENTIAL_KEYS = {
    "key",
    "apikey",
    "accesskey",
    "accesskeyid",
    "secretaccesskey",
    "accountkey",
    "token",
    "accesstoken",
    "refreshtoken",
    "authtoken",
    "bearertoken",
    "idtoken",
    "password",
    "passwd",
    "pwd",
    "secret",
    "clientsecret",
    "privatekey",
    "credential",
    "credentials",
    "authorization",
    "auth",
    "authentication",
    "cookie",
    "cookies",
    "sessionid",
    "sessiontoken",
    "signature",
    "sig",
    "sas",
    "sastoken",
    "connectionstring",
    "encryptionkey",
    "signingkey",
}
_CREDENTIAL_SUFFIXES = (
    "apikey",
    "accesstoken",
    "refreshtoken",
    "authtoken",
    "bearertoken",
    "accesskey",
    "accesskeyid",
    "secretaccesskey",
    "clientsecret",
    "privatekey",
    "password",
    "passwd",
    "credential",
    "credentials",
    "signature",
)


def credential_key(key):
    """Recognize credentials, not tokenizer/max_tokens/token_count settings."""
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(key)).lower()
    compact = re.sub(r"[^a-z0-9]", "", name)
    tokens = re.findall(r"[a-z0-9]+", name)
    words = set(tokens)
    return (
        compact in _CREDENTIAL_KEYS
        or compact.endswith(_CREDENTIAL_SUFFIXES)
        or bool(words & {"password", "passwd", "secret", "credentials", "credential"})
        or bool(tokens and tokens[-1] == "token")
        or name.endswith("_api_key")
    )


class ConfigSanitizer:
    """One redaction policy shared by run.config, artifacts and image captions."""

    def __init__(self):
        # Values are never emitted in diagnostics or returned from this object.
        values = {value for key, value in os.environ.items() if credential_key(key) and value}
        encoded = {
            variant
            for value in values
            for variant in (value, quote(value, safe=""), quote_plus(value))
        }
        self._secrets = sorted(encoded, key=len, reverse=True)

    def _replace_known(self, text):
        for value in self._secrets:
            text = text.replace(value, REDACTED)
        return text

    def _url(self, match):
        try:
            parts = urlsplit(match.group(0))
            # Remove the complete userinfo, including a username without password.
            netloc = parts.netloc.rsplit("@", 1)[-1]
            query = urlencode(
                [
                    (key, REDACTED if credential_key(key) else value)
                    for key, value in parse_qsl(parts.query, keep_blank_values=True)
                ]
            )
            # OAuth implicit-flow fragments can contain credential parameters too.
            fragment = parts.fragment
            if "=" in fragment:
                fragment = urlencode(
                    [
                        (key, REDACTED if credential_key(key) else value)
                        for key, value in parse_qsl(fragment, keep_blank_values=True)
                    ]
                )
            return urlunsplit((parts.scheme, netloc, parts.path, query, fragment))
        except ValueError:
            return REDACTED

    def text(self, value):
        return self._replace_known(_URL.sub(self._url, str(value)))

    def sanitize(self, value):
        if isinstance(value, Mapping):
            return {
                self.text(key): REDACTED if credential_key(key) else self.sanitize(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.sanitize(item) for item in value]
        if value is None or type(value) in (bool, int):
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("設定快照不可包含非有限數值。")
            return value
        if isinstance(value, (str, Path)):
            return self.text(value)
        # Do not stringify arbitrary objects (repr may expose hidden credentials).
        raise ValueError("設定快照包含不支援的資料型別。")


def configuration_snapshot(cfg, sanitizer):
    """Return sanitized effective/source documents and explicit provenance."""
    effective = sanitizer.sanitize(cfg.to_dict())
    source = getattr(cfg, "source_document", None)
    source = sanitizer.sanitize(source) if source is not None else None
    source_format = getattr(cfg, "source_format", None)
    metadata = {
        "schema_version": 1,
        "sanitized": True,
        "comments_preserved": False,
        "source_available": source is not None,
        "source_format": source_format if source_format in {"toml", "yaml"} else None,
        "source_semantics": "sanitized-as-loaded" if source is not None else None,
        "source_representation": "parsed-at-load-reserialized-yaml" if source is not None else None,
        "effective_representation": "resolved-config-json",
    }
    return effective, source, metadata


def write_configuration_snapshot(output_dir, effective, source, metadata):
    """Write only sanitized documents. Never copy a raw file or an output tree."""
    directory = Path(output_dir) / "wandb-config"
    if directory.is_symlink():
        raise ValueError("W&B 設定快照目錄不可為符號連結。")
    directory.mkdir(parents=True, exist_ok=True)
    documents = {
        "effective-config.json": json.dumps(
            effective, ensure_ascii=False, indent=2, allow_nan=False
        )
        + "\n",
        "snapshot-metadata.json": json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    }
    if source is not None:
        import yaml

        documents["source-config.yaml"] = yaml.safe_dump(
            source, allow_unicode=True, sort_keys=False
        )
    paths = []
    for name, content in documents.items():
        destination = directory / name
        temporary = None
        try:
            # Atomically replace an old snapshot, never follow a destination link.
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=directory, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        paths.append(destination)
    return paths
