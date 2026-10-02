#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  printf '%s\n' '尚未安裝。請在專案目錄執行：uv sync --locked --group dev' >&2
  exit 2
fi
export TOKENIZERS_PARALLELISM=false
# Does not cd: explicit CLI file arguments retain the caller's cwd semantics.
# Provide a predictable default config even when invoked outside the project.
if [[ $# -gt 0 && "$1" != "smoke-test" && "$1" != "verify-upstream" && "$1" != "--help" && "$1" != "-h" ]]; then
  HAS_CONFIG=false
  for arg in "$@"; do
    if [[ "$arg" == "--config" || "$arg" == --config=* ]]; then HAS_CONFIG=true; fi
  done
  if [[ "$HAS_CONFIG" == false ]]; then set -- "$@" --config "$ROOT/configs/small.toml"; fi
fi
exec "$ROOT/.venv/bin/python" -m qwen21_trainer.cli "$@"
