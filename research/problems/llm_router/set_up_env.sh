#!/usr/bin/env bash
set -euo pipefail

# Prepare problem resources for llm_router
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXEC_ROOT=$(cd "$SCRIPT_DIR/../../../execution_env" && pwd 2>/dev/null || true)

echo "[setup] llm_router setup starting..." >&2

# Use uv to create a venv and sync dependencies declared in resources/pyproject.toml
VENV_DIR="$EXEC_ROOT/.venv"
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
pip install --user uv >/dev/null 2>&1 || true
echo "[setup] Creating venv at $VENV_DIR" >&2
uv venv "$VENV_DIR"
export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"

PROJECT_DIR="$SCRIPT_DIR/resources"
if [[ -f "$PROJECT_DIR/pyproject.toml" ]]; then
  echo "[setup] uv sync project=$PROJECT_DIR" >&2
  uv --project "$PROJECT_DIR" sync --active
else
  echo "[setup] WARNING: pyproject.toml not found at $PROJECT_DIR; skipping dependency sync" >&2
fi

# Datasets are read directly from /datasets/llm_router/ by the evaluator
# No need to create datasets folder under problem directory

echo "[setup] llm_router setup done" >&2