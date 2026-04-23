#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
}

command -v ollama >/dev/null 2>&1 || {
  echo "Ollama CLI is not installed or not in PATH." >&2
  exit 1
}

"${PYTHON_BIN}" -m pip install -r requirements.txt
ollama pull "${OLLAMA_EMBED_MODEL:-nomic-embed-text}"
ollama pull "${OLLAMA_LLM_MODEL:-llama3.1}"

echo "Setup complete."
