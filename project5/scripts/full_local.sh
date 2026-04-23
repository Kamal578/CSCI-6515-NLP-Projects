#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

"${SCRIPT_DIR}/prepare_data.sh"
"${SCRIPT_DIR}/build_index.sh" --rebuild
"${SCRIPT_DIR}/evaluate.sh"
"${SCRIPT_DIR}/eval_metrics.sh"
