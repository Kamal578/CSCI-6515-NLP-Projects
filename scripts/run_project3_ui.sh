#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="outputs/project3"
if [[ $# -gt 0 ]]; then
  OUTPUT_ROOT="$1"
  shift
fi

STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
streamlit run src/project3_dashboard.py "$@" -- --output-root "${OUTPUT_ROOT}"
