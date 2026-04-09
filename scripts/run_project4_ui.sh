#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="outputs/project4/task2_reading_comprehension"
if [[ $# -gt 0 ]]; then
  OUTPUT_ROOT="$1"
  shift
fi

REPORT_TEX="${REPORT_TEX:-report/project4_report.tex}"

if [[ -x ".venv/bin/streamlit" ]]; then
  STREAMLIT_BIN=".venv/bin/streamlit"
else
  STREAMLIT_BIN="streamlit"
fi

STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
"$STREAMLIT_BIN" run src/project4_dashboard.py "$@" -- --output-root "${OUTPUT_ROOT}" --report-tex "${REPORT_TEX}"
