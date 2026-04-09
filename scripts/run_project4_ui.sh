#!/usr/bin/env bash
set -euo pipefail

SENTIMENT_ROOT="${SENTIMENT_ROOT:-outputs/project4/task1_sentiment}"
QA_ROOT="${QA_ROOT:-outputs/project4/task2_reading_comprehension}"
REPORT_TEX="${REPORT_TEX:-report/project4_report.tex}"

if [[ -x ".venv/bin/streamlit" ]]; then
  STREAMLIT_BIN=".venv/bin/streamlit"
else
  STREAMLIT_BIN="streamlit"
fi

STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
"$STREAMLIT_BIN" run src/project4_dashboard.py "$@" -- --sentiment-root "${SENTIMENT_ROOT}" --qa-root "${QA_ROOT}" --report-tex "${REPORT_TEX}"
