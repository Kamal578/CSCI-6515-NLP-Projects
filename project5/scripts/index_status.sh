#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PID_LIST="$(pgrep -f "python src/02_build_index.py" || true)"
if [[ -z "${PID_LIST}" ]]; then
  echo "Index process: not running"
else
  echo "Index process: running"
  ps -p ${PID_LIST} -o pid,etime,%cpu,%mem,command
fi

python - <<'PY'
import sqlite3
from pathlib import Path

db_path = Path("vector_db/chroma.sqlite3").resolve()
collection_name = "sp500_esg_reports"

query = """
SELECT COUNT(*)
FROM embeddings e
JOIN segments s ON e.segment_id = s.id
JOIN collections c ON s.collection = c.id
WHERE c.name = ?
"""

if not db_path.exists():
    print("Vector DB sqlite file not found")
else:
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(query, (collection_name,)).fetchone()
        count = int(row[0]) if row else 0
        print(f"Collection: {collection_name}")
        print(f"Vector count: {count}")
    except sqlite3.Error as exc:
        print(f"Collection status unavailable: {exc}")
PY
