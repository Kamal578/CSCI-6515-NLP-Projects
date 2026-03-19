from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, send_file

from .project3_common import read_json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "project3"
HTML_PATH = Path(__file__).resolve().parent / "project3_results_ui.html"

app = Flask(__name__, static_folder=None)


def _safe_read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return pd.read_csv(path).to_dict(orient="records")
    except Exception:
        return []


def _task_summary_map(output_root: Path) -> dict:
    return {
        "task1": read_json(output_root / "task1_matrices" / "summary.json"),
        "task2": read_json(output_root / "task2_word2vec" / "summary.json"),
        "task3": read_json(output_root / "task3_glove" / "summary.json"),
        "task4": read_json(output_root / "task4_compare" / "summary.json"),
        "task5": read_json(output_root / "task5_dl" / "summary.json"),
        "task6": read_json(output_root / "task6_report" / "summary.json"),
    }


@app.route("/")
def index():
    return send_file(HTML_PATH)


@app.route("/api/overview")
def overview():
    output_root = Path(app.config["PROJECT3_OUTPUT_ROOT"])
    payload = _task_summary_map(output_root)
    return jsonify(payload)


@app.route("/api/table/task4")
def task4_table():
    output_root = Path(app.config["PROJECT3_OUTPUT_ROOT"])
    rows = _safe_read_csv(output_root / "task4_compare" / "comparison.csv")
    return jsonify({"rows": rows})


@app.route("/api/table/task5")
def task5_table():
    output_root = Path(app.config["PROJECT3_OUTPUT_ROOT"])
    rows = _safe_read_csv(output_root / "task5_dl" / "leaderboard.csv")
    return jsonify({"rows": rows})


@app.route("/api/table/task1_freq")
def task1_freq():
    output_root = Path(app.config["PROJECT3_OUTPUT_ROOT"])
    rows = _safe_read_csv(output_root / "task1_matrices" / "word_frequency_distribution.csv")
    return jsonify({"rows": rows[:100]})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve Project 3 results UI.")
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Path to outputs/project3 directory")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5060)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app.config["PROJECT3_OUTPUT_ROOT"] = str(Path(args.output_root))
    print(f"Serving Project 3 UI with outputs from: {app.config['PROJECT3_OUTPUT_ROOT']}")
    print(f"Open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
