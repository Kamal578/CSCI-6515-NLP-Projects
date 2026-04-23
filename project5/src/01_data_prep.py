from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from common import resolve_from_root, sanitize_filename_part


REQUIRED_COLUMNS = {"ticker", "year", "preprocessed_content"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split ESG CSV rows into TXT files.")
    parser.add_argument("--input-csv", default="data/preprocessed_content.csv")
    parser.add_argument("--output-dir", default="data/raw_txt")
    return parser.parse_args()


def normalized_year(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return sanitize_filename_part(value)


def main() -> int:
    args = parse_args()
    input_csv = resolve_from_root(args.input_csv)
    output_dir = resolve_from_root(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    total_rows = len(df)
    written = 0
    skipped = 0
    duplicate_counters: dict[str, int] = defaultdict(int)

    for row in df.itertuples(index=False):
        ticker = sanitize_filename_part(getattr(row, "ticker"), fallback="UNKNOWN").upper()
        year = normalized_year(getattr(row, "year"))
        content = getattr(row, "preprocessed_content")

        if not year or pd.isna(content):
            skipped += 1
            continue

        text = str(content).strip()
        if not text:
            skipped += 1
            continue

        base_name = f"{ticker}_{year}"
        duplicate_counters[base_name] += 1
        file_name = f"{base_name}.txt"
        if duplicate_counters[base_name] > 1:
            file_name = f"{base_name}_{duplicate_counters[base_name]}.txt"

        output_path = output_dir / file_name
        output_path.write_text(text, encoding="utf-8")
        written += 1

    print(f"Input CSV: {input_csv}")
    print(f"Output dir: {output_dir}")
    print(f"Total rows: {total_rows}")
    print(f"Files written: {written}")
    print(f"Rows skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
