from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest
from conftest import load_src_module


def test_normalized_year() -> None:
    module = load_src_module("01_data_prep.py", "data_prep_mod")
    assert module.normalized_year(2022.0) == "2022"
    assert module.normalized_year("2021") == "2021"
    assert module.normalized_year(None) == ""


def test_main_writes_txt_files_and_duplicate_suffix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    module = load_src_module("01_data_prep.py", "data_prep_mod_main")

    csv_path = tmp_path / "input.csv"
    out_dir = tmp_path / "raw_txt"
    df = pd.DataFrame(
        [
            {"ticker": "AAPL", "year": 2022, "preprocessed_content": "alpha"},
            {"ticker": "AAPL", "year": 2022, "preprocessed_content": "beta"},
            {"ticker": "MSFT", "year": 2021, "preprocessed_content": "   "},
            {"ticker": "TSLA", "year": None, "preprocessed_content": "gamma"},
        ]
    )
    df.to_csv(csv_path, index=False)

    monkeypatch.setattr(module, "parse_args", lambda: Namespace(input_csv=str(csv_path), output_dir=str(out_dir)))

    rc = module.main()
    assert rc == 0

    files = sorted(p.name for p in out_dir.glob("*.txt"))
    assert files == ["AAPL_2022.txt", "AAPL_2022_2.txt"]
    assert (out_dir / "AAPL_2022.txt").read_text(encoding="utf-8") == "alpha"
    assert (out_dir / "AAPL_2022_2.txt").read_text(encoding="utf-8") == "beta"

    output = capsys.readouterr().out
    assert "Files written: 2" in output
    assert "Rows skipped: 2" in output


def test_main_raises_on_missing_required_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_src_module("01_data_prep.py", "data_prep_mod_missing")

    csv_path = tmp_path / "bad.csv"
    pd.DataFrame([{"ticker": "AAPL", "year": 2022}]).to_csv(csv_path, index=False)
    out_dir = tmp_path / "raw_txt"

    monkeypatch.setattr(module, "parse_args", lambda: Namespace(input_csv=str(csv_path), output_dir=str(out_dir)))

    with pytest.raises(ValueError, match="missing required columns"):
        module.main()
