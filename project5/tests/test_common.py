from __future__ import annotations

from pathlib import Path

import common


def test_resolve_from_root_handles_relative_and_absolute(tmp_path: Path) -> None:
    rel = common.resolve_from_root("data/preprocessed_content.csv")
    assert rel.is_absolute()
    assert str(rel).endswith("data/preprocessed_content.csv")

    abs_path = tmp_path / "x.txt"
    resolved = common.resolve_from_root(abs_path)
    assert resolved == abs_path


def test_sanitize_filename_part() -> None:
    assert common.sanitize_filename_part("  AAPL / 2022  ") == "AAPL_2022"
    assert common.sanitize_filename_part("***", fallback="F") == "F"


def test_parse_ticker_year_from_filename() -> None:
    assert common.parse_ticker_year_from_filename("AAPL_2022.txt") == ("AAPL", 2022)
    assert common.parse_ticker_year_from_filename("AAPL_2022_3.txt") == ("AAPL", 2022)
    assert common.parse_ticker_year_from_filename("badname.txt") == (None, None)


def test_strict_chunk_tokens_within_bounds() -> None:
    tokens = [f"t{i}" for i in range(1000)]
    chunks = common.strict_chunk_tokens(tokens, min_tokens=201, max_tokens=499, target_tokens=420)
    assert chunks
    assert sum(len(c) for c in chunks) == len(tokens)
    assert all(201 <= len(c) <= 499 for c in chunks)


def test_strict_chunk_tokens_rejects_too_short_or_impossible() -> None:
    assert common.strict_chunk_tokens(["a"] * 200, min_tokens=201, max_tokens=499, target_tokens=420) == []
    impossible = common.strict_chunk_tokens(["a"] * 1001, min_tokens=400, max_tokens=450, target_tokens=420)
    assert impossible == []
