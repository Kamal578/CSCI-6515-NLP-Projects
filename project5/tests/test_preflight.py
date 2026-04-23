from __future__ import annotations

from pathlib import Path

from conftest import load_src_module


def test_parse_ollama_list_models() -> None:
    module = load_src_module("preflight.py", "preflight_mod_parse")
    output = """NAME            ID              SIZE      MODIFIED
llama3.1:latest abc123          4.9 GB    2 days ago
nomic-embed-text:latest def456  274 MB    2 days ago
"""
    models = module.parse_ollama_list_models(output)
    assert "llama3.1:latest" in models
    assert "llama3.1" in models
    assert "nomic-embed-text" in models


def test_check_model_available() -> None:
    module = load_src_module("preflight.py", "preflight_mod_check")
    models = {"llama3.1", "nomic-embed-text:latest"}
    assert module.check_model_available(models, "llama3.1") is True
    assert module.check_model_available(models, "nomic-embed-text") is True
    assert module.check_model_available(models, "mistral") is False


def test_check_paths(tmp_path: Path) -> None:
    module = load_src_module("preflight.py", "preflight_mod_paths")

    input_csv = tmp_path / "preprocessed_content.csv"
    input_csv.write_text("ticker,year,preprocessed_content\n", encoding="utf-8")

    raw_txt = tmp_path / "raw_txt"
    raw_txt.mkdir()
    (raw_txt / "AAPL_2022.txt").write_text("abc", encoding="utf-8")

    chroma_path = tmp_path / "vector_db"
    chroma_path.mkdir()

    errors, warnings = module.check_paths(input_csv, raw_txt, chroma_path)
    assert errors == []
    assert warnings == []


def test_check_paths_missing(tmp_path: Path) -> None:
    module = load_src_module("preflight.py", "preflight_mod_missing")
    errors, warnings = module.check_paths(
        tmp_path / "missing.csv",
        tmp_path / "missing_raw_txt",
        tmp_path / "missing_vector_db",
    )
    assert errors
    assert warnings
