from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

from conftest import load_src_module


def _make_chroma_sqlite(path: Path, collection_name: str = "sp500_esg_reports", embeddings: int = 2) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT, dimension INTEGER, database_id TEXT, config_json_str TEXT, schema_str TEXT)")
    conn.execute("CREATE TABLE segments (id TEXT PRIMARY KEY, type TEXT, scope TEXT, collection TEXT)")
    conn.execute("CREATE TABLE embeddings (id INTEGER PRIMARY KEY, segment_id TEXT, embedding_id TEXT, seq_id BLOB, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    conn.execute(
        "INSERT INTO collections (id, name, dimension, database_id, config_json_str, schema_str) VALUES (?, ?, ?, ?, ?, ?)",
        ("c1", collection_name, 768, "db1", "{}", "{}"),
    )
    conn.execute("INSERT INTO segments (id, type, scope, collection) VALUES (?, ?, ?, ?)", ("s1", "vector", "local", "c1"))
    for i in range(embeddings):
        conn.execute(
            "INSERT INTO embeddings (segment_id, embedding_id, seq_id) VALUES (?, ?, ?)",
            ("s1", f"e{i}", b"1"),
        )
    conn.commit()
    conn.close()


def test_format_score() -> None:
    module = load_src_module("04_app.py", "app_mod_format")
    assert module.format_score(None) == "N/A"
    assert module.format_score(0.123456) == "0.1235"


def test_count_vectors_from_sqlite(tmp_path: Path) -> None:
    module = load_src_module("04_app.py", "app_mod_count")
    chroma_dir = tmp_path / "vector_db"
    chroma_dir.mkdir()
    _make_chroma_sqlite(chroma_dir / "chroma.sqlite3", embeddings=5)
    assert module.count_vectors_from_sqlite(chroma_dir, "sp500_esg_reports") == 5


def test_collection_exists() -> None:
    module = load_src_module("04_app.py", "app_mod_exists")
    client = SimpleNamespace(list_collections=lambda: [SimpleNamespace(name="abc")])
    assert module.collection_exists(client, "abc") is True
    assert module.collection_exists(client, "nope") is False
