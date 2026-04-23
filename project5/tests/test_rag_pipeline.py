from __future__ import annotations

import sqlite3
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import load_src_module


def _make_chroma_sqlite(path: Path, collection_name: str = "sp500_esg_reports", embeddings: int = 3) -> None:
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


def test_load_questions_defaults_and_file(tmp_path: Path) -> None:
    module = load_src_module("03_rag_pipeline.py", "rag_mod_questions")

    args_default = Namespace(question=None, questions_file=None)
    qs_default = module.load_questions(args_default)
    assert len(qs_default) == 5

    qf = tmp_path / "questions.txt"
    qf.write_text("Q1\n\nQ2\n", encoding="utf-8")
    args_file = Namespace(question=None, questions_file=str(qf))
    qs_file = module.load_questions(args_file)
    assert qs_file == ["Q1", "Q2"]


def test_count_vectors_from_sqlite(tmp_path: Path) -> None:
    module = load_src_module("03_rag_pipeline.py", "rag_mod_count")

    chroma_dir = tmp_path / "vector_db"
    chroma_dir.mkdir()
    _make_chroma_sqlite(chroma_dir / "chroma.sqlite3", embeddings=7)
    assert module.count_vectors_from_sqlite(chroma_dir, "sp500_esg_reports") == 7


def test_unique_preserve_order() -> None:
    module = load_src_module("03_rag_pipeline.py", "rag_mod_unique")
    assert module.unique_preserve_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_write_results_creates_json_and_csv(tmp_path: Path) -> None:
    module = load_src_module("03_rag_pipeline.py", "rag_mod_write")
    records = [
        {
            "question_index": 1,
            "question": "Q1",
            "baseline_answer": "B1",
            "rag_answer": "R1",
            "retrieved_files": ["AAPL_2022.txt"],
            "retrieved_chunks": [],
        }
    ]
    json_path, csv_path = module.write_results(records, str(tmp_path / "outputs"))
    assert json_path.exists()
    assert csv_path.exists()
    assert "rag_eval_" in json_path.name
    assert "rag_eval_" in csv_path.name


def test_load_query_engine_missing_path_raises(tmp_path: Path) -> None:
    module = load_src_module("03_rag_pipeline.py", "rag_mod_missing")
    args = Namespace(
        chroma_path=str(tmp_path / "missing"),
        collection="sp500_esg_reports",
        ollama_base_url="http://localhost:11434",
        llm_model="llama3.1",
        embed_model="nomic-embed-text",
        context_window=2048,
        top_k=3,
    )
    with pytest.raises(RuntimeError, match="Chroma path not found"):
        module.load_query_engine(args)


def test_load_query_engine_happy_path_with_mocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_src_module("03_rag_pipeline.py", "rag_mod_happy")

    chroma_dir = tmp_path / "vector_db"
    chroma_dir.mkdir()
    _make_chroma_sqlite(chroma_dir / "chroma.sqlite3", embeddings=4)

    args = Namespace(
        chroma_path=str(chroma_dir),
        collection="sp500_esg_reports",
        ollama_base_url="http://localhost:11434",
        llm_model="llama3.1",
        embed_model="nomic-embed-text",
        context_window=2048,
        top_k=3,
    )

    class FakeCollection:
        def peek(self, limit=1):
            return {"ids": ["id1"]}

    class FakeClient:
        def list_collections(self):
            return [SimpleNamespace(name="sp500_esg_reports")]

        def get_collection(self, name):
            return FakeCollection()

    class FakeIndex:
        def as_query_engine(self, similarity_top_k, llm, text_qa_template, refine_template):
            return "query_engine"

    monkeypatch.setattr(module, "chroma_persistent_client", lambda _path: FakeClient())
    monkeypatch.setattr(module, "Ollama", lambda **kwargs: "llm")
    monkeypatch.setattr(module, "OllamaEmbedding", lambda **kwargs: "embed")
    monkeypatch.setattr(module, "ChromaVectorStore", lambda **kwargs: "vector_store")
    monkeypatch.setattr(module.VectorStoreIndex, "from_vector_store", lambda **kwargs: FakeIndex())
    monkeypatch.setattr(module, "Settings", SimpleNamespace(llm=None, embed_model=None))

    llm, query_engine, count = module.load_query_engine(args)
    assert llm == "llm"
    assert query_engine == "query_engine"
    assert count == 4
