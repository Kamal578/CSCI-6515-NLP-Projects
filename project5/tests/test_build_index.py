from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import load_src_module


class FakeCollection:
    def __init__(self) -> None:
        self._count = 0

    def count(self) -> int:
        return self._count


class FakeClient:
    def __init__(self, collection: FakeCollection) -> None:
        self.collection = collection
        self.deleted: list[str] = []

    def delete_collection(self, name: str) -> None:
        self.deleted.append(name)

    def get_or_create_collection(self, name: str) -> FakeCollection:
        return self.collection


def test_get_or_create_collection_rebuild_deletes() -> None:
    module = load_src_module("02_build_index.py", "build_index_mod_gc")
    coll = FakeCollection()
    client = FakeClient(coll)

    result = module.get_or_create_collection(client, "c1", rebuild=True)
    assert result is coll
    assert client.deleted == ["c1"]


def test_main_rejects_invalid_token_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_src_module("02_build_index.py", "build_index_mod_invalid")

    args = Namespace(
        input_dir=str(tmp_path / "raw_txt"),
        chroma_path=str(tmp_path / "vector_db"),
        collection="sp500_esg_reports",
        ollama_base_url="http://localhost:11434",
        embed_model="nomic-embed-text",
        min_tokens=500,
        max_tokens=300,
        target_tokens=420,
        rebuild=True,
    )
    monkeypatch.setattr(module, "parse_args", lambda: args)

    with pytest.raises(ValueError, match="Token settings"):
        module.main()


def test_main_builds_index_with_mocked_dependencies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = load_src_module("02_build_index.py", "build_index_mod_main")

    in_dir = tmp_path / "raw_txt"
    in_dir.mkdir(parents=True)

    args = Namespace(
        input_dir=str(in_dir),
        chroma_path=str(tmp_path / "vector_db"),
        collection="sp500_esg_reports",
        ollama_base_url="http://localhost:11434",
        embed_model="nomic-embed-text",
        min_tokens=201,
        max_tokens=499,
        target_tokens=420,
        rebuild=False,
    )
    monkeypatch.setattr(module, "parse_args", lambda: args)

    doc = SimpleNamespace(
        text=" ".join([f"tok{i}" for i in range(450)]),
        metadata={"file_path": str(in_dir / "AAPL_2022.txt")},
    )
    monkeypatch.setattr(module, "load_documents", lambda _path: [doc])

    collection = FakeCollection()
    client = FakeClient(collection)
    monkeypatch.setattr(module, "chroma_persistent_client", lambda _path: client)

    captured: dict[str, object] = {}

    def fake_vector_store(*, chroma_collection):
        captured["collection"] = chroma_collection
        return "vector_store"

    class FakeStorageContext:
        @staticmethod
        def from_defaults(vector_store):
            captured["vector_store"] = vector_store
            return "storage_context"

    class FakeEmbed:
        pass

    def fake_embedding(*, model_name, base_url):
        captured["embedding"] = (model_name, base_url)
        return FakeEmbed()

    def fake_vector_index(*, nodes, storage_context, embed_model):
        captured["nodes"] = nodes
        captured["storage_context"] = storage_context
        captured["embed_model"] = embed_model
        return "index"

    monkeypatch.setattr(module, "ChromaVectorStore", fake_vector_store)
    monkeypatch.setattr(module, "StorageContext", FakeStorageContext)
    monkeypatch.setattr(module, "OllamaEmbedding", fake_embedding)
    monkeypatch.setattr(module, "VectorStoreIndex", fake_vector_index)
    monkeypatch.setattr(module, "Settings", SimpleNamespace(embed_model=None))

    rc = module.main()
    assert rc == 0
    nodes = captured["nodes"]
    assert isinstance(nodes, list)
    assert len(nodes) == 1
    node = nodes[0]
    assert node.metadata["source_file"] == "AAPL_2022.txt"
    assert node.metadata["ticker"] == "AAPL"
    assert node.metadata["year"] == 2022
    assert 201 <= node.metadata["token_count"] <= 499
