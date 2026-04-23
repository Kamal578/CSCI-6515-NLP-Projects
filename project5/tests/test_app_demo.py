from __future__ import annotations

from types import SimpleNamespace

from conftest import load_src_module


class FakeNode:
    def __init__(self, text: str, metadata: dict):
        self._text = text
        self.metadata = metadata

    def get_content(self, metadata_mode: str = "none") -> str:
        return self._text


class FakeSourceNode:
    def __init__(self, text: str, metadata: dict, score: float):
        self.node = FakeNode(text, metadata)
        self.score = score


class FakeResponse:
    def __init__(self, answer: str, source_nodes: list[FakeSourceNode]):
        self._answer = answer
        self.source_nodes = source_nodes

    def __str__(self) -> str:
        return self._answer


def test_response_to_history_entry() -> None:
    module = load_src_module("04_app.py", "app_mod_history")

    response = FakeResponse(
        "Grounded answer",
        [
            FakeSourceNode("chunk one", {"source_file": "AAPL_2022.txt", "ticker": "AAPL", "year": 2022}, 0.12),
            FakeSourceNode("chunk two", {"source_file": "MSFT_2021.txt", "ticker": "MSFT", "year": 2021}, 0.20),
        ],
    )

    entry = module.response_to_history_entry("What is disclosed?", response, "2026-04-22T00:00:00+00:00")
    assert entry["question"] == "What is disclosed?"
    assert entry["answer"] == "Grounded answer"
    assert entry["source_count"] == 2
    assert "AAPL_2022.txt" in entry["source_files"]


def test_history_to_markdown() -> None:
    module = load_src_module("04_app.py", "app_mod_markdown")

    history = [
        {
            "asked_at_utc": "2026-04-22T00:00:00+00:00",
            "question": "Q1",
            "answer": "A1",
            "source_files": ["AAPL_2022.txt"],
            "source_count": 1,
            "sources": [],
        }
    ]

    markdown = module.history_to_markdown(history)
    assert "# ESG RAG Session Export" in markdown
    assert "Q1" in markdown
    assert "AAPL_2022.txt" in markdown
