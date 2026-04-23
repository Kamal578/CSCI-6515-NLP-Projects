from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.config import Settings as ChromaSettings


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_LLM_MODEL = "llama3.1"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
DEFAULT_CHROMA_PATH = "vector_db"
DEFAULT_CHROMA_COLLECTION = "sp500_esg_reports"


@dataclass(frozen=True)
class RuntimeConfig:
    ollama_base_url: str
    llm_model: str
    embed_model: str
    chroma_path: str
    chroma_collection: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_from_root(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return project_root() / path


def runtime_config_from_env() -> RuntimeConfig:
    return RuntimeConfig(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        llm_model=os.getenv("OLLAMA_LLM_MODEL", DEFAULT_OLLAMA_LLM_MODEL),
        embed_model=os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_EMBED_MODEL),
        chroma_path=os.getenv("CHROMA_PATH", DEFAULT_CHROMA_PATH),
        chroma_collection=os.getenv("CHROMA_COLLECTION", DEFAULT_CHROMA_COLLECTION),
    )


def chroma_persistent_client(chroma_path: str | Path) -> chromadb.ClientAPI:
    path = resolve_from_root(chroma_path)
    settings = ChromaSettings(
        anonymized_telemetry=False,
        is_persistent=True,
        persist_directory=str(path),
    )
    return chromadb.PersistentClient(path=str(path), settings=settings)


def sanitize_filename_part(value: object, fallback: str = "UNKNOWN") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    cleaned = cleaned.strip("_.-")
    return cleaned or fallback


def parse_ticker_year_from_filename(filename: str) -> tuple[str | None, int | None]:
    stem = Path(filename).stem
    match = re.match(r"^(?P<ticker>[A-Za-z0-9.-]+)_(?P<year>\d{4})(?:_\d+)?$", stem)
    if not match:
        return None, None
    return match.group("ticker"), int(match.group("year"))


def tokenize(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def strict_chunk_tokens(
    tokens: Iterable[str],
    *,
    min_tokens: int = 201,
    max_tokens: int = 499,
    target_tokens: int = 420,
) -> list[list[str]]:
    token_list = list(tokens)
    n_tokens = len(token_list)
    if n_tokens < min_tokens:
        return []

    k_low = math.ceil(n_tokens / max_tokens)
    k_high = n_tokens // min_tokens
    if k_low > k_high:
        return []

    k_target = max(1, round(n_tokens / target_tokens))
    n_chunks = min(max(k_target, k_low), k_high)

    base, extra = divmod(n_tokens, n_chunks)
    chunk_sizes = [base + 1 if i < extra else base for i in range(n_chunks)]
    if any(size < min_tokens or size > max_tokens for size in chunk_sizes):
        return []

    chunks: list[list[str]] = []
    cursor = 0
    for size in chunk_sizes:
        chunks.append(token_list[cursor : cursor + size])
        cursor += size
    return chunks
