from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from common import (
    chroma_persistent_client,
    parse_ticker_year_from_filename,
    resolve_from_root,
    runtime_config_from_env,
    strict_chunk_tokens,
    tokenize,
)


def parse_args() -> argparse.Namespace:
    cfg = runtime_config_from_env()
    parser = argparse.ArgumentParser(description="Build Chroma vector index from TXT ESG reports.")
    parser.add_argument("--input-dir", default="data/raw_txt")
    parser.add_argument("--chroma-path", default=cfg.chroma_path)
    parser.add_argument("--collection", default=cfg.chroma_collection)
    parser.add_argument("--ollama-base-url", default=cfg.ollama_base_url)
    parser.add_argument("--embed-model", default=cfg.embed_model)
    parser.add_argument("--min-tokens", type=int, default=201)
    parser.add_argument("--max-tokens", type=int, default=499)
    parser.add_argument("--target-tokens", type=int, default=420)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def get_or_create_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
    rebuild: bool,
):
    if rebuild:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    return client.get_or_create_collection(collection_name)


def load_documents(input_dir: Path):
    reader = SimpleDirectoryReader(input_dir=str(input_dir), required_exts=[".txt"], recursive=True)
    return reader.load_data()


def main() -> int:
    args = parse_args()

    if not (args.min_tokens < args.target_tokens < args.max_tokens):
        raise ValueError("Token settings must satisfy min_tokens < target_tokens < max_tokens.")

    input_dir = resolve_from_root(args.input_dir)
    chroma_path = resolve_from_root(args.chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    documents = load_documents(input_dir)
    if not documents:
        raise RuntimeError(f"No .txt files found in: {input_dir}")

    nodes: list[TextNode] = []
    skipped_docs = 0

    for doc in documents:
        text = doc.text or ""
        if not text.strip():
            skipped_docs += 1
            continue

        metadata = doc.metadata or {}
        source_path = metadata.get("file_path") or metadata.get("filename") or ""
        source_file = Path(str(source_path)).name or "unknown.txt"
        ticker, year = parse_ticker_year_from_filename(source_file)

        chunks = strict_chunk_tokens(
            tokenize(text),
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            target_tokens=args.target_tokens,
        )
        if not chunks:
            skipped_docs += 1
            continue

        stem = Path(source_file).stem
        for idx, chunk_tokens in enumerate(chunks, start=1):
            chunk_text = " ".join(chunk_tokens)
            chunk_metadata = {
                "source_file": source_file,
                "chunk_id": idx,
                "token_count": len(chunk_tokens),
            }
            if ticker is not None:
                chunk_metadata["ticker"] = ticker
            if year is not None:
                chunk_metadata["year"] = year

            node = TextNode(
                text=chunk_text,
                metadata=chunk_metadata,
                id_=f"{stem}::chunk_{idx}",
            )
            nodes.append(node)

    if not nodes:
        raise RuntimeError(
            "No valid chunks were produced. Check input text sizes and chunking constraints."
        )

    invalid_nodes = [
        n
        for n in nodes
        if not (args.min_tokens <= int(n.metadata.get("token_count", 0)) <= args.max_tokens)
    ]
    if invalid_nodes:
        raise RuntimeError("Chunk constraints violated: found stored chunks outside token bounds.")

    embed_model = OllamaEmbedding(model_name=args.embed_model, base_url=args.ollama_base_url)
    Settings.embed_model = embed_model

    client = chroma_persistent_client(chroma_path)
    collection = get_or_create_collection(client, args.collection, args.rebuild)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=embed_model)

    print(f"Input dir: {input_dir}")
    print(f"Chroma path: {chroma_path}")
    print(f"Collection: {args.collection}")
    print(f"Documents loaded: {len(documents)}")
    print(f"Documents skipped: {skipped_docs}")
    print(f"Chunks created: {len(nodes)}")
    print(f"Chunks stored in collection: {collection.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
