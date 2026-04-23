from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from common import chroma_persistent_client, resolve_from_root, runtime_config_from_env


def parse_args() -> argparse.Namespace:
    cfg = runtime_config_from_env()
    parser = argparse.ArgumentParser(description="Run local environment checks for the ESG RAG project.")
    parser.add_argument("--input-csv", default="data/preprocessed_content.csv")
    parser.add_argument("--raw-txt-dir", default="data/raw_txt")
    parser.add_argument("--chroma-path", default=cfg.chroma_path)
    parser.add_argument("--collection", default=cfg.chroma_collection)
    parser.add_argument("--llm-model", default=cfg.llm_model)
    parser.add_argument("--embed-model", default=cfg.embed_model)
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument("--require-index", action="store_true")
    return parser.parse_args()


def parse_ollama_list_models(output: str) -> set[str]:
    models: set[str] = set()
    for line in output.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("name"):
            continue
        name = line.split()[0]
        if name:
            models.add(name)
            models.add(name.split(":", 1)[0])
    return models


def check_model_available(models: set[str], expected: str) -> bool:
    expected = expected.strip()
    if not expected:
        return False
    if expected in models:
        return True
    expected_base = expected.split(":", 1)[0]
    if expected_base in models:
        return True
    for model in models:
        if model.split(":", 1)[0] == expected_base:
            return True
    return False


def get_installed_ollama_models() -> set[str]:
    proc = subprocess.run(
        ["ollama", "list"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return set()
    return parse_ollama_list_models(proc.stdout)


def check_paths(input_csv: Path, raw_txt_dir: Path, chroma_path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not input_csv.exists():
        errors.append(f"Missing dataset CSV: {input_csv}")

    if not raw_txt_dir.exists():
        warnings.append(f"Missing raw TXT directory: {raw_txt_dir}")
    else:
        txt_count = sum(1 for _ in raw_txt_dir.glob("*.txt"))
        if txt_count == 0:
            warnings.append(f"No TXT files found in {raw_txt_dir}.")

    if not chroma_path.exists():
        warnings.append(f"Vector DB path does not exist yet: {chroma_path}")

    return errors, warnings


def check_collection_has_records(chroma_path: Path, collection_name: str) -> tuple[bool, str]:
    if not chroma_path.exists():
        return False, f"Chroma path not found: {chroma_path}"

    client = chroma_persistent_client(chroma_path)
    names = []
    for item in client.list_collections():
        names.append(item.name if hasattr(item, "name") else str(item))

    if collection_name not in names:
        return False, f"Collection '{collection_name}' not found in {chroma_path}"

    collection = client.get_collection(collection_name)
    sample = collection.peek(limit=1)
    ids = sample.get("ids", []) if isinstance(sample, dict) else []
    if not ids:
        return False, f"Collection '{collection_name}' exists but is empty"

    return True, f"Collection '{collection_name}' is present and non-empty"


def main() -> int:
    args = parse_args()

    input_csv = resolve_from_root(args.input_csv)
    raw_txt_dir = resolve_from_root(args.raw_txt_dir)
    chroma_path = resolve_from_root(args.chroma_path)

    errors, warnings = check_paths(input_csv, raw_txt_dir, chroma_path)

    print("Preflight checks")
    print("=" * 80)
    print(f"Python executable: {shutil.which('python') or 'not found'}")
    print(f"OLLAMA_BASE_URL: {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}")
    print(f"Dataset CSV: {input_csv}")
    print(f"Raw TXT dir: {raw_txt_dir}")
    print(f"Chroma path: {chroma_path}")
    print(f"Collection: {args.collection}")

    if shutil.which("ollama") is None:
        warnings.append("Ollama CLI is not in PATH.")

    if not args.skip_model_check and shutil.which("ollama") is not None:
        models = get_installed_ollama_models()
        if not models:
            warnings.append("Unable to inspect installed Ollama models via `ollama list`.")
        else:
            if not check_model_available(models, args.embed_model):
                errors.append(f"Embedding model not found in Ollama: {args.embed_model}")
            if not check_model_available(models, args.llm_model):
                errors.append(f"LLM model not found in Ollama: {args.llm_model}")

    if args.require_index:
        ok, message = check_collection_has_records(chroma_path, args.collection)
        if ok:
            print(f"Index status: {message}")
        else:
            errors.append(message)

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print("Errors:")
        for error in errors:
            print(f"- {error}")
        print("=" * 80)
        return 1

    print("All required preflight checks passed.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
