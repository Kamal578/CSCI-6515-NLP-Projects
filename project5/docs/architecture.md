# Architecture

## System Overview
- Data Prep: `src/01_data_prep.py`
- Index Build: `src/02_build_index.py`
- Evaluation Pipeline: `src/03_rag_pipeline.py`
- Metrics Aggregation: `src/05_eval_metrics.py`
- UI: `src/04_app.py`

## Runtime Configuration
All components share env-driven defaults via `src/common.py`:
- `OLLAMA_BASE_URL`
- `OLLAMA_LLM_MODEL`
- `OLLAMA_EMBED_MODEL`
- `CHROMA_PATH`
- `CHROMA_COLLECTION`

## Data Flow
1. CSV rows are split into per-report text files (`data/raw_txt`).
2. Text documents are tokenized and chunked under strict token constraints.
3. Chunk embeddings are generated with local Ollama embeddings.
4. Chunks + metadata are stored in Chroma persistent collection.
5. At query time, top-k relevant chunks are retrieved.
6. Retrieved context is passed to local Ollama LLM for answer generation.
7. Response plus source chunks are exposed in scripts and UI.

## Metadata Strategy
Each indexed chunk stores:
- `source_file`
- `ticker`
- `year`
- `chunk_id`
- `token_count`

This supports transparent citation in console outputs and Streamlit.

## Deployment Modes
- Local scripts for deterministic dev workflow.
- Docker Compose for reproducible app runtime with host Ollama access.
