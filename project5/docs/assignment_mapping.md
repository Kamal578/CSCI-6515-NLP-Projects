# Assignment Mapping

## Task 1: Data Preparation and Fundamentals

### Domain
- Domain selected: ESG sustainability reporting for S&P 500 companies.

### Dataset requirement
- Source: Kaggle ESG Sustainability Reports of S&P 500 Companies.
- Local dataset profile:
  - Documents: 866
  - Total words: 10,361,942
  - Total tokens (whitespace): 10,361,942
- Requirement check: exceeds 50 documents and 10,000+ words.

### Preprocessing
- Cleaning is assumed upstream in `preprocessed_content.csv`.
- Structured extraction to text files is implemented in `src/01_data_prep.py`.
- Chunking is implemented in `src/02_build_index.py` using strict bounds:
  - Token range per stored chunk: 201 to 499
  - Target: 420

### RAG components
- Knowledge Base: ChromaDB persistent collection (`vector_db/`, collection `sp500_esg_reports`) built from ESG report chunks.
- Retriever: LlamaIndex query engine with similarity search (`top_k`, default 3).
- Generator: Ollama local LLM (`llama3.1`) using retrieved chunks as context.

## Task 2: RAG System Implementation

### Embeddings
- Model: Ollama `nomic-embed-text`.
- Integration: `llama-index-embeddings-ollama` in `src/02_build_index.py` and `src/03_rag_pipeline.py`.

### Vector database
- Backend: ChromaDB local persistent storage.
- Integration: `llama-index-vector-stores-chroma`.
- Persisted state: `vector_db/`.

### Retriever
- Similarity top-k retrieval configured in `src/03_rag_pipeline.py` and `src/04_app.py`.
- Default `top_k=3`, override via `SIMILARITY_TOP_K` or CLI.

### LLM integration
- Generator model: Ollama `llama3.1`.
- Base URL is environment-driven to support local and Docker runtime.

### End-to-end pipeline
- Query flow implemented as:
  - Query -> Retriever -> Context nodes -> LLM answer
- Implementations:
  - Script pipeline: `src/03_rag_pipeline.py`
  - Interactive UI: `src/04_app.py`

### Baseline vs RAG comparison
- Baseline path: direct `llama3.1` answer without retrieval.
- RAG path: retrieval + generation via LlamaIndex query engine.
- Machine-readable outputs:
  - `outputs/eval/rag_eval_*.json`
  - `outputs/eval/rag_eval_*.csv`
- Post-processing metrics:
  - `src/05_eval_metrics.py`
  - output files `outputs/eval/rag_metrics_*.json` and `.csv`

## Reproducibility and Packaging
- Scripted local workflow under `scripts/`.
- Preflight checks in `scripts/preflight.sh`.
- Streamlit UI runner in `scripts/run_app.sh`.
- Containerized runtime: `Dockerfile` + `docker-compose.yml`.
- Automated tests: `pytest` + CI workflow in `.github/workflows/tests.yml`.
