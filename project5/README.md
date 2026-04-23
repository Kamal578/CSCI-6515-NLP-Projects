# ESG RAG System (S&P 500 Sustainability Reports)

End-to-end Retrieval-Augmented Generation (RAG) project using local Ollama models, LlamaIndex orchestration, and ChromaDB persistent vector storage.

Dataset used: https://www.kaggle.com/datasets/jaidityachopra/esg-sustainability-reports-of-s-and-p-500-companies/data

## Stack
- Python 3.11+
- LLM: Ollama `llama3.1`
- Embeddings: Ollama `nomic-embed-text`
- Vector DB: ChromaDB (local persistent)
- Orchestration: LlamaIndex
- UI: Streamlit
- Deployment: Docker + Docker Compose

## Project Structure
```text
project_root/
├── data/
│   ├── preprocessed_content.csv          # local (not committed)
│   ├── raw_txt/                          # generated
│   └── eval_questions.txt                # committed question bank
├── docs/
│   ├── architecture.md
│   ├── assignment_mapping.md
│   ├── dataset_profile.md
│   ├── results_summary.md
│   └── submission_checklist.md
├── outputs/
│   └── eval/                             # generated eval artifacts
├── vector_db/                            # generated local index
├── src/
│   ├── 01_data_prep.py
│   ├── 02_build_index.py
│   ├── 03_rag_pipeline.py
│   ├── 04_app.py
│   ├── 05_eval_metrics.py
│   ├── common.py
│   ├── rag_prompts.py
│   └── preflight.py
├── scripts/
├── tests/
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
└── project_progress.md
```

## Environment Variables
Defaults are built-in but can be overridden:
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_LLM_MODEL` (default: `llama3.1`)
- `OLLAMA_EMBED_MODEL` (default: `nomic-embed-text`)
- `CHROMA_PATH` (default: `vector_db`)
- `CHROMA_COLLECTION` (default: `sp500_esg_reports`)
- `SIMILARITY_TOP_K` (default: `3`)
- `OLLAMA_CONTEXT_WINDOW` (default: `2048`)

## Local Quick Start
```bash
./scripts/setup.sh
./scripts/preflight.sh
./scripts/prepare_data.sh
./scripts/build_index.sh --rebuild
./scripts/evaluate.sh --questions-file data/eval_questions.txt
./scripts/eval_metrics.sh
./scripts/run_app.sh
```

Open Streamlit at `http://localhost:8501`.

## Docker Quick Start
```bash
cp .env.example .env
./scripts/setup.sh
./scripts/prepare_data.sh
./scripts/build_index.sh --rebuild
./scripts/docker_up.sh
```

Docker notes:
- Containerized app uses host Ollama via `DOCKER_OLLAMA_BASE_URL` (default `http://host.docker.internal:11434`).
- Vector index is persisted on host through bind mount `./vector_db:/app/vector_db`.

## Evaluation Commands
Run baseline vs RAG and write machine-readable outputs:
```bash
python src/03_rag_pipeline.py --questions-file data/eval_questions.txt
```

Compute summary metrics from latest evaluation output:
```bash
python src/05_eval_metrics.py
```

Generated artifacts:
- `outputs/eval/rag_eval_*.json`
- `outputs/eval/rag_eval_*.csv`
- `outputs/eval/rag_metrics_*.json`
- `outputs/eval/rag_metrics_*.csv`

## Prompting
- RAG system prompt and synthesis templates are centralized in `src/rag_prompts.py`.
- The prompt is applied to retrieval answer synthesis in:
  - `src/03_rag_pipeline.py`
  - `src/04_app.py`
- Baseline in `src/03_rag_pipeline.py` remains intentionally unconstrained (`LLM only`) for fair baseline-vs-RAG comparison.

## Testing
Install test dependencies and run the full test suite:
```bash
python -m pip install -r requirements-dev.txt
./scripts/test.sh
```

With coverage:
```bash
./scripts/test.sh --cov=src --cov-report=term-missing
```

CI runs the same test command via GitHub Actions (`.github/workflows/tests.yml`).

## Script Reference
- `scripts/setup.sh`: install dependencies and pull Ollama models.
- `scripts/preflight.sh`: validate environment, files, and optional index state.
- `scripts/prepare_data.sh`: split CSV rows into TXT files in `data/raw_txt`.
- `scripts/build_index.sh`: build/rebuild Chroma vector index.
- `scripts/index_status.sh`: inspect index process + current vector count.
- `scripts/evaluate.sh`: baseline vs RAG evaluation.
- `scripts/eval_metrics.sh`: compute metrics from evaluation JSON.
- `scripts/run_app.sh`: run Streamlit app locally.
- `scripts/full_local.sh`: prepare -> build index -> evaluate -> metrics.
- `scripts/docker_up.sh`: build and start Streamlit container.
- `scripts/docker_down.sh`: stop container.
- `scripts/test.sh`: run pytest suite.

## Notes
- `vector_db/` and `outputs/eval/` are generated and git-ignored.
- `data/preprocessed_content.csv` is git-ignored; keep a local copy.
- Chroma telemetry warnings are mitigated by pinned versions (`chromadb==0.5.23`, `posthog<4`) and telemetry-off client settings.
