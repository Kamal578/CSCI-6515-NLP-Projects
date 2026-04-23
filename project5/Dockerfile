FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --prefix=/install -r requirements.txt

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    OLLAMA_LLM_MODEL=llama3.1 \
    OLLAMA_EMBED_MODEL=nomic-embed-text \
    CHROMA_PATH=vector_db \
    CHROMA_COLLECTION=sp500_esg_reports \
    ANONYMIZED_TELEMETRY=FALSE

WORKDIR /app

COPY --from=builder /install /usr/local
COPY . .

RUN mkdir -p data/raw_txt vector_db

EXPOSE 8501

CMD ["streamlit", "run", "src/04_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
