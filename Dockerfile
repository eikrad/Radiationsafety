# Backend: FastAPI + RAG graph. Pre-built .chroma is copied in so the app works without running ingestion.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies from pyproject.toml (no uv in image)
COPY pyproject.toml ./
COPY api/ ./api/
COPY graph/ ./graph/
COPY ingestion.py ingestion_fetch.py document_updates.py build_document_sources.py ./

# Pre-built vector DB (commit .chroma to the repo so this COPY has data)
COPY .chroma/ ./.chroma/

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
