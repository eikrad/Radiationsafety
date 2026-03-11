# Backend: FastAPI + RAG graph. Chroma DB is empty by default; run ingestion once or mount a pre-built DB.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies from pyproject.toml (no uv in image)
COPY pyproject.toml ./
COPY api/ ./api/
COPY graph/ ./graph/
COPY ingestion.py ingestion_fetch.py document_updates.py build_document_sources.py ./

# Empty .chroma so the app starts; persist via volume and run ingestion or mount pre-built data
RUN mkdir -p .chroma

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
