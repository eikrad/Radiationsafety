# Backend: FastAPI + RAG graph. Chroma DB is empty by default; run ingestion once or mount a pre-built DB.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install dependencies from pyproject.toml (no uv in image)
COPY pyproject.toml ./
COPY api/ ./api/
COPY graph/ ./graph/
COPY ingestion.py ingestion_fetch.py document_updates.py build_document_sources.py ./

# Empty .chroma so the app starts; persist via volume and run ingestion or mount pre-built data
RUN mkdir -p .chroma

RUN pip install --no-cache-dir .

RUN adduser --disabled-password --gecos "" appuser && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "10"]
