# Backend: FastAPI + RAG graph. Chroma DB is empty by default; run ingestion once or mount a pre-built DB.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dependencies come from uv.lock: the exact versions CI tests, and the lock names each
# file's download URL, so the build does not query the package index to resolve them
# (pip install . did, and a slow pypi.org/simple/ response failed the build).
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/usr/local \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

# Dependencies first, so this layer is reused until uv.lock changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY api/ ./api/
COPY graph/ ./graph/
COPY ingestion.py ingestion_fetch.py document_updates.py build_document_sources.py ./

# Empty .chroma so the app starts; docker-compose mounts the index over it
RUN mkdir -p .chroma

# The project itself, not editable, so its version is in the package metadata.
RUN uv sync --frozen --no-dev --no-editable

RUN adduser --disabled-password --gecos "" appuser && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "10"]
