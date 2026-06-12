"""FastAPI backend: /query, /health, /metrics, document updates, ingest, and optional frontend serving."""

import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api.rate_limit import enforce_rate_limit, env_float, env_int

load_dotenv()

app_state: dict = {
    "graph": None,
    "ingest_status": "idle",
    "started_at": time.time(),
    "rate_limit_store": {},
    "request_metrics": {
        "total": 0,
        "errors": 0,
        "duration_sum_seconds": 0.0,
        "query_web_search_attempts": 0,
        "query_outcomes_total": {},
        "by_path_total": {},
        "by_path_errors": {},
        "by_status_class_total": {},
    },
}
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FRONTEND_DIST = _PROJECT_ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load graph on startup (skipped when TESTING=true)."""
    app_state["started_at"] = time.time()
    if os.getenv("TESTING", "").lower() not in ("true", "1"):
        from graph.graph import app as graph_app

        app_state["graph"] = graph_app
    yield
    app_state["graph"] = None


app = FastAPI(
    title="Radiation Safety RAG API",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_observability_middleware(request: Request, call_next):
    """Attach request id and capture basic request/error/latency metrics."""
    metrics = app_state.setdefault(
        "request_metrics",
        {
            "total": 0,
            "errors": 0,
            "duration_sum_seconds": 0.0,
            "query_web_search_attempts": 0,
            "query_outcomes_total": {},
            "by_path_total": {},
            "by_path_errors": {},
            "by_status_class_total": {},
        },
    )
    started = time.perf_counter()
    request_id = request.headers.get(_REQUEST_ID_HEADER) or str(uuid.uuid4())
    response = None
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed = max(0.0, time.perf_counter() - started)
        path = request.url.path
        metrics["total"] = int(metrics.get("total", 0)) + 1
        metrics["duration_sum_seconds"] = (
            float(metrics.get("duration_sum_seconds", 0.0)) + elapsed
        )
        by_path_total = metrics.setdefault("by_path_total", {})
        by_path_total[path] = int(by_path_total.get(path, 0)) + 1
        status_class = f"{status_code // 100}xx"
        by_status_class_total = metrics.setdefault("by_status_class_total", {})
        by_status_class_total[status_class] = (
            int(by_status_class_total.get(status_class, 0)) + 1
        )
        if status_code >= 400:
            metrics["errors"] = int(metrics.get("errors", 0)) + 1
            by_path_errors = metrics.setdefault("by_path_errors", {})
            by_path_errors[path] = int(by_path_errors.get(path, 0)) + 1
        if response is not None:
            response.headers[_REQUEST_ID_HEADER] = request_id


class QueryRequest(BaseModel):
    """Request body for /query."""

    question: str
    chat_history: list[list[str]] | None = None  # [[q,a],[q,a],...] for follow-ups
    model: str | None = None  # "mistral" | "gemini" | "openai"
    model_variant: str | None = None  # e.g. "gemini-2.5-flash-lite", "gpt-4o-mini"
    api_keys: dict[str, str] | None = (
        None  # {"mistral": "...", "gemini": "...", "openai": "..."}
    )


class SourceInfo(BaseModel):
    """Source citation for an answer."""

    source: str
    document_type: str | None = None


class QueryResponse(BaseModel):
    """Response body for /query."""

    answer: str
    sources: list[SourceInfo]
    chat_history: list[list[str]]  # [[q,a],[q,a],...] including new turn
    warning: str | None = (
        None  # When web search or retrieval didn't help (in question's language)
    )
    used_web_search: bool = False  # True if Brave Search was invoked this turn
    used_web_search_label: str | None = (
        None  # Short label in question's language, e.g. "Sources incl. web search"
    )
    privacy_mode: bool = False  # True if Ollama (fully local mode)


class SetSourceUrlBody(BaseModel):
    """Request body for PATCH /documents/source/{source_id}/url."""

    url: str


api_router = APIRouter()
_ADMIN_TOKEN_HEADER = "X-Admin-Token"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_REQUEST_ID_HEADER = "X-Request-ID"


def _admin_bypass_enabled() -> bool:
    return os.getenv("ADMIN_AUTH_BYPASS", "").strip().lower() in _TRUE_VALUES


def require_admin(request: Request) -> None:
    """Protect mutating routes with token-based admin auth (fail-closed by default)."""
    enforce_rate_limit(
        request,
        bucket="admin",
        max_requests=env_int("RATE_LIMIT_ADMIN_MAX_REQUESTS", 20),
        window_seconds=env_float("RATE_LIMIT_ADMIN_WINDOW_SEC", 60.0),
        app_state=app_state,
    )
    configured_token = (os.getenv("ADMIN_TOKEN") or "").strip()
    if not configured_token:
        if _admin_bypass_enabled():
            return
        raise HTTPException(
            status_code=503,
            detail="Admin routes are disabled: ADMIN_TOKEN is not configured.",
        )

    provided = (request.headers.get(_ADMIN_TOKEN_HEADER) or "").strip()
    if not provided or not secrets.compare_digest(provided, configured_token):
        raise HTTPException(
            status_code=401,
            detail=f"Missing or invalid admin token in '{_ADMIN_TOKEN_HEADER}' header.",
        )


def _run_ingest() -> None:
    """Run ingestion and clear retriever cache. Updates app_state ingest_status."""
    app_state["ingest_status"] = "running"
    try:
        import ingestion

        ingestion.ingest()
        ingestion.clear_retrievers_cache()
    finally:
        app_state["ingest_status"] = "idle"


@api_router.get("/health")
def health() -> dict:
    """Health check."""
    return {"status": "ok", "graph_loaded": app_state["graph"] is not None}


@api_router.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    """Prometheus-style metrics for observability."""
    graph_loaded = 1 if app_state.get("graph") is not None else 0
    started_at = app_state.get("started_at") or 0
    uptime_seconds = max(0.0, time.time() - started_at)
    req = app_state.get("request_metrics") or {}
    req_total = int(req.get("total", 0))
    req_errors = int(req.get("errors", 0))
    duration_sum_seconds = float(req.get("duration_sum_seconds", 0.0))
    query_web_search_attempts = int(req.get("query_web_search_attempts", 0))
    query_outcomes_total = req.get("query_outcomes_total", {})
    by_path_total = req.get("by_path_total") or {}
    by_path_errors = req.get("by_path_errors") or {}
    by_status_class_total = req.get("by_status_class_total") or {}

    def _sanitize_label(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    lines = [
        "# HELP radiationsafety_graph_loaded 1 if the RAG graph is loaded, 0 otherwise.",
        "# TYPE radiationsafety_graph_loaded gauge",
        f"radiationsafety_graph_loaded {graph_loaded}",
        "# HELP radiationsafety_uptime_seconds Process uptime in seconds.",
        "# TYPE radiationsafety_uptime_seconds gauge",
        f"radiationsafety_uptime_seconds {uptime_seconds:.2f}",
        "# HELP radiationsafety_http_requests_total Total HTTP requests served.",
        "# TYPE radiationsafety_http_requests_total counter",
        f"radiationsafety_http_requests_total {req_total}",
        "# HELP radiationsafety_http_errors_total Total HTTP responses with status >= 400.",
        "# TYPE radiationsafety_http_errors_total counter",
        f"radiationsafety_http_errors_total {req_errors}",
        "# HELP radiationsafety_http_request_duration_seconds_sum Sum of HTTP request durations in seconds.",
        "# TYPE radiationsafety_http_request_duration_seconds_sum counter",
        f"radiationsafety_http_request_duration_seconds_sum {duration_sum_seconds:.6f}",
        "# HELP radiationsafety_query_web_search_attempts_total Query runs that attempted web search.",
        "# TYPE radiationsafety_query_web_search_attempts_total counter",
        f"radiationsafety_query_web_search_attempts_total {query_web_search_attempts}",
        "# HELP radiationsafety_query_outcomes_total Query outcomes by routing category.",
        "# TYPE radiationsafety_query_outcomes_total counter",
    ]
    if isinstance(query_outcomes_total, dict):
        for outcome, count in sorted(query_outcomes_total.items()):
            lines.append(
                f'radiationsafety_query_outcomes_total{{outcome="{_sanitize_label(str(outcome))}"}} {int(count)}'
            )
    lines.extend(
        [
            "# HELP radiationsafety_http_requests_by_path_total Total HTTP requests by path.",
            "# TYPE radiationsafety_http_requests_by_path_total counter",
        ]
    )
    for path, count in sorted(by_path_total.items()):
        escaped_path = str(path).replace("\\", "\\\\").replace('"', '\\"')
        lines.append(
            f'radiationsafety_http_requests_by_path_total{{path="{escaped_path}"}} {int(count)}'
        )
    lines.extend(
        [
            "# HELP radiationsafety_http_errors_by_path_total Total HTTP errors by path.",
            "# TYPE radiationsafety_http_errors_by_path_total counter",
        ]
    )
    for path, count in sorted(by_path_errors.items()):
        escaped_path = str(path).replace("\\", "\\\\").replace('"', '\\"')
        lines.append(
            f'radiationsafety_http_errors_by_path_total{{path="{escaped_path}"}} {int(count)}'
        )
    lines.extend(
        [
            "# HELP radiationsafety_http_responses_by_status_class_total Total HTTP responses by status class.",
            "# TYPE radiationsafety_http_responses_by_status_class_total counter",
        ]
    )
    for status_class, count in sorted(by_status_class_total.items()):
        escaped_class = str(status_class).replace("\\", "\\\\").replace('"', '\\"')
        lines.append(
            "radiationsafety_http_responses_by_status_class_total"
            f'{{status_class="{escaped_class}"}} {int(count)}'
        )
    return "\n".join(lines) + "\n"


@api_router.get("/config")
def config():
    """Return client-relevant config; e.g. whether server has LLM keys so the client can hide the API-key hint."""
    server_has_llm_key = bool(
        (os.getenv("MISTRAL_API_KEY") or "").strip()
        or (os.getenv("GOOGLE_API_KEY") or "").strip()
        or (os.getenv("OPENAI_API_KEY") or "").strip()
    )
    return {"server_has_llm_key": server_has_llm_key}


@api_router.get("/documents/check-updates")
def documents_check_updates():
    """Return list of registered sources with current/remote version and update availability."""
    try:
        from document_updates import check_updates

        sources = check_updates()
        return {"sources": sources, "recent_iaea": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@api_router.patch("/documents/source/{source_id}/url")
def documents_set_source_url(
    source_id: str, body: SetSourceUrlBody, _admin: None = Depends(require_admin)
):
    """Set or update a document source URL manually. URL must be from retsinformation.dk, sst.dk, or iaea.org."""
    url = (body.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")
    try:
        from document_updates import (
            _allowed_url,
            load_registry_raw,
            update_registry_url,
        )
    except ImportError:
        raise HTTPException(
            status_code=500, detail="document_updates not available"
        ) from None
    if not _allowed_url(url):
        raise HTTPException(
            status_code=400,
            detail="URL must be from retsinformation.dk, sst.dk, or iaea.org.",
        )
    raw = load_registry_raw()
    if not any((s.get("id") or "").strip() == source_id.strip() for s in raw):
        raise HTTPException(status_code=404, detail=f"Source '{source_id}' not found")
    update_registry_url(source_id, url)
    return {"ok": True, "message": "URL updated."}


@api_router.get("/documents/source/{source_id}/file")
def documents_get_source_file(source_id: str):
    """Serve the local PDF for a document source. Returns 404 if source not found or no local file."""
    try:
        from document_updates import _load_registry, get_local_pdf_path
    except ImportError:
        raise HTTPException(
            status_code=500, detail="document_updates not available"
        ) from None
    registry = _load_registry()
    source = next(
        (s for s in registry if (s.id or "").strip() == source_id.strip()), None
    )
    if not source:
        raise HTTPException(status_code=404, detail=f"Source '{source_id}' not found")
    path = get_local_pdf_path(source)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="No local file for this source")
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@api_router.post("/documents/source/{source_id}/lookup-url")
def documents_lookup_source_url(source_id: str, _admin: None = Depends(require_admin)):
    """Try to find the document URL (Danish: sst.dk or retsinformation.dk via Brave/probe; IAEA: search). If found, update registry and return URL."""
    try:
        from document_updates import lookup_source_url, update_registry_url
    except ImportError:
        raise HTTPException(
            status_code=500, detail="document_updates not available"
        ) from None
    url, error = lookup_source_url(source_id)
    if not url:
        raise HTTPException(status_code=404, detail=error or "URL not found")
    update_registry_url(source_id, url)
    return {"url": url, "updated": True}


@api_router.post("/documents/source/{source_id}/download-update")
def documents_download_update(source_id: str, _admin: None = Depends(require_admin)):
    """Download the new version for this source and backup the old one. Requires update_available."""
    try:
        import ingestion

        success, message = ingestion.download_update_for_source(source_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"ok": True, "message": message}


@api_router.post("/documents/sync-danish")
def documents_sync_danish(
    apply_updates: bool = False, _admin: None = Depends(require_admin)
):
    """Run incremental Danish legislation sync via Harvest + ELI graph."""
    try:
        from document_updates import sync_danish_legislation
    except ImportError:
        raise HTTPException(
            status_code=500, detail="document_updates not available"
        ) from None
    try:
        report = sync_danish_legislation(apply_updates=apply_updates)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@api_router.post("/documents/build-from-local")
def documents_build_from_local(_admin: None = Depends(require_admin)):
    """Discover local PDFs, extract versions, optionally confirm URLs, write document_sources.yaml. Returns new sources list."""
    try:
        from build_document_sources import build_sources, write_document_sources_yaml

        sources = build_sources(confirm_urls=True)
        if not sources:
            return {
                "sources": [],
                "message": "No documents discovered in documents/IAEA, IAEA_other, or Bekendtgørelse.",
            }
        write_document_sources_yaml(sources)
        from document_updates import check_updates

        updated = check_updates()
        return {
            "sources": updated,
            "message": f"Wrote {len(sources)} source(s) to document_sources.yaml.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


_MAX_PDF_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
_ALLOWED_ADD_PDF_FOLDERS = frozenset({"IAEA", "IAEA_other"})


@api_router.post("/documents/add-pdf")
async def documents_add_pdf(
    file: UploadFile = File(...),  # noqa: B008
    folder: str = Form("IAEA_other"),
    _admin: None = Depends(require_admin),
):
    """Upload a PDF from retsinformation.dk or IAEA: add to collection, extract title/version, look up URL, append to registry."""
    if folder not in _ALLOWED_ADD_PDF_FOLDERS:
        raise HTTPException(
            status_code=400,
            detail=f"folder must be one of: {', '.join(sorted(_ALLOWED_ADD_PDF_FOLDERS))}",
        )
    if not (file.filename and file.filename.lower().endswith(".pdf")):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    content = await file.read()
    if len(content) > _MAX_PDF_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"PDF must be at most {_MAX_PDF_UPLOAD_BYTES // (1024*1024)} MB.",
        )
    safe_name = (
        "".join(c for c in file.filename if c.isalnum() or c in "._- ").strip()
        or "uploaded.pdf"
    )
    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"
    docs_dir = _PROJECT_ROOT / "documents" / folder
    docs_dir.mkdir(parents=True, exist_ok=True)
    dest = docs_dir / safe_name
    try:
        dest.write_bytes(content)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}") from e
    try:
        import ingestion
        from build_document_sources import (
            _extract_iaea_search_terms,
            _extract_pdf_title_and_version,
            _slug,
        )
        from document_updates import (
            _lookup_iaea_publication_url_multi,
            append_source_to_registry,
        )

        title, version = _extract_pdf_title_and_version(dest)
        display_name = (title or safe_name).strip() or safe_name
        search_terms = _extract_iaea_search_terms(Path(dest))
        url = _lookup_iaea_publication_url_multi(search_terms)
        source_id = (
            f"iaea-{_slug(display_name)}"
            if folder == "IAEA"
            else f"iaea-other-{_slug(display_name)}"
        )
        append_source_to_registry(
            source_id=source_id,
            name=display_name,
            url=url,
            folder=folder,
            filename_hint=safe_name,
            version=version,
        )
        chunks = ingestion.add_single_pdf_to_collection(
            dest, folder=folder, source_label=display_name
        )
        ingestion.clear_retrievers_cache()
        msg = f"Added PDF to collection ({chunks} chunks)."
        if url:
            msg += f" URL: {url}"
        else:
            msg += (
                " No publication URL found (try Build list from local PDFs to refresh)."
            )
        return {
            "message": msg,
            "chunks_added": chunks,
            "url_found": bool(url),
            "url": url or None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@api_router.post("/ingest")
def ingest_trigger(
    background_tasks: BackgroundTasks, _admin: None = Depends(require_admin)
):
    """Start ingestion in the background. Returns immediately."""
    if app_state["ingest_status"] == "running":
        raise HTTPException(status_code=409, detail="Ingestion already running.")
    background_tasks.add_task(_run_ingest)
    return JSONResponse(
        status_code=202,
        content={
            "status": "accepted",
            "message": "Ingestion started; this may take several minutes.",
        },
    )


@api_router.get("/ingest/status")
def ingest_status():
    """Return current ingestion status: idle or running."""
    return {"status": app_state["ingest_status"]}


def _to_tuples(history: list[list[str]] | None) -> list[tuple[str, str]]:
    """Convert [[q,a],[q,a]] to [(q,a),(q,a)]."""
    if not history:
        return []
    return [(h[0], h[1]) for h in history if len(h) >= 2]


def _to_lists(history: list[tuple[str, str]]) -> list[list[str]]:
    """Convert [(q,a),(q,a)] to [[q,a],[q,a]]."""
    return [[q, a] for q, a in history]


# Input limits (security / DoS prevention)
_MAX_QUESTION_LEN = 10_000
_MAX_CHAT_HISTORY_LEN = 20
_MAX_API_KEY_LEN = 256

# Phrases that do not require RAG retrieval (cost savings, no DB/LLM calls)
_NON_QUESTION_PATTERNS = frozenset(
    {
        "thank you",
        "thanks",
        "danke",
        "merci",
        "thx",
        "ok",
        "okay",
        "bye",
        "goodbye",
        "tschüss",
        "ciao",
        "yes",
        "no",
        "all right",
        "alright",
        "got it",
    }
)


def _is_non_question(text: str) -> bool:
    """Return True if the input looks like a greeting/acknowledgment, not a real question."""
    t = text.strip().lower()
    if len(t) < 3:
        return True
    if t in _NON_QUESTION_PATTERNS:
        return True
    if not any(c.isalnum() for c in t):
        return True
    return False


def _validate_query_request(req: QueryRequest) -> None:
    """Validate request inputs; raise HTTPException on violation."""
    if len(req.question) > _MAX_QUESTION_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Question exceeds maximum length of {_MAX_QUESTION_LEN} characters.",
        )
    if req.chat_history and len(req.chat_history) > _MAX_CHAT_HISTORY_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Chat history exceeds maximum of {_MAX_CHAT_HISTORY_LEN} entries.",
        )
    if req.api_keys:
        for k, v in req.api_keys.items():
            if v and len(str(v)) > _MAX_API_KEY_LEN:
                raise HTTPException(
                    status_code=400,
                    detail=f"API key for {k} exceeds maximum length.",
                )


def _resolve_model_and_key(
    model: str | None,
    api_keys: dict[str, str] | None,
) -> tuple[str, str | None]:
    """Resolve model (whitelist) and api_key for the request. Returns (model, api_key)."""
    from graph.llm_factory import ALLOWED_PROVIDERS

    provider_from_env = (os.getenv("LLM_PROVIDER") or "gemini").strip()
    prov = (model or provider_from_env).lower()
    if prov not in ALLOWED_PROVIDERS:
        prov = "gemini"
    key = None
    if api_keys and isinstance(api_keys, dict):
        key = api_keys.get(prov) or api_keys.get(prov.strip())
    return prov, key if key else None


@api_router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request):
    """Run RAG pipeline and return answer with sources."""
    enforce_rate_limit(
        request,
        bucket="query",
        max_requests=env_int("RATE_LIMIT_QUERY_MAX_REQUESTS", 60),
        window_seconds=env_float("RATE_LIMIT_QUERY_WINDOW_SEC", 60.0),
        app_state=app_state,
    )
    _validate_query_request(req)
    graph = app_state["graph"]
    if not graph:
        return QueryResponse(
            answer="Backend not ready. Please try again shortly.",
            sources=[],
            chat_history=[],
            warning=None,
            used_web_search=False,
            used_web_search_label=None,
            privacy_mode=False,
        )
    chat_history = _to_tuples(req.chat_history)
    model, api_key = _resolve_model_and_key(req.model, req.api_keys)
    model_variant = req.model_variant
    is_ollama = model == "ollama"

    if _is_non_question(req.question):
        answer = "You're welcome! Ask me anything about radiation safety."
        updated_history = chat_history + [(req.question, answer)]
        return QueryResponse(
            answer=answer,
            sources=[],
            chat_history=_to_lists(updated_history),
            warning=None,
            used_web_search=False,
            used_web_search_label=None,
            privacy_mode=is_ollama,
        )
    try:
        from graph.llm_factory import APIKeyError, get_embedding_provider, get_llm

        llm = get_llm(provider=model, api_key=api_key, model_variant=model_variant)
    except APIKeyError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e
    except Exception as e:
        if is_ollama and (
            "Connection" in type(e).__name__ or "connect" in str(e).lower()
        ):
            raise HTTPException(
                status_code=503,
                detail="Ollama is not running. Start it with: ollama serve",
            ) from e
        raise
    embedding_provider = get_embedding_provider(model)
    try:
        invoke_input = {
            "question": req.question,
            "generation": "",
            "web_search": False,
            "documents": [],
            "web_search_attempted": False,
            "chat_history": chat_history,
            "llm": llm,
            "embedding_provider": embedding_provider,
        }
        run_config = {
            "run_name": "RadiationSafetyRAG",
            "tags": ["rag", "radiation-safety"],
        }
        # Privacy guard: disable tracing and web search for Ollama (fully local mode)
        disable_tracing = bool(req.api_keys) or is_ollama
        if is_ollama:
            invoke_input["privacy_mode"] = True
        if disable_tracing:
            import langsmith as ls

            with ls.tracing_context(enabled=False):
                result = graph.invoke(invoke_input, config=run_config)
        else:
            result = graph.invoke(invoke_input, config=run_config)
    except Exception as e:
        err_msg = str(e)
        if (
            "429" in err_msg
            or "RESOURCE_EXHAUSTED" in err_msg
            or "quota" in err_msg.lower()
        ):
            raise HTTPException(
                status_code=429,
                detail="API rate limit exceeded. Please wait about 30 seconds and try again, or switch to Mistral/OpenAI in Settings.",
            ) from e
        if is_ollama and (
            "Connection" in type(e).__name__
            or "connect" in err_msg.lower()
            or "refused" in err_msg.lower()
        ):
            raise HTTPException(
                status_code=503,
                detail="Ollama is not running. Start it with: ollama serve",
            ) from e
        raise HTTPException(
            status_code=500, detail=err_msg or "Internal server error"
        ) from e

    answer = result.get("generation", "")
    docs = result.get("documents", [])
    sources = []
    seen = set()
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source", "retrieved")
        dtype = meta.get("document_type")
        key = (src, dtype)
        if key not in seen:
            seen.add(key)
            sources.append(SourceInfo(source=str(src), document_type=dtype))
    updated_history = result.get("chat_history") or chat_history
    warning = result.get("retrieval_warning")
    used_web_search = result.get("web_search_attempted", False)
    if used_web_search:
        req_metrics = app_state.setdefault("request_metrics", {})
        req_metrics["query_web_search_attempts"] = (
            int(req_metrics.get("query_web_search_attempts", 0)) + 1
        )
    used_web_search_label = None
    if used_web_search:
        from graph.i18n import detect_language, get_label_sources_incl_web

        used_web_search_label = get_label_sources_incl_web(
            detect_language(req.question)
        )
    outcome = str(result.get("routing_outcome") or "").strip()
    if outcome:
        req_metrics = app_state.setdefault("request_metrics", {})
        by_outcome = req_metrics.setdefault("query_outcomes_total", {})
        by_outcome[outcome] = int(by_outcome.get(outcome, 0)) + 1
    return QueryResponse(
        answer=answer,
        sources=sources,
        chat_history=_to_lists(updated_history),
        warning=warning,
        used_web_search=used_web_search,
        used_web_search_label=used_web_search_label,
        privacy_mode=is_ollama,
    )


app.include_router(api_router, prefix="/api")
# Expose same routes at root (e.g. /health, /query) for direct API access
app.include_router(api_router)


def _root_html() -> str:
    """Helpful page when frontend is not built."""
    return """
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Radiation Safety RAG</title></head>
<body style="font-family:sans-serif;max-width:40em;margin:3em auto;line-height:1.5">
  <h1>Radiation Safety RAG</h1>
  <p>This is the API server. To use the UI:</p>
  <ol>
    <li>In a separate terminal: <code>cd frontend && npm run dev</code></li>
    <li>Open <a href="http://localhost:5173">http://localhost:5173</a></li>
  </ol>
  <p>Or build and serve the frontend: <code>cd frontend && npm run build</code>, then restart this server.</p>
  <p><a href="/api/health">API health</a></p>
</body></html>
"""


@app.get("/", response_class=HTMLResponse)
def root():
    """Serve frontend if built, else show instructions."""
    index_html = _FRONTEND_DIST / "index.html"
    if index_html.exists():
        return FileResponse(index_html)
    return HTMLResponse(_root_html())


# Serve static assets and SPA fallback when frontend is built
if _FRONTEND_DIST.exists():
    assets_dir = _FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{path:path}", response_class=HTMLResponse)
    def serve_spa(path: str):
        """SPA fallback: serve index.html for client-side routes."""
        if path.startswith("api") or path == "api":
            raise HTTPException(404)
        index_html = _FRONTEND_DIST / "index.html"
        if index_html.exists():
            return FileResponse(index_html)
        raise HTTPException(404)
