"""FastAPI backend: /query, /health, and optional frontend serving."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

app_state = {"graph": None}
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FRONTEND_DIST = _PROJECT_ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load graph on startup (skipped when TESTING=true)."""
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
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """Request body for /query."""

    question: str
    chat_history: list[list[str]] | None = None  # [[q,a],[q,a],...] for follow-ups
    model: str | None = None  # "mistral" | "gemini" | "openai"
    model_variant: str | None = None  # e.g. "gemini-2.5-flash-lite", "gpt-4o-mini"
    api_keys: dict[str, str] | None = None  # {"mistral": "...", "gemini": "...", "openai": "..."}


class SourceInfo(BaseModel):
    """Source citation for an answer."""

    source: str
    document_type: str | None = None


class QueryResponse(BaseModel):
    """Response body for /query."""

    answer: str
    sources: list[SourceInfo]
    chat_history: list[list[str]]  # [[q,a],[q,a],...] including new turn
    warning: str | None = None  # When web search or retrieval didn't help


api_router = APIRouter()


@api_router.get("/health")
def health():
    """Health check."""
    return {"status": "ok", "graph_loaded": app_state["graph"] is not None}


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
_NON_QUESTION_PATTERNS = frozenset({
    "thank you", "thanks", "danke", "merci", "thx",
    "ok", "okay", "bye", "goodbye", "tschüss", "ciao",
    "yes", "no", "all right", "alright", "got it",
})


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

    prov = (model or os.getenv("LLM_PROVIDER", "mistral")).lower()
    if prov not in ALLOWED_PROVIDERS:
        prov = "mistral"
    key = None
    if api_keys and isinstance(api_keys, dict):
        key = api_keys.get(prov) or api_keys.get(prov.strip())
    return prov, key if key else None


@api_router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Run RAG pipeline and return answer with sources."""
    _validate_query_request(req)
    graph = app_state["graph"]
    if not graph:
        return QueryResponse(
            answer="Backend not ready. Please try again shortly.",
            sources=[],
            chat_history=[],
            warning=None,
        )
    chat_history = _to_tuples(req.chat_history)
    if _is_non_question(req.question):
        answer = "You're welcome! Ask me anything about radiation safety."
        updated_history = chat_history + [(req.question, answer)]
        return QueryResponse(
            answer=answer,
            sources=[],
            chat_history=_to_lists(updated_history),
            warning=None,
        )
    model, api_key = _resolve_model_and_key(req.model, req.api_keys)
    model_variant = req.model_variant
    try:
        from graph.llm_factory import APIKeyError, get_llm

        llm = get_llm(provider=model, api_key=api_key, model_variant=model_variant)
    except APIKeyError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    try:
        invoke_input = {
            "question": req.question,
            "generation": "",
            "web_search": False,
            "documents": [],
            "web_search_attempted": False,
            "chat_history": chat_history,
            "llm": llm,
        }
        if req.api_keys:
            import langsmith as ls

            with ls.tracing_context(enabled=False):
                result = graph.invoke(invoke_input)
        else:
            result = graph.invoke(invoke_input)
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="API rate limit exceeded. Please wait about 30 seconds and try again, or switch to Mistral/OpenAI in Settings.",
            )
        raise

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
    return QueryResponse(
        answer=answer,
        sources=sources,
        chat_history=_to_lists(updated_history),
        warning=warning,
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
