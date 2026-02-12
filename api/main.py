"""FastAPI backend: /query and /health endpoints."""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

app_state = {"graph": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load graph on startup."""
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


class SourceInfo(BaseModel):
    """Source citation for an answer."""

    source: str
    document_type: str | None = None


class QueryResponse(BaseModel):
    """Response body for /query."""

    answer: str
    sources: list[SourceInfo]


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok", "graph_loaded": app_state["graph"] is not None}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Run RAG pipeline and return answer with sources."""
    graph = app_state["graph"]
    if not graph:
        return QueryResponse(
            answer="Backend not ready. Please try again shortly.",
            sources=[],
        )
    result = graph.invoke(
        {
            "question": req.question,
            "generation": "",
            "web_search": False,
            "documents": [],
            "web_search_attempted": False,
        }
    )
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
    return QueryResponse(answer=answer, sources=sources)
