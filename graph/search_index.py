"""The search index: the IAEA and the Danish-law collections of one embedding
model, searched together. Ingestion writes it (ingestion.py); answering and
eval only read it, through this module.
"""

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ContextThreadPoolExecutor

from graph.llm_factory import (
    get_embedding_model_name,
    get_embeddings,
    query_instruction_template,
)

CHROMA_DIR = Path(__file__).resolve().parent.parent / ".chroma"
IAEA_COLLECTION = "radiation-iaea"
DK_LAW_COLLECTION = "radiation-dk-law"
# a provider outside this list is not checked for readiness
_CHECKED_PROVIDERS = ("gemini", "mistral", "ollama", "scaleway")


def collection_names(embedding_provider: str) -> tuple[str, str]:
    """(IAEA collection, Danish-law collection) for an embedding model.

    Scaleway collections carry the model id (SCW_EMBED_MODEL), so several
    Scaleway models can be built side by side and compared. So do local ones
    (OLLAMA_EMBED_MODEL), except nomic-embed-text, the earlier default, whose
    collections keep the name privacy-mode users already have.
    """
    if embedding_provider == "scaleway":
        suffix = f"-scw-{get_embedding_model_name('scaleway')}"
        return (f"{IAEA_COLLECTION}{suffix}", f"{DK_LAW_COLLECTION}{suffix}")
    if embedding_provider == "mistral":
        return (f"{IAEA_COLLECTION}-mistral", f"{DK_LAW_COLLECTION}-mistral")
    if embedding_provider == "ollama":
        model = get_embedding_model_name("ollama").removesuffix(":latest")
        suffix = "-ollama"
        if model != "nomic-embed-text":
            # Chroma allows letters, digits, '.', '_' and '-' in names
            suffix += "-" + re.sub(r"[^A-Za-z0-9._-]", "-", model)
        return (f"{IAEA_COLLECTION}{suffix}", f"{DK_LAW_COLLECTION}{suffix}")
    return (IAEA_COLLECTION, DK_LAW_COLLECTION)


def retriever_k() -> int:
    """Chunks per collection the graph retrieves (RETRIEVER_K, default 5:
    chosen by the k test, ROADMAP step 5)."""
    raw = (os.getenv("RETRIEVER_K") or "").strip()
    k = int(raw) if raw else 5
    if k < 1:
        raise ValueError("RETRIEVER_K must be at least 1")
    return k


@dataclass(frozen=True)
class Readiness:
    """Whether an index can be searched; otherwise why not.

    problem: "not_built" (a collection is missing or empty) or "misconfigured"
    (detail names the setting).
    """

    problem: str = ""
    detail: str = ""

    def __bool__(self) -> bool:
        return not self.problem

    @classmethod
    def ready(cls) -> "Readiness":
        return cls()

    @classmethod
    def not_built(cls) -> "Readiness":
        return cls("not_built")

    @classmethod
    def misconfigured(cls, detail: str) -> "Readiness":
        return cls("misconfigured", detail)


class SearchIndex(Protocol):
    """What the graph needs from an index (ChromaIndex; tests.fakes.FakeIndex)."""

    embedding_provider: str

    def ready(self) -> Readiness: ...

    def search(
        self, query: str, k: int | None = None, config: RunnableConfig | None = None
    ) -> tuple[list[Document], list[Document]]: ...


class ChromaIndex:
    """The two Chroma collections of one embedding model.

    persist_dir: where Chroma stores them (default: the project's .chroma).
    k: chunks per collection a search returns (default: RETRIEVER_K).
    embeddings: the embedding model (default: the provider's, see get_embeddings).
    collections: the two collection names (default: the model's, fixed on first use).
    """

    def __init__(
        self,
        embedding_provider: str,
        *,
        persist_dir: Path | None = None,
        k: int | None = None,
        embeddings=None,
        collections: tuple[str, str] | None = None,
    ):
        self.embedding_provider = embedding_provider
        self.persist_dir = Path(persist_dir or CHROMA_DIR)
        self.k = k or retriever_k()
        self._embeddings = embeddings
        self._collections = collections
        self._stores = None

    @property
    def collections(self) -> tuple[str, str]:
        if self._collections is None:
            self._collections = collection_names(self.embedding_provider)
        return self._collections

    def ready(self) -> Readiness:
        if self.embedding_provider not in _CHECKED_PROVIDERS:
            return Readiness.ready()
        try:
            names = self.collections
        except ValueError as e:  # e.g. Scaleway without SCW_EMBED_MODEL
            return Readiness.misconfigured(str(e))
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.persist_dir))
            for name in names:
                if client.get_collection(name).count() == 0:
                    return Readiness.not_built()
        except Exception:
            return Readiness.not_built()
        return Readiness.ready()

    def search(
        self, query: str, k: int | None = None, config: RunnableConfig | None = None
    ) -> tuple[list[Document], list[Document]]:
        """The best k chunks of each collection (IAEA, Danish law), searched in
        parallel. Raises RuntimeError naming the fix when the collections were
        built by another embedding model."""
        k = k or self.k
        iaea, dk = self._open()

        def search_one(store):
            try:
                return store.as_retriever(search_kwargs={"k": k}).invoke(
                    query, config=config or {}
                )
            except Exception as e:
                if "dimension" in str(e).lower():
                    raise RuntimeError(
                        "Embedding dimension mismatch: the search index was built "
                        "with a different embedding model than the configured one "
                        f"({self.embedding_provider}). Rebuild it with the same "
                        "settings: uv run python ingestion.py (for Scaleway: add "
                        "--reembed-from gemini)."
                    ) from e
                raise

        # the threads carry the caller's context: with tracing switched off for
        # a question (privacy mode, user keys) the search must not trace it either
        with ContextThreadPoolExecutor(max_workers=2) as executor:
            found_iaea = executor.submit(search_one, iaea)
            found_dk = executor.submit(search_one, dk)
            return found_iaea.result(), found_dk.result()

    def chunk_texts(self) -> dict[str, list[str] | None]:
        """The chunk texts of both collections, in storage order; None for a
        collection that does not exist."""
        import chromadb

        client = chromadb.PersistentClient(path=str(self.persist_dir))
        texts: dict[str, list[str] | None] = {}
        for name in self.collections:
            try:
                stored = client.get_collection(name).get(include=["documents"])
            except Exception:
                texts[name] = None
                continue
            texts[name] = [t or "" for t in stored["documents"]]
        return texts

    def fingerprint(self) -> dict[str, dict | None]:
        """Per collection, the number of chunks and a hash of their texts.

        Independent of chunk ids and storage order, so the same chunks embedded
        by another model (reembed_from) fingerprint the same, and any
        re-chunking or re-ingestion that changes a chunk shows up.
        """
        fingerprint: dict[str, dict | None] = {}
        for name, texts in self.chunk_texts().items():
            if texts is None:
                fingerprint[name] = None
                continue
            digests = sorted(
                hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts
            )
            content = hashlib.sha256("".join(digests).encode("ascii")).hexdigest()
            fingerprint[name] = {"chunks": len(texts), "content_hash": content[:12]}
        return fingerprint

    def _open(self):
        if self._stores is None:
            from langchain_chroma import Chroma

            embeddings = self._embeddings or get_embeddings(self.embedding_provider)
            self._stores = tuple(
                Chroma(
                    collection_name=name,
                    embedding_function=embeddings,
                    persist_directory=str(self.persist_dir),
                )
                for name in self.collections
            )
        return self._stores


_open_indexes: dict[tuple, ChromaIndex] = {}


def index_for(embedding_provider: str) -> ChromaIndex:
    """The project's index for an embedding model, opened once.

    Keyed by the collections and the query instruction too, so two Scaleway
    models (or the instruction switched off) never share an opened index.
    """
    names = collection_names(embedding_provider)
    template = (
        query_instruction_template(get_embedding_model_name(embedding_provider))
        if embedding_provider == "scaleway"
        else None
    )
    key = (embedding_provider, names, template)
    if key not in _open_indexes:
        _open_indexes[key] = ChromaIndex(embedding_provider, collections=names)
    return _open_indexes[key]


def refresh() -> None:
    """Open every index again on next use, e.g. after ingestion replaced its collections."""
    _open_indexes.clear()


def load_chunk_texts(embedding_provider: str) -> dict[str, list[str] | None]:
    return ChromaIndex(embedding_provider).chunk_texts()


def index_fingerprint(embedding_provider: str) -> dict[str, dict | None]:
    return ChromaIndex(embedding_provider).fingerprint()
