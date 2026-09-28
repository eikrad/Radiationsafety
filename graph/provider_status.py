"""Whether each LLM provider can answer on this server.

The UI shows this before a question is asked, so a missing model id or an
unbuilt search index is explained up front instead of failing the query.
"""

import os

from graph.llm_factory import ALLOWED_PROVIDERS, get_embedding_provider

# Server-side key per answering provider. A missing one is not an issue by
# itself: the browser may send its own key.
_ANSWER_KEY_ENV = {
    "scaleway": "SCW_SECRET_KEY",
    "mistral": "MISTRAL_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# Embeddings only ever use the server's key, never one from the browser.
_EMBEDDING_KEY_ENV = {"gemini": "GOOGLE_API_KEY", "scaleway": "SCW_SECRET_KEY"}

_EMBEDDING_LABELS = {"gemini": "Gemini", "scaleway": "Scaleway", "ollama": "Ollama"}


def _is_set(name: str) -> bool:
    return bool((os.getenv(name) or "").strip())


def provider_issue(provider: str) -> str | None:
    """The server configuration that stops `provider` from answering, or None."""
    if provider == "scaleway" and not _is_set("SCW_MODEL"):
        return "Scaleway has no answer model on this server: set SCW_MODEL."
    try:
        embedding_provider = get_embedding_provider(provider)
    except ValueError as e:
        return str(e)
    key_env = _EMBEDDING_KEY_ENV.get(embedding_provider)
    if key_env and not _is_set(key_env):
        label = _EMBEDDING_LABELS.get(embedding_provider, embedding_provider)
        return f"Document search uses {label} embeddings, which need {key_env} on the server."
    if embedding_provider == "scaleway" and not _is_set("SCW_EMBED_MODEL"):
        return (
            "Document search uses Scaleway embeddings (EMBEDDING_PROVIDER is unset or "
            "scaleway), which need SCW_EMBED_MODEL on the server. To use Gemini "
            "embeddings instead, set EMBEDDING_PROVIDER=gemini."
        )
    from ingestion import check_embedding_collections_ready

    ready, message = check_embedding_collections_ready(embedding_provider)
    return None if ready else message


def providers_status() -> dict[str, dict[str, object]]:
    """Per provider: whether the server holds its answer key, and any blocking issue."""
    return {
        provider: {
            "server_key": _is_set(_ANSWER_KEY_ENV.get(provider, "")),
            "issue": provider_issue(provider),
        }
        for provider in sorted(ALLOWED_PROVIDERS)
    }
