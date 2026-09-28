"""LLM and embeddings factory based on LLM_PROVIDER env."""

import json
import os
import re

ALLOWED_PROVIDERS = frozenset({"mistral", "gemini", "openai", "ollama", "scaleway"})

# Answers and embeddings when LLM_PROVIDER / EMBEDDING_PROVIDER are unset: EU-hosted,
# and it retrieved best on the golden set (eval/README.md). Model ids come from .env.
DEFAULT_PROVIDER = "scaleway"


class APIKeyError(Exception):
    """Raised when a valid API key is required but not provided."""

    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(
            f"Please provide a valid API key for {provider} in Settings. "
            "The key is stored only locally and used only for LLM requests."
        )


class ProviderConfigError(ValueError):
    """Raised when the server lacks configuration a provider needs (e.g. SCW_MODEL).

    Unlike APIKeyError, the user cannot fix this in Settings; the operator must.
    """


_GEMINI_MODELS = frozenset(
    {"gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"}
)
_OPENAI_MODELS = frozenset({"gpt-4o-mini", "gpt-4o"})

_SCALEWAY_BASE_URL = "https://api.scaleway.ai/v1"

_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def _parse_text_reply(content: str, schema) -> object:
    """The JSON object in a plain-text reply, validated against schema."""
    match = _JSON_OBJECT.search(content or "")
    if not match:
        raise ValueError(f"no {getattr(schema, '__name__', 'JSON')} object in reply")
    data = json.loads(match.group(0))
    return schema.model_validate(data) if hasattr(schema, "model_validate") else data


def with_text_fallback(structured_with_raw, schema, include_raw: bool):
    """Wrap a structured-output runnable built with include_raw=True.

    Some models ignore a forced tool call and write the JSON as text instead;
    that reply is parsed from the text. A reply that yields nothing raises
    instead of returning None, so callers never act on a missing verdict.
    """
    from langchain_core.runnables import RunnableLambda

    def resolve(result: dict):
        if result["parsed"] is None and result["parsing_error"] is None:
            try:
                parsed = _parse_text_reply(result["raw"].content, schema)
                result = {**result, "parsed": parsed}
            except ValueError as e:  # json and pydantic errors are ValueErrors
                result = {**result, "parsing_error": e}
        if include_raw:
            return result
        if result["parsing_error"] is not None:
            raise result["parsing_error"]
        return result["parsed"]

    return structured_with_raw | RunnableLambda(resolve)


# The OpenAI client alone waits up to 600 s per attempt and retries twice, so a
# stalled provider call could hold a /query open for half an hour.
_DEFAULT_REQUEST_TIMEOUT_SEC = 60.0
_DEFAULT_OLLAMA_TIMEOUT_SEC = 300.0  # local models on a CPU answer slowly
_DEFAULT_MAX_RETRIES = 1


def _env_number(name: str, default: float) -> float:
    try:
        value = float((os.getenv(name) or "").strip())
    except ValueError:
        return default
    return value if value >= 0 else default


def request_timeout() -> float:
    """Seconds a single cloud LLM or embedding call may take (LLM_REQUEST_TIMEOUT_SEC)."""
    return _env_number("LLM_REQUEST_TIMEOUT_SEC", _DEFAULT_REQUEST_TIMEOUT_SEC)


def max_retries() -> int:
    """Retries after a failed or timed-out call (LLM_MAX_RETRIES)."""
    return int(_env_number("LLM_MAX_RETRIES", _DEFAULT_MAX_RETRIES))


def scaleway_chat(model: str, api_key: str | None = None) -> "object":
    """Chat model on Scaleway Generative APIs (OpenAI-compatible, hosted in the EU).

    No allow-list: for internal callers such as the eval judge. Requests coming
    from API clients go through get_llm, which restricts the model choice.
    """
    from langchain_openai import ChatOpenAI

    key = api_key or os.getenv("SCW_SECRET_KEY")
    if not key:
        raise APIKeyError("Scaleway")
    base_url = (os.getenv("SCW_BASE_URL") or "").strip() or _SCALEWAY_BASE_URL

    class ScalewayChat(ChatOpenAI):
        # json_schema-constrained decoding can loop until the token limit on
        # Scaleway (seen with glm-5.2); tool calling returns the same schema reliably.
        def with_structured_output(
            self, schema=None, *, method="function_calling", include_raw=False, **kwargs
        ):
            structured = super().with_structured_output(
                schema, method=method, include_raw=True, **kwargs
            )
            return with_text_fallback(structured, schema, include_raw)

    return ScalewayChat(
        model=model,
        temperature=0,
        api_key=key,
        base_url=base_url,
        timeout=request_timeout(),
        max_retries=max_retries(),
    )


def _scaleway_model(model_variant: str | None) -> str:
    """SCW_MODEL, or a client-requested variant only if listed in SCW_ALLOWED_MODELS."""
    default = (os.getenv("SCW_MODEL") or "").strip()
    allowed = {
        m.strip()
        for m in (os.getenv("SCW_ALLOWED_MODELS") or "").split(",")
        if m.strip()
    }
    if model_variant and model_variant in allowed:
        return model_variant
    if not default:
        raise ProviderConfigError(
            "Scaleway has no answer model on this server: set SCW_MODEL in the "
            "server's .env to a Scaleway model id "
            "(list them with GET https://api.scaleway.ai/v1/models)"
        )
    return default


def get_llm(
    provider: str | None = None,
    api_key: str | None = None,
    model_variant: str | None = None,
) -> "object":  # BaseChatModel
    """Return chat LLM based on provider and optional api_key/model override.

    Args:
        provider: One of 'mistral', 'gemini', 'openai', 'ollama', 'scaleway'. If None, uses
            LLM_PROVIDER, else DEFAULT_PROVIDER.
        api_key: Override API key. If None, falls back to env (MISTRAL_API_KEY, etc.).
        model_variant: Specific model ID (e.g. gemini-2.5-flash-lite, gpt-4o-mini).

    Returns:
        LangChain chat model instance.

    Raises:
        APIKeyError: When provider requires an API key but none is available.
    """
    prov = (provider or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
    if prov not in ALLOWED_PROVIDERS:
        prov = DEFAULT_PROVIDER

    if prov == "ollama":
        from langchain_ollama import ChatOllama

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        env_model = (os.getenv("OLLAMA_MODEL") or "").strip()
        model = model_variant or env_model or "llama3.1:8b"
        timeout = _env_number("OLLAMA_REQUEST_TIMEOUT_SEC", _DEFAULT_OLLAMA_TIMEOUT_SEC)
        return ChatOllama(
            model=model,
            temperature=0,
            base_url=base_url,
            client_kwargs={"timeout": timeout},
        )

    if prov == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise APIKeyError("Gemini")
        # Default: 2.5 Pro. Override via model_variant or GEMINI_MODEL (e.g. gemini-2.5-flash-lite for free tier).
        env_model = (os.getenv("GEMINI_MODEL") or "").strip()
        if model_variant and model_variant in _GEMINI_MODELS:
            model = model_variant
        elif env_model and env_model in _GEMINI_MODELS:
            model = env_model
        else:
            model = "gemini-2.5-pro"
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            google_api_key=key,
            timeout=request_timeout(),
            max_retries=max_retries(),
        )
    elif prov == "scaleway":
        return scaleway_chat(_scaleway_model(model_variant), api_key=api_key)
    elif prov == "openai":
        from langchain_openai import ChatOpenAI

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise APIKeyError("OpenAI")
        model = (
            model_variant
            if model_variant and model_variant in _OPENAI_MODELS
            else "gpt-4o-mini"
        )
        return ChatOpenAI(
            model=model,
            temperature=0,
            api_key=key,
            timeout=request_timeout(),
            max_retries=max_retries(),
        )
    else:
        from langchain_mistralai import ChatMistralAI

        key = api_key or os.getenv("MISTRAL_API_KEY")
        if not key:
            raise APIKeyError("Mistral")
        return ChatMistralAI(
            temperature=0,
            api_key=key,
            timeout=int(request_timeout()),
            max_retries=max_retries(),
        )


EMBEDDING_PROVIDERS = ("gemini", "ollama", "scaleway")


def get_embedding_provider(llm_provider: str | None = None) -> str:
    """Return which embedding backend to use for retrieval.

    Ollama (privacy mode) always embeds locally. Otherwise EMBEDDING_PROVIDER
    picks the backend independently of the answering model; unset, Scaleway. Each backend has its own Chroma
    collections (ingestion.get_collection_names).
    """
    prov = (llm_provider or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
    if prov == "ollama":
        return "ollama"
    configured = (os.getenv("EMBEDDING_PROVIDER") or "").strip().lower()
    if not configured:
        return DEFAULT_PROVIDER
    if configured not in EMBEDDING_PROVIDERS:
        raise ProviderConfigError(
            f"EMBEDDING_PROVIDER={configured!r} is not supported; "
            f"use one of {', '.join(EMBEDDING_PROVIDERS)}"
        )
    return configured


_KNOWN_EMBEDDING_PROVIDERS = ("gemini", "mistral", "ollama", "scaleway")

# Query-side instructions for instruction-aware embedding models; documents are
# embedded without one. Written in English as the model cards advise, also for
# Danish text (Qwen3 Embedding, Zhang et al. 2025: ~1-5 % retrieval lost without).
_QUERY_TASK = (
    "Given a question about radiation protection, retrieve passages from IAEA "
    "safety standards and Danish regulations that answer it"
)
_QUERY_TEMPLATES = {
    "qwen3-embedding": "Instruct: {task}\nQuery:{query}",
    "bge-multilingual-gemma2": "<instruct>{task}\n<query>{query}",
}


def get_embedding_model_name(embedding_provider: str | None = None) -> str:
    """Model id used for embeddings by the given provider (see get_embeddings)."""
    ep = (
        embedding_provider
        if embedding_provider in _KNOWN_EMBEDDING_PROVIDERS
        else get_embedding_provider()
    )
    if ep == "ollama":
        return (os.getenv("OLLAMA_EMBED_MODEL") or "").strip() or "nomic-embed-text"
    if ep == "gemini":
        return "models/gemini-embedding-001"
    if ep == "scaleway":
        model = (os.getenv("SCW_EMBED_MODEL") or "").strip()
        if not model:
            raise ProviderConfigError(
                "Set SCW_EMBED_MODEL to a Scaleway embedding model id "
                "(list them with GET https://api.scaleway.ai/v1/models)"
            )
        return model
    return "mistral-embed"


def query_instruction_template(model: str) -> str | None:
    """The query format for an instruction-aware model, or None (plain queries).

    EMBED_QUERY_INSTRUCTION=false turns it off, e.g. to measure its effect.
    """
    if (os.getenv("EMBED_QUERY_INSTRUCTION") or "").strip().lower() in (
        "0",
        "false",
        "no",
    ):
        return None
    for prefix, template in _QUERY_TEMPLATES.items():
        if model.startswith(prefix):
            return template
    return None


def _with_query_instruction(embeddings, template: str | None):
    """Embeddings that prefix questions (not documents) with the model's instruction."""
    if template is None:
        return embeddings
    from langchain_core.embeddings import Embeddings

    class QueryInstructionEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return embeddings.embed_documents(texts)

        def embed_query(self, text: str) -> list[float]:
            return embeddings.embed_query(template.format(task=_QUERY_TASK, query=text))

    return QueryInstructionEmbeddings()


def get_embeddings(embedding_provider: str | None = None):
    """Return embeddings instance for the given provider.

    Args:
        embedding_provider: 'gemini' | 'mistral' | 'ollama' | 'scaleway'. If None,
            uses get_embedding_provider().
    """
    ep = (
        embedding_provider
        if embedding_provider in _KNOWN_EMBEDDING_PROVIDERS
        else get_embedding_provider()
    )
    model = get_embedding_model_name(ep)
    if ep == "ollama":
        from langchain_ollama import OllamaEmbeddings

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaEmbeddings(model=model, base_url=base_url)
    if ep == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=model)
    if ep == "scaleway":
        from langchain_openai import OpenAIEmbeddings

        key = os.getenv("SCW_SECRET_KEY")
        if not key:
            raise APIKeyError("Scaleway")
        base_url = (os.getenv("SCW_BASE_URL") or "").strip() or _SCALEWAY_BASE_URL
        # raw text, not tiktoken ids: only OpenAI's own API accepts token ids
        plain = OpenAIEmbeddings(
            model=model,
            api_key=key,
            base_url=base_url,
            check_embedding_ctx_length=False,
            timeout=request_timeout(),
            max_retries=max_retries(),
        )
        return _with_query_instruction(plain, query_instruction_template(model))
    from langchain_mistralai import MistralAIEmbeddings

    return MistralAIEmbeddings(model=model)
