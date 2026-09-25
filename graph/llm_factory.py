"""LLM and embeddings factory based on LLM_PROVIDER env."""

import os

ALLOWED_PROVIDERS = frozenset({"mistral", "gemini", "openai", "ollama", "scaleway"})


class APIKeyError(Exception):
    """Raised when a valid API key is required but not provided."""

    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(
            f"Please provide a valid API key for {provider} in Settings. "
            "The key is stored only locally and used only for LLM requests."
        )


_GEMINI_MODELS = frozenset(
    {"gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"}
)
_OPENAI_MODELS = frozenset({"gpt-4o-mini", "gpt-4o"})

_SCALEWAY_BASE_URL = "https://api.scaleway.ai/v1"


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
    return ChatOpenAI(model=model, temperature=0, api_key=key, base_url=base_url)


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
        raise ValueError(
            "Set SCW_MODEL in .env to a Scaleway model id "
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
        provider: One of 'mistral', 'gemini', 'openai', 'ollama', 'scaleway'. If None, uses LLM_PROVIDER env.
        api_key: Override API key. If None, falls back to env (MISTRAL_API_KEY, etc.).
        model_variant: Specific model ID (e.g. gemini-2.5-flash-lite, gpt-4o-mini).

    Returns:
        LangChain chat model instance.

    Raises:
        APIKeyError: When provider requires an API key but none is available.
    """
    prov = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
    if prov not in ALLOWED_PROVIDERS:
        prov = "gemini"

    if prov == "ollama":
        from langchain_ollama import ChatOllama

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        env_model = (os.getenv("OLLAMA_MODEL") or "").strip()
        model = model_variant or env_model or "llama3.1:8b"
        return ChatOllama(model=model, temperature=0, base_url=base_url)

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
        )
    else:
        from langchain_mistralai import ChatMistralAI

        key = api_key or os.getenv("MISTRAL_API_KEY")
        if not key:
            raise APIKeyError("Mistral")
        return ChatMistralAI(temperature=0, api_key=key)


def get_embedding_provider(llm_provider: str | None = None) -> str:
    """Return which embedding backend to use for retrieval.

    Cloud providers (gemini, openai, mistral) share Gemini embeddings.
    Ollama uses local embeddings (separate Chroma collections).
    """
    prov = (llm_provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
    if prov == "ollama":
        return "ollama"
    return "gemini"


def get_embedding_model_name(embedding_provider: str | None = None) -> str:
    """Model id used for embeddings by the given provider (see get_embeddings)."""
    ep = (
        embedding_provider
        if embedding_provider in ("gemini", "mistral", "ollama")
        else get_embedding_provider()
    )
    if ep == "ollama":
        return (os.getenv("OLLAMA_EMBED_MODEL") or "").strip() or "nomic-embed-text"
    if ep == "gemini":
        return "models/gemini-embedding-001"
    return "mistral-embed"


def get_embeddings(embedding_provider: str | None = None):
    """Return embeddings instance for the given provider.

    Args:
        embedding_provider: 'gemini' | 'mistral' | 'ollama'. If None, uses get_embedding_provider().
    """
    ep = (
        embedding_provider
        if embedding_provider in ("gemini", "mistral", "ollama")
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
    from langchain_mistralai import MistralAIEmbeddings

    return MistralAIEmbeddings(model=model)
