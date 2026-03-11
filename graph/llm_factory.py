"""LLM and embeddings factory based on LLM_PROVIDER env."""

import os

ALLOWED_PROVIDERS = frozenset({"mistral", "gemini", "openai"})


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


def get_llm(
    provider: str | None = None,
    api_key: str | None = None,
    model_variant: str | None = None,
) -> "object":  # BaseChatModel
    """Return chat LLM based on provider and optional api_key/model override.

    Args:
        provider: One of 'mistral', 'gemini', 'openai'. If None, uses LLM_PROVIDER env.
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

    All providers (gemini, openai, mistral) use Gemini embeddings and the same Chroma collections,
    so the single vector store can be used regardless of which LLM is chosen for generation.
    """
    return "gemini"


def get_embeddings(embedding_provider: str | None = None):
    """Return embeddings. All LLM providers use Gemini embeddings for retrieval (shared vector store).

    Args:
        embedding_provider: 'gemini' | 'mistral'. If None, uses get_embedding_provider() (currently always 'gemini').
    """
    ep = (
        embedding_provider
        if embedding_provider in ("gemini", "mistral")
        else get_embedding_provider()
    )
    if ep == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    from langchain_mistralai import MistralAIEmbeddings

    return MistralAIEmbeddings(model="mistral-embed")
