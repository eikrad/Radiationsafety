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


_GEMINI_MODELS = frozenset({"gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"})
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
    prov = (provider or os.getenv("LLM_PROVIDER", "mistral")).lower()
    if prov not in ALLOWED_PROVIDERS:
        prov = "mistral"

    if prov == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise APIKeyError("Gemini")
        model = (
            model_variant
            if model_variant and model_variant in _GEMINI_MODELS
            else "gemini-2.5-flash-lite"
        )
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
            model_variant if model_variant and model_variant in _OPENAI_MODELS else "gpt-4o-mini"
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


def get_embeddings():
    """Return embeddings based on LLM_PROVIDER env."""
    provider = os.getenv("LLM_PROVIDER", "mistral").lower()
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    else:
        from langchain_mistralai import MistralAIEmbeddings

        return MistralAIEmbeddings(model="mistral-embed")
