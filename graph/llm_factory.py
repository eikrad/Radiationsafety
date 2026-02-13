"""LLM and embeddings factory based on LLM_PROVIDER env."""

import os

from dotenv import load_dotenv

load_dotenv()


def get_llm():
    """Return chat LLM based on LLM_PROVIDER (gemini or mistral)."""
    provider = os.getenv("LLM_PROVIDER", "mistral").lower()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
        )
    else:
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(temperature=0)


def get_embeddings():
    """Return embeddings based on LLM_PROVIDER env."""
    provider = os.getenv("LLM_PROVIDER", "mistral").lower()
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    else:
        from langchain_mistralai import MistralAIEmbeddings

        return MistralAIEmbeddings(model="mistral-embed")
