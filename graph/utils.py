"""Shared helpers for graph nodes and chains."""

import os
import time

from graph.consts import env_bool


def chat_context_prefix(chat_history: list, max_answer_chars: int = 400) -> str:
    """Prefix for LLM context so follow-up hints (e.g. 'answer is in section 5') are used.
    Formats the last user/assistant exchange with the assistant reply truncated to max_answer_chars.
    """
    if not chat_history:
        return ""
    last_q, last_a = chat_history[-1]
    truncated = last_a[:max_answer_chars] + (
        "…" if len(last_a) > max_answer_chars else ""
    )
    return f"Previous exchange:\nUser: {last_q}\nAssistant: {truncated}\n\n"


def throttle_llm_if_needed() -> None:
    """Sleep briefly to avoid LLM rate limits.

    - Gemini: when LLM_PROVIDER=gemini, use GEMINI_MIN_DELAY_SEC (e.g. 4–6 for flash-lite free tier ~15 RPM, ~12 for pro ~5 RPM). Applied before every LLM call.
    - Mistral: when WEB_SEARCH_ENABLED is true, use MISTRAL_MIN_DELAY_SEC.
    """
    prov = (os.getenv("LLM_PROVIDER") or "gemini").lower()
    if prov == "gemini":
        raw = (os.getenv("GEMINI_MIN_DELAY_SEC") or "").strip()
        if raw:
            try:
                delay = float(raw)
                if delay > 0:
                    time.sleep(delay)
            except ValueError:
                pass
        return
    if not env_bool("WEB_SEARCH_ENABLED"):
        return
    raw = (os.getenv("MISTRAL_MIN_DELAY_SEC") or "").strip()
    if not raw:
        return
    try:
        delay = float(raw)
    except ValueError:
        return
    if delay <= 0:
        return
    time.sleep(delay)
