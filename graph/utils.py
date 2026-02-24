"""Shared helpers for graph nodes and chains."""


def chat_context_prefix(chat_history: list, max_answer_chars: int = 400) -> str:
    """Prefix for LLM context so follow-up hints (e.g. 'answer is in section 5') are used.
    Formats the last user/assistant exchange with the assistant reply truncated to max_answer_chars."""
    if not chat_history:
        return ""
    last_q, last_a = chat_history[-1]
    truncated = last_a[:max_answer_chars] + ("…" if len(last_a) > max_answer_chars else "")
    return f"Previous exchange:\nUser: {last_q}\nAssistant: {truncated}\n\n"
