"""Node and chain identifiers for the graph."""

import os

RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
RETRIEVE_MISSING = "retrieve_missing"
WEB_SEARCH = "web_search"
GENERATE = "generate"
VERIFY_TRUSTED = "verify_trusted"
FINALIZE = "finalize"


def env_bool(name: str, default: bool = False) -> bool:
    """Return True if env var is set to 'true' or '1' (case-insensitive), else default."""
    raw = os.getenv(name, "true" if default else "false")
    return (raw or "").strip().lower() in ("true", "1")
