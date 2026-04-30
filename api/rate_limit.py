"""In-memory per-client rate limiting utilities."""

import os
import time
from typing import Any

from fastapi import HTTPException, Request


def env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def request_client_key(request: Request) -> str:
    xff = (request.headers.get("x-forwarded-for") or "").strip()
    if xff:
        return xff.split(",")[0].strip()
    if request.client and request.client.host:
        return str(request.client.host)
    return "unknown"


def enforce_rate_limit(
    request: Request,
    *,
    bucket: str,
    max_requests: int,
    window_seconds: float,
    app_state: dict[str, Any],
) -> None:
    store = app_state.setdefault("rate_limit_store", {})
    now = time.monotonic()
    key = f"{bucket}:{request_client_key(request)}"
    started_at, count = store.get(key, (now, 0))
    if now - started_at >= window_seconds:
        started_at, count = now, 0
    count += 1
    store[key] = (started_at, count)
    if count > max_requests:
        retry_after = max(1, int(window_seconds - (now - started_at)))
        raise HTTPException(
            status_code=429,
            detail=(
                "Rate limit exceeded for this endpoint. "
                f"Please retry after about {retry_after} seconds."
            ),
        )
