"""Per-client rate limiting utilities with in-memory and optional Redis backend."""

import os
import time
from typing import Any

from fastapi import HTTPException, Request

try:
    import redis
except ImportError:  # pragma: no cover - optional dependency resolution
    redis = None


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
    backend = (os.getenv("RATE_LIMIT_BACKEND") or "in_memory").strip().lower()
    if backend == "redis":
        if _enforce_rate_limit_redis(
            request,
            bucket=bucket,
            max_requests=max_requests,
            window_seconds=window_seconds,
            app_state=app_state,
        ):
            return

    # Default and fallback backend: process-local in-memory limiter.
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
            headers={"Retry-After": str(retry_after)},
        )


def _enforce_rate_limit_redis(
    request: Request,
    *,
    bucket: str,
    max_requests: int,
    window_seconds: float,
    app_state: dict[str, Any],
) -> bool:
    """Return True when Redis backend was used, False if fallback is required."""
    redis_url = (os.getenv("RATE_LIMIT_REDIS_URL") or "").strip()
    if not redis_url or redis is None:
        return False

    try:
        client = app_state.get("rate_limit_redis_client")
        if client is None:
            client = redis.Redis.from_url(redis_url, decode_responses=True)
            app_state["rate_limit_redis_client"] = client

        now = time.time()
        window = max(1, int(window_seconds))
        window_slot = int(now // window)
        key = f"rl:{bucket}:{request_client_key(request)}:{window_slot}"
        count = int(client.incr(key))
        if count == 1:
            client.expire(key, window + 1)
        if count > max_requests:
            retry_after = max(1, window - int(now % window))
            raise HTTPException(
                status_code=429,
                detail=(
                    "Rate limit exceeded for this endpoint. "
                    f"Please retry after about {retry_after} seconds."
                ),
                headers={"Retry-After": str(retry_after)},
            )
        return True
    except HTTPException:
        raise
    except Exception:
        # Fail open to in-memory limiter if Redis backend is unavailable.
        return False
