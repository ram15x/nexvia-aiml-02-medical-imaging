"""Request guards for a publicly hosted demo.

Everything here is off by default, so local development behaves exactly as before. The hosted
image turns the guards on through environment variables:

NEXVIA_DISABLE_ORTHANC       true  -> every /orthanc/* route answers 404 (no DICOM server is attached)
NEXVIA_RATE_LIMIT_PER_MINUTE  N    -> at most N POST requests per client per minute
NEXVIA_QA_LIMIT_PER_HOUR      N    -> at most N /qa requests per client per hour (each one spends API credit)
NEXVIA_TRUST_PROXY           true  -> identify clients by the last X-Forwarded-For entry (set only behind a proxy)

The limiter keeps its counters in memory, so it is per process and resets on restart. That is
enough to stop casual abuse of a demo; it is not a replacement for an API gateway.
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

from starlette.requests import Request
from starlette.responses import JSONResponse


def _flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _limit(name: str) -> int:
    raw = (os.environ.get(name) or "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        raise ValueError(f"{name} must be a whole number, got {raw!r}") from None


class SlidingWindowLimiter:
    """Allow at most `limit` events per `window` seconds for each key."""

    def __init__(self, limit: int, window: float, clock=time.monotonic) -> None:
        self.limit = limit
        self.window = window
        self._clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = self._clock()
        with self._lock:
            events = self._events[key]
            while events and now - events[0] >= self.window:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(now)
            if len(self._events) > 10_000:  # drop idle clients so memory stays bounded
                for stale in [k for k, v in self._events.items() if not v]:
                    del self._events[stale]
            return True


def client_key(request: Request, trust_proxy: bool) -> str:
    if trust_proxy:
        forwarded = request.headers.get("x-forwarded-for", "")
        parts = [part.strip() for part in forwarded.split(",") if part.strip()]
        if parts:
            return parts[-1]
    return request.client.host if request.client else "unknown"


def install_guards(app) -> None:
    """Attach the hosted-demo guards to a FastAPI app according to the environment."""
    disable_orthanc = _flag("NEXVIA_DISABLE_ORTHANC")
    trust_proxy = _flag("NEXVIA_TRUST_PROXY")
    post_limiter = SlidingWindowLimiter(_limit("NEXVIA_RATE_LIMIT_PER_MINUTE"), 60.0)
    qa_limiter = SlidingWindowLimiter(_limit("NEXVIA_QA_LIMIT_PER_HOUR"), 3600.0)
    app.state.post_limiter = post_limiter
    app.state.qa_limiter = qa_limiter

    @app.middleware("http")
    async def hosted_demo_guards(request: Request, call_next):
        path = request.url.path
        if disable_orthanc and (path == "/orthanc" or path.startswith("/orthanc/")):
            return JSONResponse({"detail": "Orthanc is not connected in this deployment."}, status_code=404)
        if request.method == "POST":
            key = client_key(request, trust_proxy)
            if path == "/qa" and not qa_limiter.allow(key):
                return JSONResponse(
                    {"detail": "Question limit reached for now. Try again later."},
                    status_code=429, headers={"Retry-After": "3600"},
                )
            if not post_limiter.allow(key):
                return JSONResponse(
                    {"detail": "Too many requests. Wait a minute and try again."},
                    status_code=429, headers={"Retry-After": "60"},
                )
        return await call_next(request)
