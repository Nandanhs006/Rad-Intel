"""
Request rate limits.

Three distinct problems, which is why there is not one global limit:

* The public analysis endpoints run a neural network and, for a report, call
  an external language model. Left open they are a way to spend someone
  else's GPU time and API budget.
* The sign-in endpoints are the way in. Without a limit, twelve characters of
  one-time code and an unknown Rad-Intel ID are both guessable by a script,
  and the code-request endpoint is a way to flood a clinician's phone.
* The shared-data endpoints -- submitting a review, writing an export,
  starting a training run -- are cheap individually but collide. The limits
  here are loose enough that no honest reviewer will meet one and tight
  enough that a stuck script cannot fill the store.

Identity is the signed-in account where there is one and the client address
otherwise, so one clinician on a hospital network cannot exhaust the budget
of everybody behind the same address.

Scope: the counters live in this process. That is correct for how the service
runs -- a single Uvicorn worker -- and would need to move to shared storage
before running several. It is stated here rather than discovered later.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Request

from rad_intel.auth.dependencies import current_clinician

_buckets: dict[str, deque[float]] = defaultdict(deque)
_lock = threading.Lock()


def client_address(request: Request) -> str:
    """
    The caller's address.

    X-Forwarded-For is honoured only when a proxy is declared through
    RAD_INTEL_TRUSTED_PROXY, because any client can set that header, and a
    rate limiter that believes it is no limit at all.
    """
    import os

    if os.environ.get("RAD_INTEL_TRUSTED_PROXY", "").strip() in ("1", "true", "yes"):
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check(key: str, max_calls: int, window: float) -> float:
    """Record a call. Returns 0.0 when allowed, else the seconds to wait."""
    now = time.monotonic()
    with _lock:
        hits = _buckets[key]
        while hits and now - hits[0] > window:
            hits.popleft()
        if len(hits) >= max_calls:
            return max(1.0, window - (now - hits[0]))
        hits.append(now)
        return 0.0


def limit(bucket: str, max_calls: int, window_seconds: float, what: str = "requests"):
    """
    Build a dependency that caps `bucket` at max_calls per window.

    Used as `Depends(limit("feedback-submit", 60, 3600, "review submissions"))`.
    """

    def dependency(request: Request, user: dict | None = Depends(current_clinician)) -> None:
        identity = f"user:{user['id']}" if user else f"ip:{client_address(request)}"
        wait = _check(f"{bucket}:{identity}", max_calls, window_seconds)
        if wait:
            minutes = max(1, int(wait // 60))
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit reached: at most {max_calls} {what} per "
                    f"{_describe(window_seconds)}. Try again in about "
                    f"{minutes} minute{'' if minutes == 1 else 's'}."
                ),
                headers={"Retry-After": str(int(wait))},
            )

    return dependency


def limit_by_address(bucket: str, max_calls: int, window_seconds: float,
                     what: str = "requests"):
    """
    As `limit`, but always keyed by client address.

    For the endpoints reached before anybody is signed in, where there is no
    account to key on and the whole point is to slow a caller down.
    """

    def dependency(request: Request) -> None:
        wait = _check(f"{bucket}:ip:{client_address(request)}", max_calls, window_seconds)
        if wait:
            minutes = max(1, int(wait // 60))
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Too many {what} from this address. Try again in about "
                    f"{minutes} minute{'' if minutes == 1 else 's'}."
                ),
                headers={"Retry-After": str(int(wait))},
            )

    return dependency


def _describe(window_seconds: float) -> str:
    if window_seconds >= 3600:
        hours = int(window_seconds // 3600)
        return "hour" if hours == 1 else f"{hours} hours"
    minutes = max(1, int(window_seconds // 60))
    return "minute" if minutes == 1 else f"{minutes} minutes"


def reset() -> None:
    """Clear every counter. For tests."""
    with _lock:
        _buckets.clear()
