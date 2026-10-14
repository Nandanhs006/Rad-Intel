"""
Signed session cookies.

A token is `user_id.expiry.signature`, where the signature is an HMAC-SHA256
over the first two parts keyed by the server secret. Nothing sensitive is
carried in the cookie, and a tampered one fails verification, so a session
cannot be forged without the secret.

The secret comes from RAD_INTEL_SECRET_KEY. When it is unset a random one is
generated at import, which is fine for a local demo and has the deliberate
side effect of invalidating every session on restart. A deployment should set
it so logins survive a restart.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import time

logger = logging.getLogger(__name__)

COOKIE_NAME = "radintel_session"
SESSION_TTL_SECONDS = 12 * 60 * 60  # a working day

_SECRET = os.environ.get("RAD_INTEL_SECRET_KEY", "").strip()
if not _SECRET:
    _SECRET = secrets.token_hex(32)
    logger.warning(
        "RAD_INTEL_SECRET_KEY is not set; using a random per-process secret. "
        "Sessions will not survive a restart. Set it for any real deployment."
    )
_SECRET_BYTES = _SECRET.encode("utf-8")


def _sign(payload: str) -> str:
    return hmac.new(_SECRET_BYTES, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def issue(user_id: str, ttl: int = SESSION_TTL_SECONDS) -> str:
    payload = f"{user_id}.{int(time.time()) + ttl}"
    return f"{payload}.{_sign(payload)}"


def verify(token: str | None) -> str | None:
    """Return the user id for a valid, unexpired token, else None."""
    if not token:
        return None
    parts = token.rsplit(".", 2)
    if len(parts) != 3:
        return None
    user_id, expiry, signature = parts
    if not hmac.compare_digest(signature, _sign(f"{user_id}.{expiry}")):
        return None
    try:
        if int(expiry) < time.time():
            return None
    except ValueError:
        return None
    return user_id
