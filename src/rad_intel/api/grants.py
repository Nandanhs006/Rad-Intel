"""
Download grants: proof that this browser is the one that ran an analysis.

The public tool writes each report to disk and the interface can fetch it
back by its id. That id was the only thing standing between a stranger and
somebody else's report -- and a report carries the patient reference and the
clinical history whoever ran it typed in. A 122-bit UUID is not guessable,
but it is a capability: it travels in URLs, in browser history, in a shared
link, in a server log. "Nobody will guess it" is not the same as "nobody
else can have it".

So a report is downloadable by the browser that produced it, and by an
administrator. The browser proves it with a cookie holding the ids it has
run, signed with the server secret so nothing can be added to the list by
editing it. The cookie identifies no one: it is a list of opaque ids and a
signature over them, and it opens nothing except reports this same browser
asked for.
"""
from __future__ import annotations

import hashlib
import hmac

from rad_intel.auth.crypto import derive_key

COOKIE_NAME = "radintel_runs"
# Long enough that a report is still downloadable the next day, short enough
# that a shared machine does not accumulate a stranger's history.
COOKIE_MAX_AGE = 24 * 60 * 60
# How many analyses a browser may hold a grant for at once. Twenty is far
# more than a sitting, and it keeps the cookie well inside the 4 KB a browser
# will carry.
MAX_GRANTS = 20

_KEY = derive_key("analysis-download-grants")
_SEPARATOR = "."


def _sign(ids: list[str]) -> str:
    return hmac.new(_KEY, ",".join(ids).encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def read(cookie: str | None) -> list[str]:
    """The ids a cookie legitimately carries. Anything tampered with is empty."""
    if not cookie or _SEPARATOR not in cookie:
        return []
    payload, _, signature = cookie.rpartition(_SEPARATOR)
    ids = [part for part in payload.split(",") if part]
    if not ids or not hmac.compare_digest(signature, _sign(ids)):
        return []
    return ids


def grant(cookie: str | None, analysis_id: str) -> str:
    """Add an id to the list and re-sign it, keeping the most recent ones."""
    ids = [i for i in read(cookie) if i != analysis_id]
    ids.insert(0, analysis_id)
    ids = ids[:MAX_GRANTS]
    return f"{','.join(ids)}{_SEPARATOR}{_sign(ids)}"


def holds(cookie: str | None, analysis_id: str) -> bool:
    return analysis_id in read(cookie)
