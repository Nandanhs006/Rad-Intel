"""
Server-side sessions: one device per account, thirty days.

The cookie carries nothing but a random 256-bit token. Everything that
matters -- which account it belongs to, when it was opened, when it was last
used -- lives in a row on the server, and only a hash of the token is stored,
so a copy of the database cannot be used to log in as anybody.

The rules, and what each one costs and buys:

* **One account, one device.** Logging in anywhere displaces the session that
  was open somewhere else. Two clinicians cannot quietly share one Rad-Intel
  ID and work in parallel, and if an ID is passed around, the person it was
  issued to finds themselves logged out -- which is a far better alarm than a
  line in a log nobody reads.
* **Thirty days, measured from the sign-in.** Not a sliding window: a session
  opened on the first expires on the thirty-first however much it was used,
  so a one-time code is needed once a month rather than never again.
* **No idle timeout and no browser-close logout.** The cookie carries an
  expiry, so it survives closing the browser and restarting the machine. That
  is the point -- a clinician reviewing a handful of cases a week should not
  be made to find a code every time -- and it is a deliberate trade: a
  workstation left unlocked stays logged in. The thirty-day ceiling and the
  administrator's ability to close any session are what bound it.
* **Revocation is immediate.** Deleting the row ends the session on the next
  request. A self-contained signed token would stay valid until its own
  expiry whatever the server thought, which is why none is used here.
"""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

from rad_intel.auth import store
from rad_intel.auth.crypto import fast_hash
from rad_intel.config import BASE_DIR

DB_PATH = BASE_DIR / "rad_intel.db"

COOKIE_NAME = "radintel_session"

# Thirty days from the sign-in, after which a fresh one-time code is needed.
SESSION_TTL_SECONDS = 30 * 24 * 60 * 60

# How stale last_seen_at may get before it is written again. It no longer
# governs expiry -- it is only what the administrator's view shows -- and
# without the throttle every request took a write lock on the shared database
# purely to move a timestamp a few seconds, which the parallel requests a page
# load makes were enough to collide over.
TOUCH_INTERVAL_SECONDS = 5 * 60

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    token_hash TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    ip TEXT,
    user_agent TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""


def _connect():
    """A transaction on this module's tables: commits, and always closes."""
    return store.transaction(DB_PATH)


def init() -> None:
    """Build the tables, once per process: see auth/store.py."""
    store.ensure_schema(DB_PATH, "sessions", lambda conn: conn.executescript(SCHEMA))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat()


def _parse(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def active_for_user(user_id: str) -> list[dict]:
    """
    The account's live sessions. Normally none or one.

    Read before issuing, so the caller can tell the person signing in that
    they are about to displace a device they still have open.
    """
    init()
    purge_expired()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, user_id, created_at, last_seen_at, expires_at, ip, user_agent"
            " FROM sessions WHERE user_id = ? ORDER BY last_seen_at DESC;",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def issue(user_id: str, ttl: int = SESSION_TTL_SECONDS, ip: str | None = None,
          user_agent: str | None = None, single_device: bool = True) -> str:
    """
    Open a session and return the opaque token to put in the cookie.

    With `single_device`, every other session this account holds is closed
    first, so the account is signed in on exactly one device at a time.
    """
    init()
    token = secrets.token_urlsafe(32)
    now = _now()
    with _connect() as conn:
        if single_device:
            conn.execute("DELETE FROM sessions WHERE user_id = ?;", (user_id,))
        conn.execute(
            "INSERT INTO sessions (id, token_hash, user_id, created_at, last_seen_at,"
            " expires_at, ip, user_agent) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
            (secrets.token_hex(8), fast_hash(token), user_id, _iso(now), _iso(now),
             _iso(now + timedelta(seconds=ttl)), ip, (user_agent or "")[:200] or None),
        )
    return token


def verify(token: str | None) -> str | None:
    """
    Return the user id for a live session, else None.

    Expiry is absolute: a session is over thirty days after it was opened,
    however recently it was used. Using it keeps last_seen_at fresh for the
    administrator's view and nothing else.
    """
    if not token:
        return None
    init()
    now = _now()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE token_hash = ?;", (fast_hash(token),)
        ).fetchone()
        if not row:
            return None

        try:
            expires_at = _parse(row["expires_at"])
            last_seen = _parse(row["last_seen_at"])
        except ValueError:
            conn.execute("DELETE FROM sessions WHERE id = ?;", (row["id"],))
            return None

        if now >= expires_at:
            conn.execute("DELETE FROM sessions WHERE id = ?;", (row["id"],))
            return None

        if (now - last_seen).total_seconds() >= TOUCH_INTERVAL_SECONDS:
            conn.execute("UPDATE sessions SET last_seen_at = ? WHERE id = ?;",
                         (_iso(now), row["id"]))
        return row["user_id"]


def expires_at(token: str | None) -> str | None:
    """When this session ends, so the interface can say so."""
    if not token:
        return None
    init()
    with _connect() as conn:
        row = conn.execute("SELECT expires_at FROM sessions WHERE token_hash = ?;",
                           (fast_hash(token),)).fetchone()
    return row["expires_at"] if row else None


def destroy(token: str | None) -> None:
    """End one session. Logging out requires a new code to come back."""
    if not token:
        return
    init()
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?;", (fast_hash(token),))


def destroy_all_for_user(user_id: str) -> int:
    """Close every session an account has. Used when access is revoked."""
    init()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE user_id = ?;", (user_id,))
        return cur.rowcount


def list_active() -> list[dict]:
    """Live sessions, most recently used first, for the administrator's view."""
    init()
    purge_expired()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, user_id, created_at, last_seen_at, expires_at, ip, user_agent"
            " FROM sessions ORDER BY last_seen_at DESC;"
        ).fetchall()
    return [dict(r) for r in rows]


def destroy_by_id(session_id: str) -> bool:
    init()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE id = ?;", (session_id,))
        return cur.rowcount > 0


def purge_expired() -> int:
    """Drop sessions past their thirty days. Cheap; called opportunistically."""
    init()
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM sessions WHERE datetime(expires_at) <= datetime(?);",
            (_iso(_now()),),
        )
        return cur.rowcount
