"""
Server-side sessions.

The cookie carries nothing but a random 256-bit token. Everything that
matters -- which account it belongs to, when it was opened, when it was last
used -- lives in a row on the server, and only a hash of the token is stored,
so a copy of the database cannot be used to log in as anybody.

Holding the state server-side buys three things a self-contained signed token
cannot give:

* **Revocation.** Deleting the row ends the session on the next request.
  A signed token stays valid until it expires, whatever the server thinks.
* **Idle expiry.** A session left open on a shared workstation closes itself
  after IDLE_TIMEOUT of no activity.
* **Visibility.** The administrator can see who currently has a session open
  and close any of them.

The cookie is a session cookie -- no Max-Age, no Expires -- so the browser
discards it when it closes, which is the "log out when the site is closed"
behaviour the workspace is specified to have. The server-side idle timeout is
the backstop, because a browser that restores its previous tabs will also
restore session cookies.
"""
from __future__ import annotations

import secrets
import sqlite3
from datetime import datetime, timedelta, timezone

from rad_intel.auth.crypto import fast_hash
from rad_intel.auth import store
from rad_intel.config import BASE_DIR

DB_PATH = BASE_DIR / "rad_intel.db"

COOKIE_NAME = "radintel_session"

# An absolute ceiling on one session, regardless of activity.
SESSION_TTL_SECONDS = 12 * 60 * 60
# Inactivity after which a session is closed even if the browser stayed open.
IDLE_TIMEOUT_SECONDS = 60 * 60
# How stale last_seen_at may get before it is written again. Without this,
# every request -- including the background status poll -- took a write lock
# on the shared database purely to move a timestamp a few seconds, and the
# parallel requests a page load makes were enough to collide.
TOUCH_INTERVAL_SECONDS = 60

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


def issue(user_id: str, ttl: int = SESSION_TTL_SECONDS, ip: str | None = None,
          user_agent: str | None = None) -> str:
    """Open a session and return the opaque token to put in the cookie."""
    init()
    token = secrets.token_urlsafe(32)
    now = _now()
    with _connect() as conn:
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

    Touches last_seen_at, which is what makes the idle timeout a timeout on
    inactivity rather than on age.
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

        if now >= expires_at or (now - last_seen).total_seconds() > IDLE_TIMEOUT_SECONDS:
            conn.execute("DELETE FROM sessions WHERE id = ?;", (row["id"],))
            return None

        if (now - last_seen).total_seconds() >= TOUCH_INTERVAL_SECONDS:
            conn.execute("UPDATE sessions SET last_seen_at = ? WHERE id = ?;",
                         (_iso(now), row["id"]))
        return row["user_id"]


def destroy(token: str | None) -> None:
    """End one session. Signing out must not end the user's other sessions."""
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
    """Live sessions, newest first, for the administrator's view."""
    init()
    purge_expired()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, user_id, created_at, last_seen_at, ip, user_agent FROM sessions"
            " ORDER BY datetime(last_seen_at) DESC;"
        ).fetchall()
    return [dict(r) for r in rows]


def destroy_by_id(session_id: str) -> bool:
    init()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE id = ?;", (session_id,))
        return cur.rowcount > 0


def purge_expired() -> int:
    """Drop rows that are past their ceiling or idle. Cheap; called opportunistically."""
    init()
    now = _now()
    cutoff = _iso(now - timedelta(seconds=IDLE_TIMEOUT_SECONDS))
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM sessions WHERE datetime(expires_at) <= datetime(?)"
            " OR datetime(last_seen_at) <= datetime(?);",
            (_iso(now), cutoff),
        )
        return cur.rowcount
