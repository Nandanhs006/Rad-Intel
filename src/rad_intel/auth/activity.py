"""
Activity log for the review workspace.

Every action a signed-in account takes against shared data is recorded: who,
what, when, from where. Two reasons this exists rather than being optional.

The first is accountability. The queue holds radiographs from several sites
and every participant can see all of them. If a record is edited or deleted,
or a training run is started, it has to be answerable who did it -- otherwise
"every clinician can see everything" has no counterweight.

The second is that the log is the only way to notice misuse. A single account
pulling every image in the store looks exactly like normal use at the level of
one request, and only shows up as a pattern.

Entries are written on a background thread. Two reasons, and the second is
the one that forced it: a request must never be slowed down by its own audit
trail, and -- because the endpoints are async -- a synchronous write here runs
on the event loop. When several clinicians edited cases at the same moment,
that blocking write sat on the loop waiting for a lock that only another
task's commit could release, and the whole server deadlocked until the busy
timeout expired. Nothing waits on the log, so nothing should block for it.

The log is itself sensitive -- it is a record of named clinicians' behaviour --
so it is readable by the administrator alone and entries carry no request
bodies, no image contents and no credentials. The client address is kept in
full rather than truncated: a partial address cannot distinguish two people
on the same network, which is the only question the log exists to answer.
Entries are dropped after a year.
"""
from __future__ import annotations

import atexit
import logging
import queue
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from rad_intel.auth import store
from rad_intel.config import BASE_DIR

logger = logging.getLogger(__name__)

DB_PATH = BASE_DIR / "rad_intel.db"

RETENTION_DAYS = 365

SCHEMA = """
CREATE TABLE IF NOT EXISTS activity_log (
    id TEXT PRIMARY KEY,
    at TEXT NOT NULL,
    user_id TEXT,
    actor TEXT,
    role TEXT,
    action TEXT NOT NULL,
    detail TEXT,
    ip TEXT,
    outcome TEXT NOT NULL DEFAULT 'ok'
);
CREATE INDEX IF NOT EXISTS idx_activity_at ON activity_log(at);
CREATE INDEX IF NOT EXISTS idx_activity_user ON activity_log(user_id);
"""


def _connect():
    """A transaction on this module's tables: commits, and always closes."""
    return store.transaction(DB_PATH)


def init() -> None:
    """Build the tables, once per process: see auth/store.py."""
    store.ensure_schema(DB_PATH, "activity_log", lambda conn: conn.executescript(SCHEMA))


_pending: "queue.Queue[tuple | None]" = queue.Queue(maxsize=10_000)
_writer: threading.Thread | None = None
_writer_lock = threading.Lock()


def _drain() -> None:
    """Write queued entries until the process ends."""
    while True:
        row = _pending.get()
        try:
            if row is None:
                return
            try:
                init()
                with _connect() as conn:
                    conn.execute(
                        "INSERT INTO activity_log (id, at, user_id, actor, role, action,"
                        " detail, ip, outcome) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);",
                        row,
                    )
            except Exception as exc:  # pragma: no cover - never break a request
                logger.warning("Could not write an activity entry: %s", exc)
        finally:
            _pending.task_done()


def _ensure_writer() -> None:
    global _writer
    if _writer and _writer.is_alive():
        return
    with _writer_lock:
        if _writer and _writer.is_alive():
            return
        _writer = threading.Thread(target=_drain, name="activity-log", daemon=True)
        _writer.start()


def record(action: str, user: dict[str, Any] | None = None, detail: str | None = None,
           ip: str | None = None, outcome: str = "ok", actor: str | None = None) -> None:
    """
    Append one entry. Returns at once and never raises.

    `actor` covers the cases where there is no account yet -- a sign-in attempt
    against an address that does not exist, or an access request from a
    stranger -- so a failed attempt is still attributable to something.
    """
    row = (
        str(uuid.uuid4()),
        datetime.now(timezone.utc).isoformat(),
        (user or {}).get("id"),
        actor or (user or {}).get("email") or "anonymous",
        (user or {}).get("role"),
        action,
        (detail or "")[:500] or None,
        ip,
        outcome,
    )
    try:
        _ensure_writer()
        # Drop rather than block if the queue ever backs up: a request must
        # not wait on its own audit trail.
        _pending.put_nowait(row)
    except Exception:  # pragma: no cover
        logger.warning("Activity entry dropped: the log queue is full or unavailable.")


def flush(timeout: float = 5.0) -> None:
    """
    Wait for queued entries to reach the database.

    Reading the log waits for its own writes, so the console and the tests
    never show a view that is a few milliseconds out of date.
    """
    if _pending.unfinished_tasks:
        deadline = threading.Event()
        threading.Thread(target=lambda: (_pending.join(), deadline.set()),
                         daemon=True).start()
        deadline.wait(timeout)


@atexit.register
def _stop_writer() -> None:  # pragma: no cover - interpreter shutdown
    if _writer and _writer.is_alive():
        flush(2.0)


def recent(limit: int = 200, offset: int = 0, user_id: str | None = None,
           action: str | None = None) -> list[dict[str, Any]]:
    flush()
    init()
    clauses, params = [], []
    if user_id:
        clauses.append("user_id = ?")
        params.append(user_id)
    if action:
        clauses.append("action = ?")
        params.append(action)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.extend([min(limit, 1000), offset])
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM activity_log {where} ORDER BY at DESC"
            " LIMIT ? OFFSET ?;", params,
        ).fetchall()
    return [dict(r) for r in rows]


def count() -> int:
    flush()
    init()
    with _connect() as conn:
        (total,) = conn.execute("SELECT COUNT(*) FROM activity_log;").fetchone()
    return int(total)


def purge_old() -> int:
    """Drop entries past the retention window."""
    init()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).isoformat()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM activity_log WHERE datetime(at) < datetime(?);",
                           (cutoff,))
        return cur.rowcount
