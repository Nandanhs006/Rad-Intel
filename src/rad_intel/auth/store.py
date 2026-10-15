"""
SQLite connections for the account, session, code and activity tables.

All four live in one database file, and the workspace opens several requests
at once -- the queue, the roster, the model list, the console -- each of which
runs its dependency in its own thread with its own connection. On the default
SQLite settings that produced `database is locked` almost immediately, so the
rules are set in one place here rather than per module.

What this fixes, in the order it mattered:

* **Schema setup ran on every call.** Each accessor started with `init()`,
  which executes DDL and therefore takes a write lock. Reading one account
  took a write lock; eight parallel requests fought over it. The schema is
  now built once per database file per process.
* **Rollback journalling blocks readers.** Write-ahead logging lets readers
  carry on while a write is in progress, which is the normal case here: lots
  of small reads around the occasional write.
* **A busy connection gave up at once.** A fifteen-second busy timeout makes
  a blocked statement wait for the lock instead of failing the request.
* **Connections were never closed.** `with sqlite3.connect(...) as conn`
  commits on exit; it does not close. Every call leaked an open connection
  that went on holding its read mark, so the locks piled up until a write
  could not get through. `transaction()` below commits *and* closes, and is
  the only way these tables are opened.

WAL also makes this safe across processes, which matters because the
administrator's command-line tools open the same file while the server is
running.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Callable, Iterator

BUSY_TIMEOUT_SECONDS = 15.0

_schema_lock = threading.Lock()
_built: set[str] = set()


def connect(db_path: Path | str) -> sqlite3.Connection:
    """A raw connection with the workspace's locking behaviour applied."""
    conn = sqlite3.connect(db_path, timeout=BUSY_TIMEOUT_SECONDS)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {int(BUSY_TIMEOUT_SECONDS * 1000)};")
    return conn


@contextmanager
def transaction(db_path: Path | str) -> Iterator[sqlite3.Connection]:
    """
    Open, commit on success, roll back on failure, and always close.

    Used as `with transaction(DB_PATH) as conn:`. The closing is the part
    that matters: sqlite3's own connection context manager commits but leaves
    the connection open, which under WAL means its read mark is held and the
    next writer eventually cannot get a lock.
    """
    with closing(connect(db_path)) as conn:
        with conn:
            yield conn


def ensure_schema(db_path: Path | str, name: str,
                  build: Callable[[sqlite3.Connection], None]) -> None:
    """
    Build a module's tables, once per database file per process.

    Keyed by path as well as by name so a test pointing the module at a
    scratch database still gets its schema built.
    """
    token = f"{name}@{db_path}"
    if token in _built:
        return
    with _schema_lock:
        if token in _built:
            return
        with transaction(db_path) as conn:
            # Persistent on the file; setting it again on an existing WAL
            # database is a no-op.
            conn.execute("PRAGMA journal_mode = WAL;")
            # Durability one notch below full: a crash can lose the last
            # transaction, never corrupt the file. For a review queue and a
            # session table that is the right trade.
            conn.execute("PRAGMA synchronous = NORMAL;")
            build(conn)
        _built.add(token)


def forget(db_path: Path | str | None = None) -> None:
    """Drop the built-schema memo. For tests that recreate a database."""
    with _schema_lock:
        if db_path is None:
            _built.clear()
        else:
            for token in [t for t in _built if t.endswith(f"@{db_path}")]:
                _built.discard(token)
