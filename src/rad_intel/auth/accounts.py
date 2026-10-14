"""
Clinician accounts for the review workspace.

There is no public sign-up. An account exists only because an administrator
created it with scripts/manage_clinicians.py, which is the whole point: the
review workspace is a closed group of named clinicians who have agreed to
contribute cases to a shared model, not an open service.

People who want in submit an access request, which lands in a table for the
administrator to read and act on. Submitting a request grants nothing.

Passwords are stored as scrypt hashes with a per-account salt. scrypt is in
the standard library, so this adds no dependency, and its memory cost makes
offline guessing expensive.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rad_intel.config import BASE_DIR

DB_PATH = BASE_DIR / "rad_intel.db"

# scrypt parameters. n=2**15 keeps a single verification near 100 ms on a
# laptop, which is unnoticeable to a user and slow enough to make bulk
# guessing impractical.
_SCRYPT_N = 2 ** 15
_SCRYPT_R = 8
_SCRYPT_P = 1
_KEYLEN = 64
# OpenSSL caps scrypt memory at 32 MB by default, which these parameters
# exceed (128 * N * r is 32 MB exactly, plus overhead), so raise the ceiling.
_SCRYPT_MAXMEM = 96 * 1024 * 1024

SCHEMA = """
CREATE TABLE IF NOT EXISTS clinicians (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    full_name TEXT NOT NULL,
    institution TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'clinician',
    status TEXT NOT NULL DEFAULT 'active',
    password_salt TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    must_change_password INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_login_at TEXT
);

CREATE TABLE IF NOT EXISTS access_requests (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    full_name TEXT NOT NULL,
    email TEXT NOT NULL,
    institution TEXT NOT NULL,
    registration TEXT,
    message TEXT,
    status TEXT NOT NULL DEFAULT 'new'
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Return (salt, hash) as hex. A fresh salt is generated when none is given."""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=bytes.fromhex(salt),
        n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_KEYLEN,
        maxmem=_SCRYPT_MAXMEM,
    )
    return salt, digest.hex()


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    _, candidate = hash_password(password, salt)
    return hmac.compare_digest(candidate, expected_hash)


# --- administration --------------------------------------------------------

def create_clinician(
    email: str,
    full_name: str,
    institution: str,
    password: str,
    role: str = "clinician",
) -> dict[str, Any]:
    """Create an account. Raises ValueError when the email already exists."""
    init()
    email = email.strip().lower()
    salt, digest = hash_password(password)
    record = {
        "id": str(uuid.uuid4()),
        "email": email,
        "full_name": full_name.strip(),
        "institution": institution.strip(),
        "role": role,
        "status": "active",
        "password_salt": salt,
        "password_hash": digest,
        "must_change_password": 1,
        "created_at": _now(),
        "last_login_at": None,
    }
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO clinicians (id, email, full_name, institution, role, status,"
                " password_salt, password_hash, must_change_password, created_at,"
                " last_login_at) VALUES (:id, :email, :full_name, :institution, :role,"
                " :status, :password_salt, :password_hash, :must_change_password,"
                " :created_at, :last_login_at);",
                record,
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"An account already exists for {email}") from exc
    return record


def set_password(email: str, password: str, force_change: bool = True) -> bool:
    init()
    salt, digest = hash_password(password)
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE clinicians SET password_salt = ?, password_hash = ?,"
            " must_change_password = ? WHERE email = ?;",
            (salt, digest, 1 if force_change else 0, email.strip().lower()),
        )
        return cur.rowcount > 0


def set_status(email: str, status: str) -> bool:
    """status is 'active' or 'disabled'. Disabling takes effect on next request."""
    init()
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE clinicians SET status = ? WHERE email = ?;",
            (status, email.strip().lower()),
        )
        return cur.rowcount > 0


def list_clinicians() -> list[dict[str, Any]]:
    init()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, email, full_name, institution, role, status, created_at,"
            " last_login_at FROM clinicians ORDER BY created_at;"
        ).fetchall()
    return [dict(r) for r in rows]


def get_by_email(email: str) -> dict[str, Any] | None:
    init()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM clinicians WHERE email = ?;", (email.strip().lower(),)
        ).fetchone()
    return dict(row) if row else None


def get_by_id(user_id: str) -> dict[str, Any] | None:
    init()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM clinicians WHERE id = ?;", (user_id,)).fetchone()
    return dict(row) if row else None


def authenticate(email: str, password: str) -> dict[str, Any] | None:
    """Return the account on success, None on any failure."""
    user = get_by_email(email)
    if not user or user["status"] != "active":
        # Still spend the hashing time, so a disabled or unknown address cannot
        # be distinguished from a wrong password by how long the reply takes.
        hash_password(password)
        return None
    if not verify_password(password, user["password_salt"], user["password_hash"]):
        return None
    with _connect() as conn:
        conn.execute("UPDATE clinicians SET last_login_at = ? WHERE id = ?;",
                     (_now(), user["id"]))
    return user


def change_own_password(user_id: str, current: str, new: str) -> tuple[bool, str]:
    user = get_by_id(user_id)
    if not user:
        return False, "Account not found."
    if not verify_password(current, user["password_salt"], user["password_hash"]):
        return False, "Current password is incorrect."
    if len(new) < 10:
        return False, "Choose a password of at least 10 characters."
    salt, digest = hash_password(new)
    with _connect() as conn:
        conn.execute(
            "UPDATE clinicians SET password_salt = ?, password_hash = ?,"
            " must_change_password = 0 WHERE id = ?;",
            (salt, digest, user_id),
        )
    return True, "Password updated."


# --- access requests -------------------------------------------------------

def create_access_request(
    full_name: str,
    email: str,
    institution: str,
    registration: str | None = None,
    message: str | None = None,
) -> str:
    """Record a request for access. This grants nothing on its own."""
    init()
    request_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO access_requests (id, created_at, full_name, email, institution,"
            " registration, message, status) VALUES (?, ?, ?, ?, ?, ?, ?, 'new');",
            (request_id, _now(), full_name.strip()[:120], email.strip().lower()[:200],
             institution.strip()[:160], (registration or "").strip()[:120] or None,
             (message or "").strip()[:1000] or None),
        )
    return request_id


def list_access_requests(status: str | None = None) -> list[dict[str, Any]]:
    init()
    query = "SELECT * FROM access_requests"
    params: tuple = ()
    if status:
        query += " WHERE status = ?"
        params = (status,)
    query += " ORDER BY datetime(created_at) DESC;"
    with _connect() as conn:
        return [dict(r) for r in conn.execute(query, params).fetchall()]


def set_request_status(request_id: str, status: str) -> bool:
    init()
    with _connect() as conn:
        cur = conn.execute("UPDATE access_requests SET status = ? WHERE id = ?;",
                           (status, request_id))
        return cur.rowcount > 0
