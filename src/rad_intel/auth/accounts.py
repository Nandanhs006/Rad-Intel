"""
Clinician accounts for the review workspace.

There is no public sign-up. An account exists only because an administrator
created it, which is the whole point: the review workspace is a closed group
of named clinicians who have agreed to contribute cases to a shared model,
not an open service. People who want in submit an access request, which
records their details for the administrator and grants nothing.

Two kinds of account, with deliberately different first factors:

* A **clinician** proves who they are with their email and the Rad-Intel ID
  the administrator handed them in person. They have no password, so there is
  no password for them to reuse from somewhere else, nothing to phish out of
  them in a form, and nothing for us to store that a leak could expose
  elsewhere.
* An **administrator** proves it with email and a password, because that
  account can read and delete everybody's data and should not be reachable
  with an identifier that gets written down.

Both then have to enter a one-time code sent to the phone number on their
record (see otp.py). Neither first factor alone opens a session.

What is stored how:

* Passwords are scrypt hashes with a per-account salt. They are never
  readable, not even by the administrator.
* Rad-Intel IDs and phone numbers are AES-GCM encrypted, because the
  administrator genuinely has to read them back -- to re-send an ID to a
  clinician who lost it, and to send a code to a phone. Each is stored
  alongside a keyed blind index so an exact lookup still works without
  decrypting the whole table, and a last-four fragment so lists can be shown
  without decrypting anything.
"""
from __future__ import annotations

import hmac
import secrets
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from rad_intel.auth import crypto, store
from rad_intel.config import BASE_DIR

DB_PATH = BASE_DIR / "rad_intel.db"

# No I, L, O, 0 or 1: these IDs are read aloud and copied by hand, and in
# most fonts those five are two pairs and a near-miss.
_ID_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
ID_PREFIX = "RAD"
ID_BODY_LENGTH = 9

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

# Columns added after the first release. SQLite has no "ADD COLUMN IF NOT
# EXISTS", so they are applied by comparing against PRAGMA table_info.
_ADDED_COLUMNS = {
    "clinicians": {
        "radintel_id_enc": "TEXT",
        "radintel_id_index": "TEXT",
        "radintel_id_tail": "TEXT",
        "phone_enc": "TEXT",
        "phone_tail": "TEXT",
        "admin_note": "TEXT",
        "updated_at": "TEXT",
    },
    "access_requests": {
        "phone_enc": "TEXT",
        "phone_tail": "TEXT",
        "notified": "INTEGER NOT NULL DEFAULT 0",
    },
}


def _connect():
    """A transaction on this module's tables: commits, and always closes."""
    return store.transaction(DB_PATH)


def _build(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    for table, columns in _ADDED_COLUMNS.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table});")}
        for name, decl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl};")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_clinicians_radid"
        " ON clinicians(radintel_id_index);"
    )


def init() -> None:
    """
    Build the tables, once per process.

    Every accessor below calls this. It used to execute the DDL each time,
    which takes a write lock, so reading one account blocked every other
    request. Now it is a set lookup after the first call.
    """
    store.ensure_schema(DB_PATH, "accounts", _build)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Rad-Intel IDs ---------------------------------------------------------

def generate_radintel_id() -> str:
    """A fresh Rad-Intel ID, e.g. RAD7KQ4MX2PH. Checked for collisions."""
    init()
    for _ in range(50):
        candidate = ID_PREFIX + "".join(
            secrets.choice(_ID_ALPHABET) for _ in range(ID_BODY_LENGTH)
        )
        if not get_by_radintel_id(candidate):
            return candidate
    raise RuntimeError("Could not generate an unused Rad-Intel ID.")


def _id_index(radintel_id: str) -> str:
    """Keyed index over the normalised ID, for exact lookup without decryption."""
    return crypto.fast_hash("radid:" + radintel_id.strip().upper())


def normalise_id(radintel_id: str | None) -> str:
    """Accept rad-7kq4, RAD 7KQ4 ... as the same ID. Spaces and dashes are noise."""
    if not radintel_id:
        return ""
    return "".join(ch for ch in radintel_id.upper() if ch.isalnum())


def _phone_tail(phone: str | None) -> str | None:
    digits = "".join(ch for ch in (phone or "") if ch.isdigit())
    return digits[-4:] if len(digits) >= 4 else None


def _row_to_account(row: sqlite3.Row | None) -> dict[str, Any] | None:
    """A row plus the decrypted fields, with the hash material left out."""
    if row is None:
        return None
    account = dict(row)
    account["radintel_id"] = crypto.decrypt_field(account.get("radintel_id_enc"))
    account["phone"] = crypto.decrypt_field(account.get("phone_enc"))
    return account


def public_view(account: dict[str, Any]) -> dict[str, Any]:
    """
    What may be shown to another clinician: a name and a site, nothing more.

    The roster exists so reviewers know whose cases are in the queue. It is
    not a directory of contact details, so emails, phone numbers and
    Rad-Intel IDs never appear in it.
    """
    return {
        "full_name": account["full_name"],
        "institution": account["institution"],
        "role": account["role"],
        "status": account["status"],
    }


def self_view(account: dict[str, Any]) -> dict[str, Any]:
    """Everything an account may see about itself."""
    return {
        "id": account["id"],
        "email": account["email"],
        "full_name": account["full_name"],
        "institution": account["institution"],
        "role": account["role"],
        "status": account["status"],
        "radintel_id": account.get("radintel_id"),
        "phone": account.get("phone"),
        "created_at": account["created_at"],
        "last_login_at": account.get("last_login_at"),
    }


# --- administration --------------------------------------------------------

def create_clinician(
    email: str,
    full_name: str,
    institution: str,
    password: str | None = None,
    role: str = "clinician",
    phone: str | None = None,
    radintel_id: str | None = None,
    admin_note: str | None = None,
) -> dict[str, Any]:
    """
    Create an account and return it, including its Rad-Intel ID in the clear.

    This is the only moment the caller is handed the ID without going through
    the administrator's own view, because it has to be passed to the clinician.

    A clinician needs no password: the Rad-Intel ID is their first factor. One
    is still generated and stored as an unusable random value, so that every
    row has hash material and a login attempt against a passwordless account
    takes the same time as one against an administrator.
    """
    init()
    email = email.strip().lower()
    if not email or "@" not in email:
        raise ValueError("A valid email address is required.")
    if not full_name.strip():
        raise ValueError("A full name is required.")
    if not institution.strip():
        raise ValueError("An institution is required.")
    if role == "admin" and not password:
        raise ValueError("An administrator account needs a password.")

    salt, digest = crypto.hash_secret(password or secrets.token_urlsafe(32))
    assigned_id = normalise_id(radintel_id) or generate_radintel_id()
    if not assigned_id.startswith(ID_PREFIX):
        raise ValueError(f"A Rad-Intel ID must start with {ID_PREFIX}.")

    record = {
        "id": str(uuid.uuid4()),
        "email": email,
        "full_name": full_name.strip()[:120],
        "institution": institution.strip()[:160],
        "role": role,
        "status": "active",
        "password_salt": salt,
        "password_hash": digest,
        "must_change_password": 1 if (password and role == "admin") else 0,
        "created_at": _now(),
        "updated_at": _now(),
        "last_login_at": None,
        "radintel_id_enc": crypto.encrypt_field(assigned_id),
        "radintel_id_index": _id_index(assigned_id),
        "radintel_id_tail": assigned_id[-4:],
        "phone_enc": crypto.encrypt_field((phone or "").strip() or None),
        "phone_tail": _phone_tail(phone),
        "admin_note": (admin_note or "").strip()[:500] or None,
    }
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO clinicians (id, email, full_name, institution, role, status,"
                " password_salt, password_hash, must_change_password, created_at,"
                " updated_at, last_login_at, radintel_id_enc, radintel_id_index,"
                " radintel_id_tail, phone_enc, phone_tail, admin_note)"
                " VALUES (:id, :email, :full_name, :institution, :role, :status,"
                " :password_salt, :password_hash, :must_change_password, :created_at,"
                " :updated_at, :last_login_at, :radintel_id_enc, :radintel_id_index,"
                " :radintel_id_tail, :phone_enc, :phone_tail, :admin_note);",
                record,
            )
    except sqlite3.IntegrityError as exc:
        if "radintel_id_index" in str(exc):
            raise ValueError("That Rad-Intel ID is already in use.") from exc
        raise ValueError(f"An account already exists for {email}") from exc

    out = dict(record)
    out["radintel_id"] = assigned_id
    out["phone"] = (phone or "").strip() or None
    return out


def update_clinician(user_id: str, **fields: Any) -> dict[str, Any] | None:
    """
    Change an account's details. Only the listed fields can be set here.

    Rotating the Rad-Intel ID is allowed, since it is a credential and a
    clinician may need a new one; the caller is responsible for telling them.
    """
    init()
    sets: list[str] = ["updated_at = ?"]
    params: list[Any] = [_now()]

    for column in ("full_name", "institution", "role", "status", "admin_note"):
        if column in fields and fields[column] is not None:
            sets.append(f"{column} = ?")
            params.append(str(fields[column]).strip()[:500])

    if fields.get("email"):
        sets.append("email = ?")
        params.append(str(fields["email"]).strip().lower())

    if "phone" in fields:
        phone = (fields["phone"] or "").strip() or None
        sets.extend(["phone_enc = ?", "phone_tail = ?"])
        params.extend([crypto.encrypt_field(phone), _phone_tail(phone)])

    if fields.get("radintel_id"):
        new_id = normalise_id(fields["radintel_id"])
        if not new_id.startswith(ID_PREFIX):
            raise ValueError(f"A Rad-Intel ID must start with {ID_PREFIX}.")
        sets.extend(["radintel_id_enc = ?", "radintel_id_index = ?", "radintel_id_tail = ?"])
        params.extend([crypto.encrypt_field(new_id), _id_index(new_id), new_id[-4:]])

    params.append(user_id)
    try:
        with _connect() as conn:
            cur = conn.execute(
                f"UPDATE clinicians SET {', '.join(sets)} WHERE id = ?;", params
            )
            if cur.rowcount == 0:
                return None
    except sqlite3.IntegrityError as exc:
        field = "Rad-Intel ID" if "radintel_id_index" in str(exc) else "email address"
        raise ValueError(f"That {field} is already in use by another account.") from exc
    return get_by_id(user_id)


def delete_clinician(user_id: str) -> bool:
    """Remove an account. Its review records are left in place; see routes/admin."""
    init()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM clinicians WHERE id = ?;", (user_id,))
        return cur.rowcount > 0


def set_password(email: str, password: str, force_change: bool = True) -> bool:
    init()
    salt, digest = crypto.hash_secret(password)
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE clinicians SET password_salt = ?, password_hash = ?,"
            " must_change_password = ?, updated_at = ? WHERE email = ?;",
            (salt, digest, 1 if force_change else 0, _now(), email.strip().lower()),
        )
        return cur.rowcount > 0


def set_status(email: str, status: str) -> bool:
    """status is 'active' or 'disabled'. Disabling takes effect immediately."""
    init()
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE clinicians SET status = ?, updated_at = ? WHERE email = ?;",
            (status, _now(), email.strip().lower()),
        )
        return cur.rowcount > 0


def rotate_radintel_id(user_id: str) -> str | None:
    """Issue a new ID for an account and return it, for the administrator to pass on."""
    new_id = generate_radintel_id()
    return new_id if update_clinician(user_id, radintel_id=new_id) else None


def list_clinicians() -> list[dict[str, Any]]:
    init()
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM clinicians ORDER BY full_name COLLATE NOCASE;").fetchall()
    return [_row_to_account(r) for r in rows]


def search_clinicians(query: str | None = None) -> list[dict[str, Any]]:
    """
    Find accounts by name, email or Rad-Intel ID.

    An ID lookup goes through the blind index and so has to be exact; name and
    email are substring matches, which is what a person searching a roster
    expects.
    """
    accounts = list_clinicians()
    term = (query or "").strip()
    if not term:
        return accounts

    exact = get_by_radintel_id(term)
    if exact:
        return [exact]

    lowered = term.lower()
    return [
        a for a in accounts
        if lowered in a["full_name"].lower()
        or lowered in a["email"].lower()
        or lowered in (a["institution"] or "").lower()
    ]


def get_by_email(email: str) -> dict[str, Any] | None:
    init()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM clinicians WHERE email = ?;", (email.strip().lower(),)
        ).fetchone()
    return _row_to_account(row)


def get_by_id(user_id: str) -> dict[str, Any] | None:
    init()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM clinicians WHERE id = ?;", (user_id,)).fetchone()
    return _row_to_account(row)


def get_by_radintel_id(radintel_id: str) -> dict[str, Any] | None:
    init()
    normalised = normalise_id(radintel_id)
    if not normalised:
        return None
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM clinicians WHERE radintel_id_index = ?;",
            (_id_index(normalised),),
        ).fetchone()
    return _row_to_account(row)


def admins() -> list[dict[str, Any]]:
    return [a for a in list_clinicians() if a["role"] == "admin" and a["status"] == "active"]


# --- first factors ---------------------------------------------------------
#
# Neither of these opens a session. They establish *which* account is being
# claimed; otp.py then has to confirm the claim against the registered phone.
# Both deliberately do the same amount of work whatever the outcome, so a
# wrong guess cannot be told from a non-existent account by how long the
# response takes.

def _unusable_work() -> None:
    """Burn the same time a real verification costs, for a failed lookup."""
    crypto.hash_secret("decoy", "00" * 16)


def identify_clinician(email: str, radintel_id: str) -> dict[str, Any] | None:
    """Check that this email and this Rad-Intel ID name the same live account."""
    user = get_by_email(email)
    if not user or user["status"] != "active":
        _unusable_work()
        return None
    stored = user.get("radintel_id")
    if not stored:
        _unusable_work()
        return None
    if not hmac.compare_digest(stored.upper(), normalise_id(radintel_id)):
        _unusable_work()
        return None
    return user


def identify_admin(email: str, password: str) -> dict[str, Any] | None:
    """Check an administrator's email and password."""
    user = get_by_email(email)
    if not user or user["status"] != "active" or user["role"] != "admin":
        _unusable_work()
        return None
    if not crypto.verify_secret(password, user["password_salt"], user["password_hash"]):
        return None
    return user


def note_successful_login(user_id: str) -> None:
    init()
    with _connect() as conn:
        conn.execute("UPDATE clinicians SET last_login_at = ? WHERE id = ?;",
                     (_now(), user_id))


def change_own_password(user_id: str, current: str, new: str) -> tuple[bool, str]:
    user = get_by_id(user_id)
    if not user:
        return False, "Account not found."
    if not crypto.verify_secret(current, user["password_salt"], user["password_hash"]):
        return False, "Current password is incorrect."
    if len(new) < 10:
        return False, "Choose a password of at least 10 characters."
    salt, digest = crypto.hash_secret(new)
    with _connect() as conn:
        conn.execute(
            "UPDATE clinicians SET password_salt = ?, password_hash = ?,"
            " must_change_password = 0, updated_at = ? WHERE id = ?;",
            (salt, digest, _now(), user_id),
        )
    return True, "Password updated."


# --- access requests -------------------------------------------------------

def create_access_request(
    full_name: str,
    email: str,
    institution: str,
    phone: str | None = None,
    registration: str | None = None,
    message: str | None = None,
) -> str:
    """Record a request for access. This grants nothing on its own."""
    init()
    request_id = str(uuid.uuid4())
    phone = (phone or "").strip() or None
    with _connect() as conn:
        conn.execute(
            "INSERT INTO access_requests (id, created_at, full_name, email, institution,"
            " registration, message, status, phone_enc, phone_tail, notified)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, 'new', ?, ?, 0);",
            (request_id, _now(), full_name.strip()[:120], email.strip().lower()[:200],
             institution.strip()[:160], (registration or "").strip()[:120] or None,
             (message or "").strip()[:1000] or None,
             crypto.encrypt_field(phone), _phone_tail(phone)),
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
        rows = conn.execute(query, params).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["phone"] = crypto.decrypt_field(item.get("phone_enc"))
        out.append(item)
    return out


def set_request_status(request_id: str, status: str) -> bool:
    init()
    with _connect() as conn:
        cur = conn.execute("UPDATE access_requests SET status = ? WHERE id = ?;",
                           (status, request_id))
        return cur.rowcount > 0


def mark_request_notified(request_id: str) -> None:
    init()
    with _connect() as conn:
        conn.execute("UPDATE access_requests SET notified = 1 WHERE id = ?;", (request_id,))


def delete_access_request(request_id: str) -> bool:
    init()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM access_requests WHERE id = ?;", (request_id,))
        return cur.rowcount > 0
