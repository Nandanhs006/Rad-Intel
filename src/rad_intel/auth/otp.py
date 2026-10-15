"""
One-time verification codes.

Every sign-in, for everyone, needs a fresh code delivered to the phone number
on the account's record. The first factor -- a Rad-Intel ID for a clinician, a
password for the administrator -- only says which account is being claimed.
This is what confirms the claim.

Properties the implementation has to hold, and the reason each one matters:

* **Single use.** A code is consumed the moment it verifies, so one observed
  over someone's shoulder is already spent.
* **Short-lived.** Ten minutes. Long enough to read a message and type twelve
  characters, short enough that an old screenshot is worthless.
* **Never repeated.** A hash of every code an account has ever been issued is
  kept and checked, so no code is ever sent twice to the same person. With a
  95-character alphabet and twelve characters this would essentially never
  happen by chance; it is enforced rather than assumed because "essentially
  never" is not a security property.
* **Attempt-limited.** Five wrong guesses burn the code. Without this, twelve
  characters of entropy would not help against a script.
* **Request-limited.** Five codes per fifteen minutes per account, so the
  endpoint cannot be used to flood someone's phone or run up an SMS bill.
* **Stored hashed.** A copy of the database contains no usable code.
* **One outstanding code at a time.** Requesting a new one invalidates the
  previous, so two codes in a phone's message list are never both live.
"""
from __future__ import annotations

import logging
import secrets
import sqlite3
import string
import uuid
from datetime import datetime, timedelta, timezone

from rad_intel.auth.crypto import fast_hash
from rad_intel.auth import store
from rad_intel.config import BASE_DIR

logger = logging.getLogger(__name__)

DB_PATH = BASE_DIR / "rad_intel.db"

CODE_LENGTH = 12
# Letters, digits and symbols, as specified. Quotes, backslashes and spaces
# are left out: they survive neither copy-paste out of a chat app nor being
# read aloud over a phone.
CODE_ALPHABET = string.ascii_letters + string.digits + "!#$%*+-=?@"

CODE_TTL_SECONDS = 10 * 60
MAX_ATTEMPTS = 5
REQUEST_WINDOW_SECONDS = 15 * 60

# Pressing the button again straight away is almost always impatience rather
# than a lost code, so within this window the request is declined without
# minting anything and without counting against the allowance below. The
# message points at the inbox instead. This is what stops ordinary use
# reaching a limit meant for abuse.
RESEND_COOLDOWN_SECONDS = 60

# How many codes an account may ask for in a window. Loose, because a code
# costs nothing to send and the only harm a run of them does is clutter an
# inbox. It is still a limit: without one, an unknown Rad-Intel ID and twelve
# characters of code are both things a script can sit and guess at.
MAX_REQUESTS_PER_WINDOW = 12

# How long issued-code hashes are kept, purely to guarantee no code is reused.
HISTORY_RETENTION_DAYS = 365

SCHEMA = """
CREATE TABLE IF NOT EXISTS login_codes (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    code_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT 'live',
    channel TEXT,
    destination_tail TEXT,
    mode TEXT NOT NULL DEFAULT 'clinician'
);
CREATE INDEX IF NOT EXISTS idx_login_codes_user ON login_codes(user_id);
CREATE INDEX IF NOT EXISTS idx_login_codes_hash ON login_codes(code_hash);
"""


def _connect():
    """A transaction on this module's tables: commits, and always closes."""
    return store.transaction(DB_PATH)


def _build(conn) -> None:
    """
    Create the tables, and add columns to a database made before them.

    SQLite has no "ADD COLUMN IF NOT EXISTS", and CREATE TABLE IF NOT
    EXISTS silently does nothing to a table that is already there, so a
    new column has to be applied by comparing against PRAGMA table_info.
    Without this an existing deployment keeps its old schema and fails on
    the first insert.
    """
    conn.executescript(SCHEMA)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(login_codes);")}
    if "mode" not in existing:
        conn.execute("ALTER TABLE login_codes ADD COLUMN mode TEXT NOT NULL"
                     " DEFAULT 'clinician';")


def init() -> None:
    """Build the tables, once per process: see auth/store.py."""
    store.ensure_schema(DB_PATH, "login_codes", _build)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat()


def _parse(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def generate_code() -> str:
    """Twelve characters drawn uniformly from the alphabet, ~78 bits of entropy."""
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))


def seconds_since_last_code(user_id: str) -> float | None:
    """Age of the most recent code request, or None when there has not been one."""
    init()
    with _connect() as conn:
        row = conn.execute(
            "SELECT created_at FROM login_codes WHERE user_id = ?"
            " ORDER BY created_at DESC LIMIT 1;", (user_id,),
        ).fetchone()
    if not row:
        return None
    return (_now() - _parse(row["created_at"])).total_seconds()


def recent_request_count(user_id: str) -> int:
    """How many codes this account has asked for inside the rate-limit window."""
    init()
    cutoff = _iso(_now() - timedelta(seconds=REQUEST_WINDOW_SECONDS))
    with _connect() as conn:
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM login_codes WHERE user_id = ?"
            " AND datetime(created_at) > datetime(?);",
            (user_id, cutoff),
        ).fetchone()
    return int(count)


def seconds_until_next_allowed(user_id: str) -> int:
    """How long an account that has hit the limit must wait, in whole seconds."""
    init()
    cutoff = _now() - timedelta(seconds=REQUEST_WINDOW_SECONDS)
    with _connect() as conn:
        rows = conn.execute(
            "SELECT created_at FROM login_codes WHERE user_id = ?"
            " AND datetime(created_at) > datetime(?) ORDER BY datetime(created_at);",
            (user_id, _iso(cutoff)),
        ).fetchall()
    if len(rows) < MAX_REQUESTS_PER_WINDOW:
        return 0
    oldest = _parse(rows[0]["created_at"])
    return max(1, int((oldest + timedelta(seconds=REQUEST_WINDOW_SECONDS) - _now())
                      .total_seconds()))


def issue_code(user_id: str, channel: str = "email",
               destination_tail: str | None = None,
               rate_limited: bool = True,
               mode: str = "clinician") -> tuple[str | None, str]:
    """
    Mint a code for an account.

    Returns (code, message). The code is returned exactly once, to be handed
    straight to the delivery channel; it is never recoverable afterwards. On
    refusal the code is None and the message says why.

    `mode` records which door the code was asked for -- "clinician" or
    "admin" -- and comes back from verify_code. It is decided here, where the
    first factor was actually checked, rather than being taken on trust from
    whoever submits the code.

    `rate_limited` is False only for a code minted on the server's own
    console. The limit exists to stop the endpoint being used to flood
    somebody's phone and run up an SMS bill; a console code sends nothing to
    anybody, and the person running it already has a shell on the machine.
    Keeping the limit there would mean an administrator locked out by a
    failing provider gets locked out a second time by the recovery path.
    """
    init()
    if rate_limited:
        age = seconds_since_last_code(user_id)
        if age is not None and age < RESEND_COOLDOWN_SECONDS:
            wait = int(RESEND_COOLDOWN_SECONDS - age) + 1
            return None, (
                f"A code was sent less than a minute ago — check your inbox, including "
                f"the spam folder. If it has not arrived, you can ask for another in "
                f"{wait} second{'' if wait == 1 else 's'}."
            )

        if recent_request_count(user_id) >= MAX_REQUESTS_PER_WINDOW:
            wait = seconds_until_next_allowed(user_id)
            minutes = max(1, round(wait / 60))
            return None, (
                f"Too many codes requested for this account. Wait about "
                f"{minutes} minute{'' if minutes == 1 else 's'} and try again. If you "
                f"are locked out and need in now, the study administrator can issue a "
                f"code directly."
            )

    now = _now()
    with _connect() as conn:
        # Only ever one live code, so a phone never holds two valid ones.
        conn.execute(
            "UPDATE login_codes SET state = 'superseded' WHERE user_id = ? AND state = 'live';",
            (user_id,),
        )

        for _ in range(20):
            code = generate_code()
            code_hash = fast_hash(code)
            seen = conn.execute(
                "SELECT 1 FROM login_codes WHERE user_id = ? AND code_hash = ?;",
                (user_id, code_hash),
            ).fetchone()
            if not seen:
                break
        else:
            return None, "Could not generate a new verification code. Try again."

        conn.execute(
            "INSERT INTO login_codes (id, user_id, code_hash, created_at, expires_at,"
            " attempts, state, channel, destination_tail, mode)"
            " VALUES (?, ?, ?, ?, ?, 0, 'live', ?, ?, ?);",
            (str(uuid.uuid4()), user_id, code_hash, _iso(now),
             _iso(now + timedelta(seconds=CODE_TTL_SECONDS)), channel, destination_tail,
             mode),
        )
    return code, "Verification code issued."


def verify_code(user_id: str, code: str) -> tuple[bool, str, str | None]:
    """
    Check a code and consume it on success.

    Returns (ok, message, mode). `mode` is the door the code was asked for,
    so the session that follows can be opened with the privileges that door
    actually grants.

    Deliberately explicit about *why* a code failed -- expired, used up,
    too many attempts -- because the person holding the phone is the one
    reading the message, and "incorrect code" when the real problem is that
    eleven minutes have passed just makes them try the same code again.
    """
    init()
    submitted = (code or "").strip()
    if not submitted:
        return False, "Enter the verification code that was sent to you.", None

    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM login_codes WHERE user_id = ? AND state = 'live'"
            " ORDER BY datetime(created_at) DESC LIMIT 1;",
            (user_id,),
        ).fetchone()

        if not row:
            return False, "No verification code is waiting. Request a new one.", None

        if _now() >= _parse(row["expires_at"]):
            conn.execute("UPDATE login_codes SET state = 'expired' WHERE id = ?;",
                         (row["id"],))
            return False, "That code has expired. Request a new one.", None

        if row["attempts"] >= MAX_ATTEMPTS:
            conn.execute("UPDATE login_codes SET state = 'locked' WHERE id = ?;",
                         (row["id"],))
            return False, "Too many incorrect attempts. Request a new code.", None

        if not secrets.compare_digest(fast_hash(submitted), row["code_hash"]):
            attempts = row["attempts"] + 1
            state = "locked" if attempts >= MAX_ATTEMPTS else "live"
            conn.execute("UPDATE login_codes SET attempts = ?, state = ? WHERE id = ?;",
                         (attempts, state, row["id"]))
            left = MAX_ATTEMPTS - attempts
            if left <= 0:
                return False, "Too many incorrect attempts. Request a new code.", None
            return False, (f"Incorrect code. {left} attempt{'' if left == 1 else 's'} "
                           f"left before it is cancelled."), None

        conn.execute("UPDATE login_codes SET state = 'used' WHERE id = ?;", (row["id"],))
    return True, "Verified.", row["mode"]


def invalidate_outstanding(user_id: str) -> None:
    """Cancel any live code, e.g. when delivery failed or access was revoked."""
    init()
    with _connect() as conn:
        conn.execute(
            "UPDATE login_codes SET state = 'cancelled' WHERE user_id = ? AND state = 'live';",
            (user_id,),
        )


def purge_history() -> int:
    """Drop code hashes older than the retention window."""
    init()
    cutoff = _iso(_now() - timedelta(days=HISTORY_RETENTION_DAYS))
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM login_codes WHERE datetime(created_at) < datetime(?);", (cutoff,)
        )
        return cur.rowcount
