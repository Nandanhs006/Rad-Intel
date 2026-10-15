"""
Access control for the closed review workspace.

The properties worth protecting, and what each test here holds to:

* Nobody who has not been given an account by an administrator can read or
  write review data.
* Neither factor opens a session on its own. A Rad-Intel ID read off a
  notepad, or an administrator's password, is not enough without the code
  sent to the registered phone.
* A code works once, expires, and is never issued twice to the same person.
* Revoking an account takes effect at once rather than when a session expires.
* Nothing recoverable is stored: no password, no Rad-Intel ID and no
  verification code sits in the database in a form that can be replayed.
"""

import pathlib
import tempfile

import pytest

from rad_intel.auth import accounts, activity, crypto, otp, session, store


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch):
    """Point every store at one scratch database for the duration of a test."""
    path = pathlib.Path(tempfile.mkdtemp()) / "auth_test.db"
    store.forget()
    for module in (accounts, session, otp, activity):
        monkeypatch.setattr(module, "DB_PATH", path)
    accounts.init()
    session.init()
    otp.init()
    activity.init()
    yield path
    activity.flush(2.0)


def _clinician(email="doc@hospital.org", phone="+91 98765 43210"):
    return accounts.create_clinician(email, "Dr Example", "Example Hospital",
                                     phone=phone)


def _admin(email="boss@study.org", password="correct-horse-battery"):
    return accounts.create_clinician(email, "Study Admin", "Example University",
                                     password=password, role="admin",
                                     phone="+91 98440 00000")


# --- what is stored --------------------------------------------------------

def test_password_is_not_stored_in_the_clear():
    user = _admin(password="correct-horse-battery")
    assert "correct-horse-battery" not in user["password_hash"]
    assert user["password_hash"] != user["password_salt"]
    assert len(user["password_hash"]) == 128  # 64 bytes, hex


def test_each_account_gets_its_own_salt():
    a = _admin("a@h.org", "same-password-here")
    b = _admin("b@h.org", "same-password-here")
    assert a["password_salt"] != b["password_salt"]
    assert a["password_hash"] != b["password_hash"], "identical passwords must not collide"


def test_radintel_id_and_phone_are_not_stored_in_the_clear():
    user = _clinician()
    assert user["radintel_id"] not in (user["radintel_id_enc"] or "")
    assert "98765" not in (user["phone_enc"] or "")
    # ...and both come back correctly for the administrator who needs them.
    reloaded = accounts.get_by_email("doc@hospital.org")
    assert reloaded["radintel_id"] == user["radintel_id"]
    assert reloaded["phone"] == "+91 98765 43210"


def test_the_same_phone_number_encrypts_differently_each_time():
    """A fresh nonce per value, so the store cannot be mined for duplicates."""
    first = crypto.encrypt_field("+91 98765 43210")
    second = crypto.encrypt_field("+91 98765 43210")
    assert first != second
    assert crypto.decrypt_field(first) == crypto.decrypt_field(second) == "+91 98765 43210"


def test_a_tampered_encrypted_field_does_not_decrypt_to_a_guess():
    blob = crypto.encrypt_field("+91 98765 43210")
    mangled = blob[:-6] + ("A" if blob[-6] != "A" else "B") + blob[-5:]
    assert crypto.decrypt_field(mangled) is None


# --- Rad-Intel IDs ---------------------------------------------------------

def test_generated_ids_start_with_rad_and_are_unique():
    issued = {accounts.generate_radintel_id() for _ in range(40)}
    assert len(issued) == 40
    assert all(i.startswith("RAD") and len(i) == 12 for i in issued)
    assert all(i.isalnum() and i.isupper() for i in issued)


def test_ids_avoid_characters_that_are_misread():
    for _ in range(50):
        assert not set(accounts.generate_radintel_id()[3:]) & set("O0I1L")


def test_an_id_can_be_typed_with_spaces_or_lower_case():
    user = _clinician()
    spaced = " ".join([user["radintel_id"][:4], user["radintel_id"][4:]]).lower()
    assert accounts.identify_clinician("doc@hospital.org", spaced)


def test_rotating_an_id_invalidates_the_old_one():
    user = _clinician()
    old = user["radintel_id"]
    new = accounts.rotate_radintel_id(user["id"])
    assert new != old
    assert accounts.identify_clinician("doc@hospital.org", old) is None
    assert accounts.identify_clinician("doc@hospital.org", new)


# --- first factors ---------------------------------------------------------

def test_clinician_needs_the_right_email_and_id_together():
    user = _clinician()
    other = _clinician("two@hospital.org", "+91 90000 00000")
    assert accounts.identify_clinician("doc@hospital.org", user["radintel_id"])
    assert accounts.identify_clinician("doc@hospital.org", other["radintel_id"]) is None, \
        "another clinician's ID must not work against this email"
    assert accounts.identify_clinician("doc@hospital.org", "RADZZZZZZZZZ") is None
    assert accounts.identify_clinician("nobody@nowhere.io", user["radintel_id"]) is None


def test_clinician_accounts_have_no_usable_password():
    """
    A clinician's first factor is their ID, so the stored password hash must
    not be guessable -- including as the empty string, which is what a
    passwordless account would hash to if nothing were stored.
    """
    _clinician()
    for guess in ("", " ", "password", "doc@hospital.org"):
        assert accounts.identify_admin("doc@hospital.org", guess) is None


def test_admin_identification_requires_the_admin_role():
    _clinician()
    admin = _admin()
    assert accounts.identify_admin("boss@study.org", "correct-horse-battery")
    assert accounts.identify_admin("boss@study.org", "wrong") is None
    assert admin["role"] == "admin"
    # A clinician account can never pass the administrator route, whatever is
    # guessed, because the role is checked before the password.
    assert accounts.identify_admin("doc@hospital.org", "anything") is None


def test_duplicate_email_is_refused():
    _clinician()
    with pytest.raises(ValueError):
        _clinician()


def test_duplicate_radintel_id_is_refused():
    user = _clinician()
    with pytest.raises(ValueError):
        accounts.create_clinician("other@h.org", "Dr Two", "Other Hospital",
                                  phone="+91 90000 00000",
                                  radintel_id=user["radintel_id"])


# --- verification codes ----------------------------------------------------

def _past_cooldown() -> None:
    """
    Age every issued code past the resend cooldown.

    Tests about what a code *is* -- single use, never repeated, superseded by
    the next one -- have to ask for two in a row, which real use does not.
    They step over the cooldown rather than switching off the rate limiting
    they are not the subject of.
    """
    import sqlite3

    with sqlite3.connect(otp.DB_PATH) as conn:
        conn.execute("UPDATE login_codes SET created_at = datetime('now', '-70 seconds');")

def test_a_code_is_twelve_characters_from_the_specified_alphabet():
    code = otp.generate_code()
    assert len(code) == 12
    assert all(c in otp.CODE_ALPHABET for c in code)
    assert otp.generate_code() != otp.generate_code(), "codes must not be predictable"
    # Letters, digits and symbols, as specified -- and nothing that would not
    # survive being copied out of a chat app.
    assert any(c.isalpha() for c in otp.CODE_ALPHABET)
    assert any(c.isdigit() for c in otp.CODE_ALPHABET)
    assert any(not c.isalnum() for c in otp.CODE_ALPHABET)
    assert not set(otp.CODE_ALPHABET) & set("\'\" \\")


def test_a_code_verifies_once_and_then_is_spent():
    user = _clinician()
    code, _ = otp.issue_code(user["id"])
    assert otp.verify_code(user["id"], code)[0]
    ok, message, _ = otp.verify_code(user["id"], code)
    assert not ok and "request a new one" in message.lower()


def test_a_code_is_not_stored_in_a_replayable_form():
    user = _clinician()
    code, _ = otp.issue_code(user["id"])
    import sqlite3
    with sqlite3.connect(otp.DB_PATH) as conn:
        stored = conn.execute("SELECT code_hash FROM login_codes;").fetchall()
    assert stored and all(code not in row[0] for row in stored)


def test_a_wrong_code_is_rejected_and_the_attempts_run_out():
    user = _clinician()
    code, _ = otp.issue_code(user["id"])
    for _ in range(otp.MAX_ATTEMPTS):
        assert not otp.verify_code(user["id"], "wrong-code12")[0]
    ok, message, _ = otp.verify_code(user["id"], code)
    assert not ok, "the real code must stop working once the attempts are used up"
    assert "request a new" in message.lower()


def test_an_expired_code_is_rejected():
    user = _clinician()
    code, _ = otp.issue_code(user["id"])
    import sqlite3
    with sqlite3.connect(otp.DB_PATH) as conn:
        conn.execute("UPDATE login_codes SET expires_at = '2020-01-01T00:00:00+00:00';")
    ok, message, _ = otp.verify_code(user["id"], code)
    assert not ok and "expired" in message.lower()


def test_requesting_a_new_code_cancels_the_previous_one():
    """Two live codes in a phone's message list would be two chances to guess."""
    user = _clinician()
    first, _ = otp.issue_code(user["id"])
    _past_cooldown()
    second, _ = otp.issue_code(user["id"])
    assert not otp.verify_code(user["id"], first)[0]
    assert otp.verify_code(user["id"], second)[0]


def test_no_code_is_ever_issued_twice_to_the_same_account(monkeypatch):
    """
    The guarantee is enforced, not assumed.

    With the generator pinned to one value, a second request must not hand
    back the code that account has already had.
    """
    user = _clinician()
    monkeypatch.setattr(otp, "generate_code", lambda: "Fixed-Code1!")
    first, _ = otp.issue_code(user["id"])
    assert first == "Fixed-Code1!"
    _past_cooldown()
    second, message = otp.issue_code(user["id"])
    assert second is None, "a repeat of a previously issued code must be refused"
    assert "could not generate" in message.lower()


def test_code_requests_are_rate_limited():
    user = _clinician()
    for _ in range(otp.MAX_REQUESTS_PER_WINDOW):
        code, _ = otp.issue_code(user["id"])
        assert code
        _past_cooldown()
    code, message = otp.issue_code(user["id"])
    assert code is None
    assert "too many" in message.lower()
    assert otp.seconds_until_next_allowed(user["id"]) > 0


def test_codes_are_bound_to_one_account():
    first = _clinician()
    second = _clinician("two@hospital.org", "+91 90000 00000")
    code, _ = otp.issue_code(first["id"])
    assert not otp.verify_code(second["id"], code)[0], \
        "a code issued to one clinician must not verify for another"
    assert otp.verify_code(first["id"], code)[0]


# --- sessions --------------------------------------------------------------

def test_session_round_trips():
    assert session.verify(session.issue("user-123")) == "user-123"


def test_an_unknown_or_tampered_token_is_rejected():
    token = session.issue("user-123")
    assert session.verify(token[:-4] + "0000") is None
    assert session.verify("nonsense") is None
    assert session.verify(None) is None
    assert session.verify("") is None
    assert session.verify(token) == "user-123", "the real token still works"


def test_a_session_token_is_not_stored_in_the_clear():
    token = session.issue("user-123")
    import sqlite3
    with sqlite3.connect(session.DB_PATH) as conn:
        rows = conn.execute("SELECT token_hash FROM sessions;").fetchall()
    assert rows and all(token not in row[0] for row in rows)


def test_expired_session_is_rejected():
    assert session.verify(session.issue("user-123", ttl=-1)) is None


def test_an_idle_session_stays_open():
    """
    No idle timeout: a clinician who reviews a handful of cases a week should
    not have to find a code every time. The thirty-day ceiling bounds it.
    """
    token = session.issue("user-123")
    import sqlite3
    with sqlite3.connect(session.DB_PATH) as conn:
        conn.execute("UPDATE sessions SET last_seen_at = '2020-01-01T00:00:00+00:00';")
    assert session.verify(token) == "user-123"



def test_a_session_ends_when_the_browser_does():
    """
    No Max-Age on the cookie is what actually ends it, which is the
    browser's business. The server keeps a ceiling for the case the browser
    does not honour -- one told to restore its tabs restores session cookies
    with them -- and twelve hours is short enough to matter.
    """
    import sqlite3
    from datetime import datetime, timedelta, timezone

    assert session.SESSION_TTL_SECONDS == 12 * 60 * 60
    token = session.issue("user-123")
    with sqlite3.connect(session.DB_PATH) as conn:
        (expires,) = conn.execute("SELECT expires_at FROM sessions;").fetchone()
    left = (datetime.fromisoformat(expires) - datetime.now(timezone.utc)).total_seconds()
    assert timedelta(hours=11).total_seconds() < left <= timedelta(hours=12).total_seconds()
    assert session.verify(token) == "user-123"

def test_expiry_does_not_slide_forward_with_use():
    """Thirty days from the sign-in, not thirty days from the last request."""
    import sqlite3

    token = session.issue("user-123")
    with sqlite3.connect(session.DB_PATH) as conn:
        (before,) = conn.execute("SELECT expires_at FROM sessions;").fetchone()
    for _ in range(3):
        assert session.verify(token) == "user-123"
    with sqlite3.connect(session.DB_PATH) as conn:
        (after,) = conn.execute("SELECT expires_at FROM sessions;").fetchone()
    assert before == after


def test_a_session_past_the_ceiling_is_gone():
    import sqlite3

    token = session.issue("user-123")
    with sqlite3.connect(session.DB_PATH) as conn:
        conn.execute("UPDATE sessions SET expires_at = '2020-01-01T00:00:00+00:00';")
    assert session.verify(token) is None
    with sqlite3.connect(session.DB_PATH) as conn:
        (left,) = conn.execute("SELECT COUNT(*) FROM sessions;").fetchone()
    assert left == 0, "an expired session must not be left lying in the table"


# --- one account, one device -----------------------------------------------

def test_logging_in_elsewhere_closes_the_session_that_was_open():
    laptop = session.issue("user-123")
    assert session.verify(laptop) == "user-123"

    phone = session.issue("user-123")
    assert session.verify(phone) == "user-123"
    assert session.verify(laptop) is None, "the old device must be signed out"
    assert len(session.active_for_user("user-123")) == 1


def test_one_account_being_displaced_does_not_touch_another():
    theirs = session.issue("user-456")
    session.issue("user-123")
    session.issue("user-123")
    assert session.verify(theirs) == "user-456"


def test_what_was_open_can_be_read_before_it_is_displaced():
    """
    The sign-in route reads this first, so the person can be told they have
    just signed something else out rather than finding out when their other
    machine stops working.
    """
    assert session.active_for_user("user-123") == []
    session.issue("user-123")
    assert len(session.active_for_user("user-123")) == 1
    session.issue("user-123")
    assert len(session.active_for_user("user-123")) == 1


def test_logging_out_means_a_new_code_is_needed():
    token = session.issue("user-123")
    session.destroy(token)
    assert session.verify(token) is None
    assert session.active_for_user("user-123") == []


def test_using_a_session_keeps_it_alive():
    token = session.issue("user-123")
    before = session.list_active()[0]["last_seen_at"]
    assert session.verify(token) == "user-123"
    assert session.list_active()[0]["last_seen_at"] >= before


def test_revoking_an_account_closes_its_session_and_nobody_elses():
    session.issue("user-123")
    other = session.issue("user-456")
    assert session.destroy_all_for_user("user-123") == 1
    assert session.verify(other) == "user-456"


def test_an_administrator_can_close_one_session():
    token = session.issue("user-123")
    row = session.list_active()[0]
    assert session.destroy_by_id(row["id"])
    assert session.verify(token) is None
    assert not session.destroy_by_id(row["id"])


# --- revocation ------------------------------------------------------------

def test_disabled_account_cannot_be_identified():
    user = _clinician()
    accounts.set_status("doc@hospital.org", "disabled")
    assert accounts.identify_clinician("doc@hospital.org", user["radintel_id"]) is None
    accounts.set_status("doc@hospital.org", "active")
    assert accounts.identify_clinician("doc@hospital.org", user["radintel_id"])


def test_deleting_an_account_removes_it():
    user = _clinician()
    assert accounts.delete_clinician(user["id"])
    assert accounts.get_by_email("doc@hospital.org") is None
    assert accounts.identify_clinician("doc@hospital.org", user["radintel_id"]) is None


def test_password_change_requires_the_current_one():
    user = _admin()
    ok, _ = accounts.change_own_password(user["id"], "wrong", "a-new-long-password")
    assert not ok
    ok, _ = accounts.change_own_password(user["id"], "correct-horse-battery", "short")
    assert not ok, "a short password must be refused"
    ok, _ = accounts.change_own_password(user["id"], "correct-horse-battery",
                                         "a-new-long-password")
    assert ok
    assert accounts.identify_admin("boss@study.org", "a-new-long-password")
    assert accounts.identify_admin("boss@study.org", "correct-horse-battery") is None


# --- what other clinicians can see -----------------------------------------

def test_the_participant_roster_carries_no_contact_details():
    user = _clinician()
    visible = accounts.public_view(user)
    assert set(visible) == {"full_name", "institution", "role", "status"}
    serialised = str(visible)
    assert user["radintel_id"] not in serialised
    assert "98765" not in serialised
    assert "doc@hospital.org" not in serialised


# --- search ----------------------------------------------------------------

def test_administrator_search_finds_by_name_email_and_id():
    user = _clinician()
    _clinician("other@clinic.org", "+91 90000 00000")
    assert len(accounts.search_clinicians()) == 2
    assert [u["email"] for u in accounts.search_clinicians("other@")] == ["other@clinic.org"]
    found = accounts.search_clinicians(user["radintel_id"])
    assert [u["id"] for u in found] == [user["id"]]
    assert accounts.search_clinicians("nobody at all") == []


# --- access requests -------------------------------------------------------

def test_access_request_creates_no_account():
    accounts.create_access_request("Dr Hopeful", "hope@clinic.org", "Some Clinic",
                                   phone="+91 99999 11111")
    assert accounts.get_by_email("hope@clinic.org") is None, \
        "requesting access must never grant it"
    rows = accounts.list_access_requests()
    assert len(rows) == 1
    assert rows[0]["phone"] == "+91 99999 11111"


def test_an_applicants_phone_number_is_encrypted_too():
    accounts.create_access_request("Dr Hopeful", "hope@clinic.org", "Some Clinic",
                                   phone="+91 99999 11111")
    import sqlite3
    with sqlite3.connect(accounts.DB_PATH) as conn:
        (blob,) = conn.execute("SELECT phone_enc FROM access_requests;").fetchone()
    assert "99999" not in blob


# --- activity log ----------------------------------------------------------

def test_activity_is_recorded_and_readable_in_order():
    user = _clinician()
    activity.record("login.success", user=user, ip="127.0.0.1")
    activity.record("review.submitted", user=user, detail="abc12345")
    entries = activity.recent()
    assert [e["action"] for e in entries] == ["review.submitted", "login.success"]
    assert entries[0]["actor"] == "doc@hospital.org"
    assert activity.count() == 2


def test_a_failed_attempt_is_attributed_even_with_no_account():
    activity.record("login.identify_failed", actor="stranger@nowhere.io",
                    ip="10.0.0.9", outcome="refused")
    entry = activity.recent()[0]
    assert entry["actor"] == "stranger@nowhere.io"
    assert entry["outcome"] == "refused"
    assert entry["user_id"] is None


def test_logging_never_breaks_a_request(monkeypatch):
    """A broken log must not be able to fail the action it was recording."""
    monkeypatch.setattr(activity, "_connect", lambda: (_ for _ in ()).throw(RuntimeError()))
    activity.record("review.submitted", actor="someone@x.org")  # must not raise


def test_an_entry_is_written_to_the_database_it_was_made_against(monkeypatch, tmp_path):
    """
    The writer runs on its own thread and may drain an entry after the test
    that made it has finished and put DB_PATH back. Binding the destination
    at the point the entry is made is what stops a test's rows landing in the
    real log, which is exactly what happened before this.
    """
    scratch = tmp_path / "scratch.db"
    store.forget()
    monkeypatch.setattr(activity, "DB_PATH", scratch)
    activity.init()
    activity.record("review.submitted", actor="someone@scratch.test")

    # Put the module back before the entry is allowed to be written, which is
    # what pytest's teardown does.
    monkeypatch.undo()
    activity.flush()

    import sqlite3
    with sqlite3.connect(scratch) as conn:
        actors = [r[0] for r in conn.execute("SELECT actor FROM activity_log;")]
    assert actors == ["someone@scratch.test"], "the entry must follow its own database"


# --- not making ordinary use feel like abuse --------------------------------

def test_pressing_send_again_straight_away_does_not_burn_the_allowance():
    """
    Impatience is not abuse. A second press within the cooldown is declined
    without minting anything and without counting, and says to look in the
    inbox rather than quoting a limit.
    """
    user = _clinician()
    first, _ = otp.issue_code(user["id"])
    assert first

    blocked, message = otp.issue_code(user["id"])
    assert blocked is None
    assert "check your inbox" in message.lower()
    assert "spam" in message.lower()
    assert otp.recent_request_count(user["id"]) == 1, \
        "a refused resend must not count against the allowance"


def test_the_cooldown_lets_go_after_a_minute():
    import sqlite3

    user = _clinician()
    otp.issue_code(user["id"])
    with sqlite3.connect(otp.DB_PATH) as conn:
        conn.execute("UPDATE login_codes SET created_at = '2020-01-01T00:00:00+00:00';")
    code, _ = otp.issue_code(user["id"])
    assert code, "once the cooldown has passed a new code must be issued"


def test_the_allowance_is_still_enforced():
    import sqlite3

    user = _clinician()
    for _ in range(otp.MAX_REQUESTS_PER_WINDOW):
        code, message = otp.issue_code(user["id"])
        assert code, message
        # Step past the cooldown without leaving the rate-limit window.
        with sqlite3.connect(otp.DB_PATH) as conn:
            conn.execute("UPDATE login_codes SET created_at = datetime('now', '-70 seconds')"
                         " WHERE state = 'live';")
    code, message = otp.issue_code(user["id"])
    assert code is None
    assert "too many" in message.lower()
    assert "administrator can issue a code" in message.lower(), \
        "somebody locked out needs to be told there is a way in"


def test_an_administrator_minted_code_ignores_both_limits():
    user = _clinician()
    for _ in range(otp.MAX_REQUESTS_PER_WINDOW + 4):
        code, message = otp.issue_code(user["id"], channel="console", rate_limited=False)
        assert code, message
