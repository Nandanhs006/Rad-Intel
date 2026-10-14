"""
Access control for the closed review workspace.

The property worth protecting is that nobody who has not been given an
account by an administrator can read or write review data, and that
revoking an account takes effect at once rather than when a token expires.
"""

import pathlib
import tempfile

import pytest

from rad_intel.auth import accounts, session


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch):
    """Point the account store at a scratch database for every test."""
    path = pathlib.Path(tempfile.mkdtemp()) / "auth_test.db"
    monkeypatch.setattr(accounts, "DB_PATH", path)
    accounts.init()
    return path


def _make(email="doc@hospital.org", password="correct-horse-battery"):
    return accounts.create_clinician(email, "Dr Example", "Example Hospital", password)


# --- passwords -------------------------------------------------------------

def test_password_is_not_stored_in_the_clear():
    user = _make(password="correct-horse-battery")
    assert "correct-horse-battery" not in user["password_hash"]
    assert user["password_hash"] != user["password_salt"]
    assert len(user["password_hash"]) == 128  # 64 bytes, hex


def test_each_account_gets_its_own_salt():
    a = _make("a@h.org", "same-password-here")
    b = _make("b@h.org", "same-password-here")
    assert a["password_salt"] != b["password_salt"]
    assert a["password_hash"] != b["password_hash"], "identical passwords must not collide"


def test_authentication_accepts_only_the_right_password():
    _make(password="correct-horse-battery")
    assert accounts.authenticate("doc@hospital.org", "correct-horse-battery")
    assert accounts.authenticate("doc@hospital.org", "wrong") is None
    assert accounts.authenticate("doc@hospital.org", "") is None


def test_unknown_account_is_rejected():
    assert accounts.authenticate("nobody@nowhere.io", "anything") is None


def test_duplicate_email_is_refused():
    _make()
    with pytest.raises(ValueError):
        _make()


# --- revocation ------------------------------------------------------------

def test_disabled_account_cannot_authenticate():
    _make()
    accounts.set_status("doc@hospital.org", "disabled")
    assert accounts.authenticate("doc@hospital.org", "correct-horse-battery") is None
    accounts.set_status("doc@hospital.org", "active")
    assert accounts.authenticate("doc@hospital.org", "correct-horse-battery")


def test_password_change_requires_the_current_one():
    user = _make()
    ok, _ = accounts.change_own_password(user["id"], "wrong", "a-new-long-password")
    assert not ok
    ok, _ = accounts.change_own_password(user["id"], "correct-horse-battery", "short")
    assert not ok, "a short password must be refused"
    ok, _ = accounts.change_own_password(user["id"], "correct-horse-battery", "a-new-long-password")
    assert ok
    assert accounts.authenticate("doc@hospital.org", "a-new-long-password")
    assert accounts.authenticate("doc@hospital.org", "correct-horse-battery") is None


# --- sessions --------------------------------------------------------------

def test_session_round_trips():
    assert session.verify(session.issue("user-123")) == "user-123"


def test_tampered_session_is_rejected():
    token = session.issue("user-123")
    assert session.verify(token[:-4] + "0000") is None
    assert session.verify("user-999." + token.split(".", 1)[1]) is None
    assert session.verify("nonsense") is None
    assert session.verify(None) is None


def test_expired_session_is_rejected():
    assert session.verify(session.issue("user-123", ttl=-1)) is None


# --- access requests -------------------------------------------------------

def test_access_request_creates_no_account():
    accounts.create_access_request("Dr Hopeful", "hope@clinic.org", "Some Clinic")
    assert accounts.get_by_email("hope@clinic.org") is None, \
        "requesting access must never grant it"
    assert len(accounts.list_access_requests()) == 1
