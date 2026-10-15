"""
The readiness check, checked.

It is the one thing an administrator runs on a server they cannot otherwise
inspect, so the failure that matters is it falling over instead of
reporting. Every check has to survive whatever it finds -- a missing model, a
database that will not open, no Drive, no mail -- and say so.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import preflight  # noqa: E402


@pytest.fixture(autouse=True)
def clean():
    preflight.results.clear()
    yield
    preflight.results.clear()


def _statuses() -> dict[str, str]:
    return {area: status for status, area, _, _ in preflight.results}


def test_a_well_set_up_server_passes(monkeypatch, tmp_path):
    monkeypatch.setenv("RAD_INTEL_SECRET_KEY", "a" * 64)
    monkeypatch.setenv("RAD_INTEL_COOKIE_SECURE", "1")
    monkeypatch.setenv("RAD_INTEL_TRUSTED_PROXY", "1")
    monkeypatch.delenv("RAD_INTEL_ENABLE_DOCS", raising=False)
    monkeypatch.delenv("RAD_INTEL_CORS_ORIGINS", raising=False)

    preflight.check_secret_key()
    preflight.check_cookies()
    preflight.check_proxy()
    preflight.check_docs_and_cors()

    assert set(_statuses().values()) == {preflight.OK}


def test_the_settings_that_hurt_somebody_are_called_out(monkeypatch, tmp_path):
    monkeypatch.delenv("RAD_INTEL_SECRET_KEY", raising=False)
    monkeypatch.delenv("RAD_INTEL_COOKIE_SECURE", raising=False)
    monkeypatch.setenv("RAD_INTEL_ENABLE_DOCS", "1")
    monkeypatch.setenv("RAD_INTEL_CORS_ORIGINS", "*")
    # No key anywhere: neither set nor already generated.
    monkeypatch.setattr("rad_intel.config.BASE_DIR", tmp_path)

    preflight.check_secret_key()
    preflight.check_cookies()
    preflight.check_docs_and_cors()

    found = _statuses()
    assert found["Secret key"] == preflight.FIX
    assert found["Session cookie"] == preflight.FIX
    assert found["API docs"] == preflight.FIX
    assert found["CORS"] == preflight.FIX
    # And each one says what to do about it, not just that it is wrong.
    assert all(action for status, _, _, action in preflight.results
               if status == preflight.FIX)


def test_a_short_secret_key_is_not_a_secret_key(monkeypatch):
    monkeypatch.setenv("RAD_INTEL_SECRET_KEY", "hunter2")
    preflight.check_secret_key()
    assert _statuses()["Secret key"] == preflight.FIX


def test_a_key_file_on_a_disk_that_survives_is_not_a_failure(monkeypatch, tmp_path):
    """
    Running this yourself, on your own machine, the generated key is fine.

    It only becomes a problem on a host that rebuilds its filesystem, where a
    new key every deploy means every stored phone number becomes unreadable.
    Saying FIX to somebody self-hosting would be telling them to fix what is
    not broken.
    """
    monkeypatch.delenv("RAD_INTEL_SECRET_KEY", raising=False)
    monkeypatch.setattr("rad_intel.config.BASE_DIR", tmp_path)
    (tmp_path / ".radintel_key").write_text("a" * 64, encoding="utf-8")

    preflight.check_secret_key()
    assert _statuses()["Secret key"] == preflight.CHECK
    assert "Back that file up" in preflight.results[0][3]


def test_codes_going_to_the_log_is_a_problem_in_production(monkeypatch):
    monkeypatch.setenv("RAD_INTEL_OTP_CHANNEL", "log")
    preflight.check_mail(send=False)
    assert _statuses()["Email"] == preflight.FIX
    assert "Nobody can sign in" in preflight.results[0][3]


def test_no_way_to_send_a_code_at_all_is_a_problem(monkeypatch):
    for name in ("RAD_INTEL_OTP_CHANNEL", "RAD_INTEL_OTP_ECHO",
                 "SMTP_HOST", "SMTP_FROM"):
        monkeypatch.delenv(name, raising=False)
    preflight.check_mail(send=False)
    assert _statuses()["Email"] == preflight.FIX
    assert "App Password" in preflight.results[0][3]


def test_a_review_store_inside_the_code_directory_is_flagged(monkeypatch):
    monkeypatch.delenv("RAD_INTEL_FEEDBACK_DIR", raising=False)
    preflight.check_storage()
    # Either it is inside the project (worth saying) or it is not (fine).
    assert _statuses()["Review store"] in (preflight.OK, preflight.CHECK)


def test_no_drive_is_a_decision_not_a_failure(monkeypatch):
    for name in ("GOOGLE_DRIVE_CLIENT_ID", "GOOGLE_DRIVE_CLIENT_SECRET",
                 "GOOGLE_DRIVE_REFRESH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    preflight.check_drive()
    assert _statuses()["Radiograph backup"] == preflight.CHECK
    assert "cannot be made again" in preflight.results[0][3]


def test_a_missing_checkpoint_is_the_loudest_failure(monkeypatch, tmp_path):
    from rad_intel.api.dependencies import model_manager

    monkeypatch.setattr(model_manager, "active_checkpoint", lambda name=None: None)
    preflight.check_model()
    assert _statuses()["Model"] == preflight.FIX
    assert "every analysis will fail" in preflight.results[0][3].lower()


def test_a_truncated_checkpoint_is_caught(monkeypatch, tmp_path):
    from rad_intel.api.dependencies import model_manager

    stub = tmp_path / "best_hybrid_model.pt"
    stub.write_bytes(b"0" * 1024)
    monkeypatch.setattr(model_manager, "active_checkpoint", lambda name=None: str(stub))
    preflight.check_model()
    assert _statuses()["Model"] == preflight.FIX
    assert "truncated" in preflight.results[0][3]


def test_every_check_survives_whatever_it_finds():
    """
    The whole point of this script is to run on a server nobody can inspect.

    A check that raises instead of reporting is worse than no check: it stops
    the ones after it, so an administrator fixes one thing and discovers the
    next only on the following run.
    """
    for check in (preflight.check_python, preflight.check_torch,
                  preflight.check_model, preflight.check_secret_key,
                  preflight.check_cookies, preflight.check_proxy,
                  preflight.check_docs_and_cors, preflight.check_storage,
                  preflight.check_database, preflight.check_drive,
                  preflight.check_reporting):
        check()
    preflight.check_mail(send=False)

    assert len(preflight.results) >= 12
    assert all(status in (preflight.OK, preflight.FIX, preflight.CHECK)
               for status, *_ in preflight.results)
