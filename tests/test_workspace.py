"""
The review workspace, through the HTTP surface.

The unit tests in test_auth.py hold the credential machinery. These hold the
rules the workspace is *for*, at the level a browser sees them:

* Every workspace endpoint refuses an unauthenticated caller.
* A clinician can correct and delete their own cases and nobody else's, and
  cannot change the image under a verdict somebody has already read.
* The participant roster carries no contact details.
* There is one training run at a time; it names its holder, only its holder or
  an administrator can cancel it, and it blocks exports but not uploads.
* The administrator console is closed to clinicians.
"""

import pathlib
import tempfile

import httpx
import pytest

from rad_intel.api import ratelimit
from rad_intel.api.main import app
from rad_intel.api.routes import feedback as feedback_routes
from rad_intel.api.training_job import TrainingJob
from rad_intel.auth import accounts, activity, otp, session, store
from rad_intel.storage.database import DatabaseManager, db_manager


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch, tmp_path):
    """A scratch database and a clean rate limiter for every test."""
    path = pathlib.Path(tempfile.mkdtemp()) / "workspace_test.db"
    store.forget()
    for module in (accounts, session, otp, activity):
        monkeypatch.setattr(module, "DB_PATH", path)
    scratch = DatabaseManager(path)
    monkeypatch.setattr(db_manager, "db_path", scratch.db_path)
    monkeypatch.setattr(db_manager, "_initialized", False)
    accounts.init()
    session.init()
    otp.init()
    activity.init()
    ratelimit.reset()
    # A fresh job coordinator, so a test cannot inherit a lock from another.
    monkeypatch.setattr(feedback_routes, "job", TrainingJob(tmp_path / "training.log"))
    yield path
    ratelimit.reset()


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://test")


def _signed_in(user: dict) -> dict[str, str]:
    """A cookie jar for an account, bypassing the two-step sign-in."""
    return {session.COOKIE_NAME: session.issue(user["id"])}


def _make_clinician(email="a@hospital.org", name="Dr A", site="A Hospital"):
    return accounts.create_clinician(email, name, site, phone="+91 90000 00001")


def _make_admin():
    return accounts.create_clinician("boss@study.org", "Study Admin", "MCE",
                                     password="a-long-enough-password", role="admin",
                                     phone="+91 98440 00000")


async def _queue_case(reviewer: dict, sha="a" * 64, verdict="correct"):
    """Write a review straight to the store, so tests need no model."""
    return await db_manager.save_feedback(
        reviewer=reviewer["full_name"],
        model_version="hybrid",
        preprocessing_version="clahe2.0-8x8_pil-bilinear-224_imagenet-norm",
        image_sha256=sha,
        predicted_label="PNEUMONIA",
        predicted_probability=0.87,
        threshold=0.5574,
        verdict=verdict,
        institution=reviewer["institution"],
        owner_id=reviewer["id"],
    )


# --- the gate --------------------------------------------------------------

WORKSPACE_ENDPOINTS = [
    ("get", "/api/v1/feedback"),
    ("get", "/api/v1/feedback/warning"),
    ("get", "/api/v1/feedback/participants"),
    ("get", "/api/v1/feedback/exports"),
    ("get", "/api/v1/feedback/storage"),
    ("get", "/api/v1/feedback/models"),
    ("get", "/api/v1/feedback/train/status"),
    ("get", "/api/v1/feedback/image/any-id"),
    ("post", "/api/v1/feedback/export"),
    ("post", "/api/v1/feedback/train"),
    ("post", "/api/v1/feedback/train/cancel"),
    ("post", "/api/v1/feedback/submit"),
    ("patch", "/api/v1/feedback/any-id"),
    ("delete", "/api/v1/feedback/any-id"),
    ("get", "/api/v1/admin/overview"),
    ("get", "/api/v1/admin/users"),
    ("get", "/api/v1/admin/new-id"),
    ("get", "/api/v1/admin/requests"),
    ("get", "/api/v1/admin/activity"),
    ("get", "/api/v1/admin/sessions"),
    ("post", "/api/v1/admin/users"),
    ("patch", "/api/v1/admin/users/any-id"),
    ("delete", "/api/v1/admin/users/any-id"),
]


@pytest.mark.parametrize("method,path", WORKSPACE_ENDPOINTS)
async def test_every_workspace_endpoint_refuses_an_anonymous_caller(method, path):
    async with _client() as client:
        res = await client.request(method, path)
    assert res.status_code == 401, f"{method.upper()} {path} answered {res.status_code}"


async def test_the_public_endpoints_stay_public():
    """Signing in must not be required to use the tool or to ask for access."""
    async with _client() as client:
        assert (await client.get("/api/v1/health")).status_code == 200
        assert (await client.get("/api/v1/assistant/status")).status_code == 200
        # A malformed access request is a 400, not a 401: it was allowed through.
        res = await client.post("/api/v1/auth/request-access", json={
            "full_name": "", "email": "", "institution": "", "phone": "",
        })
        assert res.status_code == 400


async def test_a_disabled_account_loses_access_at_once():
    user = _make_clinician()
    cookies = _signed_in(user)
    async with _client() as client:
        assert (await client.get("/api/v1/feedback", cookies=cookies)).status_code == 200
        accounts.set_status(user["email"], "disabled")
        assert (await client.get("/api/v1/feedback", cookies=cookies)).status_code == 401


# --- what a clinician may change -------------------------------------------

async def test_a_clinician_can_edit_their_own_case():
    user = _make_clinician()
    record = await _queue_case(user)
    async with _client() as client:
        res = await client.patch(f"/api/v1/feedback/{record}",
                                 json={"note": "heatmap sat over the clavicle"},
                                 cookies=_signed_in(user))
    assert res.status_code == 200
    assert res.json()["note"] == "heatmap sat over the clavicle"
    assert res.json()["editable"] is True


async def test_a_clinician_cannot_edit_somebody_elses_case():
    mine = _make_clinician()
    theirs = _make_clinician("b@clinic.org", "Dr B", "B Clinic")
    record = await _queue_case(theirs)
    async with _client() as client:
        res = await client.patch(f"/api/v1/feedback/{record}", json={"note": "no"},
                                 cookies=_signed_in(mine))
        listing = await client.get("/api/v1/feedback", cookies=_signed_in(mine))
    assert res.status_code == 403
    assert "Dr B" in res.json()["detail"]
    # Visible, but flagged as not the caller's to change.
    assert [i["editable"] for i in listing.json()["items"]] == [False]


async def test_a_clinician_cannot_delete_somebody_elses_case():
    mine = _make_clinician()
    theirs = _make_clinician("b@clinic.org", "Dr B", "B Clinic")
    record = await _queue_case(theirs)
    async with _client() as client:
        res = await client.delete(f"/api/v1/feedback/{record}", cookies=_signed_in(mine))
    assert res.status_code == 403
    assert await db_manager.get_feedback(record) is not None


async def test_an_administrator_can_edit_and_delete_anybody_s_case():
    admin = _make_admin()
    other = _make_clinician()
    record = await _queue_case(other)
    async with _client() as client:
        patched = await client.patch(f"/api/v1/feedback/{record}",
                                     json={"note": "corrected by the administrator"},
                                     cookies=_signed_in(admin))
        deleted = await client.delete(f"/api/v1/feedback/{record}",
                                      cookies=_signed_in(admin))
    assert patched.status_code == 200
    assert deleted.status_code == 200
    assert await db_manager.get_feedback(record) is None


async def test_an_exported_case_is_frozen_for_its_author():
    """The record has to keep matching what was actually trained on."""
    user = _make_clinician()
    record = await _queue_case(user)
    await db_manager.mark_feedback_exported([record], "reviewed-export-TEST")
    async with _client() as client:
        res = await client.patch(f"/api/v1/feedback/{record}", json={"note": "late edit"},
                                 cookies=_signed_in(user))
    assert res.status_code == 409
    assert "reviewed-export-TEST" in res.json()["detail"]


async def test_the_edit_endpoint_offers_no_way_to_change_the_image():
    """
    Changing the pixels under a verdict somebody else has read would make the
    record a lie, so there is no field for it rather than a check against it.
    """
    from rad_intel.api.schemas import FeedbackUpdateRequest

    assert set(FeedbackUpdateRequest.model_fields) == {
        "verdict", "corrected_label", "note", "status",
    }


async def test_the_mine_filter_returns_only_the_callers_cases():
    mine = _make_clinician()
    theirs = _make_clinician("b@clinic.org", "Dr B", "B Clinic")
    await _queue_case(mine, sha="1" * 64)
    await _queue_case(theirs, sha="2" * 64)
    async with _client() as client:
        everything = await client.get("/api/v1/feedback", cookies=_signed_in(mine))
        just_mine = await client.get("/api/v1/feedback?scope=mine",
                                     cookies=_signed_in(mine))
    assert len(everything.json()["items"]) == 2
    assert [i["reviewer"] for i in just_mine.json()["items"]] == ["Dr A"]


async def test_the_queue_can_be_sorted_by_reviewer_name():
    zoe = _make_clinician("z@clinic.org", "Dr Zoe", "Z Clinic")
    adam = _make_clinician("a2@clinic.org", "Dr Adam", "A Clinic")
    await _queue_case(zoe, sha="1" * 64)
    await _queue_case(adam, sha="2" * 64)
    async with _client() as client:
        recent = await client.get("/api/v1/feedback", cookies=_signed_in(zoe))
        by_name = await client.get("/api/v1/feedback?sort=name", cookies=_signed_in(zoe))
    assert [i["reviewer"] for i in recent.json()["items"]] == ["Dr Adam", "Dr Zoe"]
    assert [i["reviewer"] for i in by_name.json()["items"]] == ["Dr Adam", "Dr Zoe"]


async def test_an_unknown_sort_order_cannot_reach_the_query():
    """The sort parameter is mapped, not interpolated, so this is inert."""
    user = _make_clinician()
    await _queue_case(user)
    async with _client() as client:
        res = await client.get("/api/v1/feedback?sort=created_at;DROP TABLE clinician_feedback",
                               cookies=_signed_in(user))
    assert res.status_code == 200
    assert len(res.json()["items"]) == 1


async def test_deleting_a_case_keeps_an_image_another_case_still_uses():
    user = _make_clinician()
    first = await _queue_case(user, sha="c" * 64)
    await _queue_case(user, sha="c" * 64)
    async with _client() as client:
        res = await client.delete(f"/api/v1/feedback/{first}", cookies=_signed_in(user))
    assert res.status_code == 200
    assert res.json()["image_removed"] is False


# --- what other clinicians can see -----------------------------------------

async def test_the_participant_roster_has_names_and_sites_only():
    user = _make_clinician()
    _make_clinician("b@clinic.org", "Dr B", "B Clinic")
    async with _client() as client:
        res = await client.get("/api/v1/feedback/participants", cookies=_signed_in(user))
    body = res.text
    assert res.status_code == 200
    assert {p["full_name"] for p in res.json()["participants"]} == {"Dr A", "Dr B"}
    for leak in ("b@clinic.org", "90000", user["radintel_id"]):
        assert leak not in body, f"the roster must not carry {leak!r}"


async def test_a_clinician_cannot_read_their_own_record_through_the_admin_console():
    user = _make_clinician()
    async with _client() as client:
        res = await client.get("/api/v1/admin/users", cookies=_signed_in(user))
    assert res.status_code == 403


async def test_a_clinician_sees_their_own_details_and_nobody_elses():
    user = _make_clinician()
    _make_clinician("b@clinic.org", "Dr B", "B Clinic")
    async with _client() as client:
        res = await client.get("/api/v1/auth/me", cookies=_signed_in(user))
    body = res.json()
    assert body["radintel_id"] == user["radintel_id"]
    assert body["phone"] == "+91 90000 00001"
    assert "b@clinic.org" not in res.text


# --- the training run ------------------------------------------------------

async def test_only_one_run_at_a_time_and_it_names_its_holder(tmp_path):
    first = _make_clinician()
    second = _make_clinician("b@clinic.org", "Dr B", "B Clinic")
    feedback_routes.job.claim(first, tmp_path / "m.jsonl", cases=8, epochs=3)

    async with _client() as client:
        blocked = await client.post("/api/v1/feedback/train",
                                    data={"epochs": "3"}, cookies=_signed_in(second))
        status = await client.get("/api/v1/feedback/train/status",
                                  cookies=_signed_in(second))
    assert blocked.status_code == 409
    assert "Dr A" in blocked.json()["detail"]
    assert status.json()["holder_name"] == "Dr A"
    assert status.json()["is_holder"] is False
    assert status.json()["can_cancel"] is False


async def test_a_run_in_progress_blocks_an_export_but_not_an_upload(tmp_path):
    holder = _make_clinician()
    other = _make_clinician("b@clinic.org", "Dr B", "B Clinic")
    await _queue_case(other)
    feedback_routes.job.claim(holder, tmp_path / "m.jsonl", cases=8, epochs=3)

    async with _client() as client:
        export = await client.post("/api/v1/feedback/export", cookies=_signed_in(other))
        # Uploading and listing stay open: neither touches the manifest the
        # run already holds.
        listing = await client.get("/api/v1/feedback", cookies=_signed_in(other))
    assert export.status_code == 409
    assert "Dr A" in export.json()["detail"]
    assert listing.status_code == 200


async def test_only_the_holder_or_an_administrator_can_cancel(tmp_path):
    holder = _make_clinician()
    bystander = _make_clinician("b@clinic.org", "Dr B", "B Clinic")
    admin = _make_admin()
    feedback_routes.job.claim(holder, tmp_path / "m.jsonl", cases=8, epochs=3)

    async with _client() as client:
        refused = await client.post("/api/v1/feedback/train/cancel",
                                    cookies=_signed_in(bystander))
        assert refused.status_code == 403
        allowed = await client.post("/api/v1/feedback/train/cancel",
                                    cookies=_signed_in(admin))
    assert allowed.status_code == 200
    assert feedback_routes.job.running is False


async def test_cancelling_with_no_run_in_progress_is_a_conflict():
    user = _make_clinician()
    async with _client() as client:
        res = await client.post("/api/v1/feedback/train/cancel", cookies=_signed_in(user))
    assert res.status_code == 409


async def test_the_holders_poll_is_the_heartbeat(tmp_path):
    """A run whose starter has gone away must not hold the queue for ever."""
    holder = _make_clinician()
    bystander = _make_clinician("b@clinic.org", "Dr B", "B Clinic")
    job = feedback_routes.job
    job.claim(holder, tmp_path / "m.jsonl", cases=8, epochs=3)
    job._last_heartbeat = 0.0

    async with _client() as client:
        await client.get("/api/v1/feedback/train/status", cookies=_signed_in(bystander))
        assert job._last_heartbeat == 0.0, "somebody else's poll is not a heartbeat"
        await client.get("/api/v1/feedback/train/status", cookies=_signed_in(holder))
    assert job._last_heartbeat > 0.0


async def test_an_abandoned_run_is_cancelled_with_a_reason(tmp_path):
    holder = _make_clinician()
    job = feedback_routes.job
    job.claim(holder, tmp_path / "m.jsonl", cases=8, epochs=3)
    job.cancel(reason="the connection to the clinician who started it was lost",
               by="watchdog")
    state = job.snapshot()
    assert state["running"] is False
    assert state["cancelled_by"] == "watchdog"
    assert "connection" in state["cancel_reason"]


async def test_a_training_request_needs_a_sensible_epoch_count():
    user = _make_clinician()
    async with _client() as client:
        res = await client.post("/api/v1/feedback/train", data={"epochs": "500"},
                                cookies=_signed_in(user))
    assert res.status_code == 400
    assert "1 and 20" in res.json()["detail"]


async def test_training_on_a_handful_of_cases_is_refused_in_plain_language():
    """The message is read by a clinician, so it says why, not which check failed."""
    user = _make_clinician()
    export_dir = feedback_routes.EXPORT_DIR
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest = export_dir / "reviewed-export-99999999T000000Z.jsonl"
    manifest.write_text('{"image_path": "x", "label": "NORMAL"}\n', encoding="utf-8")
    try:
        async with _client() as client:
            res = await client.post("/api/v1/feedback/train", data={"epochs": "3"},
                                    cookies=_signed_in(user))
        assert res.status_code == 400
        detail = res.json()["detail"]
        assert "Only 1 reviewed radiograph" in detail
        assert "normal and pneumonia" in detail
    finally:
        manifest.unlink()


# --- the administrator console ---------------------------------------------

async def test_creating_an_account_returns_an_id_to_hand_over():
    admin = _make_admin()
    async with _client() as client:
        res = await client.post("/api/v1/admin/users", cookies=_signed_in(admin), json={
            "full_name": "Dr New", "email": "new@clinic.org",
            "institution": "New Clinic", "phone": "+91 91234 56789",
        })
    assert res.status_code == 201
    body = res.json()
    assert body["radintel_id"].startswith("RAD")
    assert body["radintel_id"] in body["detail"]
    assert accounts.identify_clinician("new@clinic.org", body["radintel_id"])


async def test_an_account_cannot_be_created_without_a_reachable_phone():
    admin = _make_admin()
    async with _client() as client:
        res = await client.post("/api/v1/admin/users", cookies=_signed_in(admin), json={
            "full_name": "Dr New", "email": "new@clinic.org",
            "institution": "New Clinic", "phone": "123",
        })
    assert res.status_code == 400
    assert "country code" in res.json()["detail"]


async def test_generated_ids_come_from_the_server():
    admin = _make_admin()
    async with _client() as client:
        first = await client.get("/api/v1/admin/new-id", cookies=_signed_in(admin))
        second = await client.get("/api/v1/admin/new-id", cookies=_signed_in(admin))
    assert first.json()["radintel_id"] != second.json()["radintel_id"]
    assert first.json()["radintel_id"].startswith("RAD")


async def test_the_administrator_cannot_lock_themselves_out():
    admin = _make_admin()
    cookies = _signed_in(admin)
    async with _client() as client:
        for body in ({"status": "disabled"}, {"role": "clinician"}):
            res = await client.patch(f"/api/v1/admin/users/{admin['id']}",
                                     json=body, cookies=cookies)
            assert res.status_code == 400, body
        deleted = await client.delete(f"/api/v1/admin/users/{admin['id']}", cookies=cookies)
    assert deleted.status_code == 400
    assert accounts.get_by_id(admin["id"])["status"] == "active"


async def test_disabling_an_account_closes_its_sessions_immediately():
    admin = _make_admin()
    victim = _make_clinician()
    victim_cookies = _signed_in(victim)
    async with _client() as client:
        assert (await client.get("/api/v1/feedback", cookies=victim_cookies)).status_code == 200
        await client.patch(f"/api/v1/admin/users/{victim['id']}",
                           json={"status": "disabled"}, cookies=_signed_in(admin))
        assert (await client.get("/api/v1/feedback", cookies=victim_cookies)).status_code == 401


async def test_rotating_an_id_closes_the_clinicians_sessions():
    admin = _make_admin()
    user = _make_clinician()
    cookies = _signed_in(user)
    async with _client() as client:
        res = await client.post(f"/api/v1/admin/users/{user['id']}/rotate-id",
                                cookies=_signed_in(admin))
        assert res.json()["radintel_id"] != user["radintel_id"]
        assert (await client.get("/api/v1/feedback", cookies=cookies)).status_code == 401


async def test_deleting_an_account_leaves_its_reviews_in_the_queue():
    """Other participants have already seen and trained on them."""
    admin = _make_admin()
    user = _make_clinician()
    record = await _queue_case(user)
    async with _client() as client:
        res = await client.delete(f"/api/v1/admin/users/{user['id']}",
                                  cookies=_signed_in(admin))
    assert res.status_code == 200
    assert "reviews remain" in res.json()["detail"]
    assert (await db_manager.get_feedback(record))["reviewer"] == "Dr A"


async def test_the_console_search_finds_a_clinician_by_id():
    admin = _make_admin()
    user = _make_clinician()
    async with _client() as client:
        by_id = await client.get(f"/api/v1/admin/users?q={user['radintel_id']}",
                                 cookies=_signed_in(admin))
        by_name = await client.get("/api/v1/admin/users?q=Dr A",
                                   cookies=_signed_in(admin))
    assert [u["email"] for u in by_id.json()["users"]] == ["a@hospital.org"]
    assert [u["email"] for u in by_name.json()["users"]] == ["a@hospital.org"]


async def test_the_console_says_how_codes_and_email_are_delivered():
    admin = _make_admin()
    async with _client() as client:
        res = await client.get("/api/v1/admin/overview", cookies=_signed_in(admin))
    body = res.json()
    assert set(body["delivery"]) >= {"code_channel", "sms_configured", "email_configured"}
    assert body["users"]["admins"] == 1
    assert "contact the administrator" in body["contact_note"]


async def test_the_activity_log_is_administrator_only_and_records_actions():
    admin = _make_admin()
    user = _make_clinician()
    record = await _queue_case(user)
    async with _client() as client:
        await client.patch(f"/api/v1/feedback/{record}", json={"note": "x"},
                           cookies=_signed_in(user))
        refused = await client.get("/api/v1/admin/activity", cookies=_signed_in(user))
        allowed = await client.get("/api/v1/admin/activity", cookies=_signed_in(admin))
    assert refused.status_code == 403
    assert "review.edited" in [e["action"] for e in allowed.json()["entries"]]


async def test_open_sessions_are_listed_for_the_administrator():
    admin = _make_admin()
    user = _make_clinician()
    _signed_in(user)
    async with _client() as client:
        res = await client.get("/api/v1/admin/sessions", cookies=_signed_in(admin))
    rows = res.json()["sessions"]
    assert {r["full_name"] for r in rows} == {"Dr A", "Study Admin"}
    assert sum(1 for r in rows if r["is_you"]) == 1


# --- sign-in, end to end ---------------------------------------------------

async def test_a_clinician_cannot_sign_in_without_the_code(monkeypatch):
    user = _make_clinician()
    monkeypatch.setenv("RAD_INTEL_OTP_ECHO", "1")
    async with _client() as client:
        # Step one succeeds and sets no cookie.
        first = await client.post("/api/v1/auth/code", json={
            "email": user["email"], "radintel_id": user["radintel_id"],
        })
        assert first.status_code == 200
        assert session.COOKIE_NAME not in first.cookies
        # Step two refuses a guess, and still sets no cookie.
        second = await client.post("/api/v1/auth/login", json={
            "email": user["email"], "code": "guess-a-code",
        })
        assert second.status_code == 401
        assert session.COOKIE_NAME not in second.cookies
        assert (await client.get("/api/v1/feedback")).status_code == 401


async def test_the_whole_sign_in_works_with_the_real_code(monkeypatch):
    user = _make_clinician()
    issued: list[str] = []
    monkeypatch.setattr(
        "rad_intel.api.routes.auth.notify.send_verification_code",
        lambda phone, code, name: (issued.append(code), (True, "test", "sent"))[1],
    )
    async with _client() as client:
        await client.post("/api/v1/auth/code", json={
            "email": user["email"], "radintel_id": user["radintel_id"],
        })
        res = await client.post("/api/v1/auth/login",
                                json={"email": user["email"], "code": issued[0]})
        assert res.status_code == 200
        assert res.json()["full_name"] == "Dr A"
        assert (await client.get("/api/v1/feedback")).status_code == 200
        await client.post("/api/v1/auth/logout")
        assert (await client.get("/api/v1/feedback")).status_code == 401


async def test_the_session_cookie_expires_with_the_browser(monkeypatch):
    """No Max-Age and no Expires, so closing the browser ends the session."""
    user = _make_clinician()
    monkeypatch.setattr(
        "rad_intel.api.routes.auth.notify.send_verification_code",
        lambda phone, code, name: (True, "test", "sent"),
    )
    monkeypatch.setattr(otp, "generate_code", lambda: "Known-Code1!")
    async with _client() as client:
        await client.post("/api/v1/auth/code", json={
            "email": user["email"], "radintel_id": user["radintel_id"],
        })
        res = await client.post("/api/v1/auth/login",
                                json={"email": user["email"], "code": "Known-Code1!"})
    header = res.headers["set-cookie"].lower()
    assert "httponly" in header
    assert "samesite=lax" in header
    assert "max-age" not in header and "expires" not in header


async def test_a_code_that_cannot_be_delivered_refuses_the_sign_in(monkeypatch):
    """Never fall back to showing the code: that is no second factor at all."""
    user = _make_clinician()
    monkeypatch.setattr(
        "rad_intel.api.routes.auth.notify.send_verification_code",
        lambda phone, code, name: (False, "none", "No message provider is configured."),
    )
    async with _client() as client:
        res = await client.post("/api/v1/auth/code", json={
            "email": user["email"], "radintel_id": user["radintel_id"],
        })
    assert res.status_code == 503
    assert "code" not in res.json()
    assert "provider" in res.json()["detail"]


async def test_a_failed_sign_in_says_the_same_thing_for_every_cause():
    _make_clinician()
    async with _client() as client:
        wrong_id = await client.post("/api/v1/auth/code", json={
            "email": "a@hospital.org", "radintel_id": "RADZZZZZZZZZ",
        })
        no_account = await client.post("/api/v1/auth/code", json={
            "email": "nobody@nowhere.io", "radintel_id": "RADZZZZZZZZZ",
        })
    assert wrong_id.status_code == no_account.status_code == 401
    assert wrong_id.json()["detail"] == no_account.json()["detail"]


async def test_there_is_no_registration_endpoint():
    """The only way in is an account an administrator created."""
    async with _client() as client:
        for path in ("/api/v1/auth/register", "/api/v1/auth/signup"):
            assert (await client.post(path, json={})).status_code == 404


async def test_a_clinician_has_no_password_to_change():
    user = _make_clinician()
    async with _client() as client:
        res = await client.post("/api/v1/auth/password", cookies=_signed_in(user),
                                json={"current_password": "x", "new_password": "y" * 12})
    assert res.status_code == 400
    assert "Rad-Intel ID" in res.json()["detail"]


async def test_an_administrator_can_change_their_own_password_and_details():
    admin = _make_admin()
    cookies = _signed_in(admin)
    async with _client() as client:
        changed = await client.post("/api/v1/auth/password", cookies=cookies, json={
            "current_password": "a-long-enough-password",
            "new_password": "an-even-longer-password",
        })
        profile = await client.patch("/api/v1/admin/profile", cookies=cookies,
                                     json={"phone": "+91 98440 07409"})
    assert changed.status_code == 200
    assert profile.status_code == 200
    assert accounts.identify_admin("boss@study.org", "an-even-longer-password")
    assert accounts.get_by_id(admin["id"])["phone"] == "+91 98440 07409"


# --- rate limiting ---------------------------------------------------------

async def test_repeated_sign_in_attempts_are_slowed_down():
    _make_clinician()
    async with _client() as client:
        statuses = [
            (await client.post("/api/v1/auth/code", json={
                "email": "a@hospital.org", "radintel_id": "RADZZZZZZZZZ",
            })).status_code
            for _ in range(14)
        ]
    assert 429 in statuses, "guessing must eventually be refused outright"
    assert statuses[0] == 401


async def test_access_requests_are_rate_limited():
    body = {"full_name": "Dr Hopeful", "email": "hope@clinic.org",
            "institution": "Some Clinic", "phone": "+91 99999 11111"}
    async with _client() as client:
        statuses = [(await client.post("/api/v1/auth/request-access", json=body)).status_code
                    for _ in range(6)]
    assert statuses.count(200) <= 4
    assert statuses[-1] == 429


async def test_a_rate_limited_reply_says_when_to_come_back():
    body = {"full_name": "Dr Hopeful", "email": "hope@clinic.org",
            "institution": "Some Clinic", "phone": "+91 99999 11111"}
    async with _client() as client:
        last = None
        for _ in range(6):
            last = await client.post("/api/v1/auth/request-access", json=body)
    assert last.status_code == 429
    assert "Retry-After" in last.headers
    assert "Try again" in last.json()["detail"]


async def test_one_clinician_hitting_a_limit_does_not_block_another():
    """Limits are per account, not per address, once somebody is signed in."""
    from rad_intel.api.ratelimit import limit

    first = _make_clinician()
    second = _make_clinician("b@clinic.org", "Dr B", "B Clinic")
    app.router.routes  # touch, so the app is built before the probe route
    probe = limit("probe-bucket", 2, 3600, "probes")

    class _Request:
        def __init__(self):
            self.client = type("c", (), {"host": "10.0.0.1"})()
            self.headers = {}

    for _ in range(2):
        probe(_Request(), first)
    with pytest.raises(Exception):
        probe(_Request(), first)
    probe(_Request(), second)  # must not raise


# --- model versions --------------------------------------------------------

async def test_the_version_panel_says_which_checkpoint_is_live():
    user = _make_clinician()
    async with _client() as client:
        res = await client.get("/api/v1/feedback/models", cookies=_signed_in(user))
    body = res.json()
    assert body["shared_model"] == "hybrid"
    assert body["live_file"].endswith(".pt")
    assert "Analyse X-ray" in body["where_to_test"]
    base = [v for v in body["versions"] if v["is_base"]]
    assert len(base) == 1, "there is exactly one base checkpoint"
    assert sum(1 for v in body["versions"] if v["is_live"]) == 1


async def test_only_an_administrator_can_change_the_served_version():
    user = _make_clinician()
    async with _client() as client:
        res = await client.post("/api/v1/feedback/models/activate",
                                data={"file": "best_hybrid_model.pt"},
                                cookies=_signed_in(user))
    assert res.status_code == 403


async def test_activating_a_checkpoint_outside_the_weights_directory_is_refused():
    admin = _make_admin()
    async with _client() as client:
        for attempt in ("../rad_intel.db", "/etc/passwd", "nope.pt"):
            res = await client.post("/api/v1/feedback/models/activate",
                                    data={"file": attempt}, cookies=_signed_in(admin))
            assert res.status_code == 404, attempt


# --- several people at once ------------------------------------------------

async def test_many_parallel_requests_do_not_collide():
    """
    Opening the workspace fires eight requests at once, and several
    clinicians may do it at the same moment.

    This failed before the account tables were moved to write-ahead logging:
    every accessor re-ran its schema setup, which takes a write lock, so a
    single page load deadlocked itself with "database is locked". The
    assertion is simply that nothing comes back a server error.
    """
    import asyncio

    people = [_make_clinician(f"doc{n}@hospital.org", f"Dr {n}", "A Hospital")
              for n in range(4)]
    for index, person in enumerate(people):
        await _queue_case(person, sha=str(index) * 64)

    paths = ["/api/v1/feedback", "/api/v1/feedback/participants",
             "/api/v1/feedback/exports", "/api/v1/feedback/storage",
             "/api/v1/feedback/train/status", "/api/v1/auth/me"]

    async with _client() as client:
        results = await asyncio.gather(*[
            client.get(path, cookies=_signed_in(person))
            for person in people for path in paths
        ])

    server_errors = [r for r in results if r.status_code >= 500]
    assert not server_errors, \
        f"{len(server_errors)} of {len(results)} requests failed: " \
        f"{server_errors[0].text[:200]}"


async def test_two_clinicians_can_work_on_the_queue_at_the_same_time():
    """Concurrent writes from different people must all land."""
    import asyncio

    first = _make_clinician()
    second = _make_clinician("b@clinic.org", "Dr B", "B Clinic")
    records = [await _queue_case(first, sha=f"{n:064d}") for n in range(3)]
    records += [await _queue_case(second, sha=f"{n:064x}") for n in range(3, 6)]

    async with _client() as client:
        responses = await asyncio.gather(*[
            client.patch(f"/api/v1/feedback/{record}", json={"note": f"note {index}"},
                         cookies=_signed_in(first if index < 3 else second))
            for index, record in enumerate(records)
        ])

    assert [r.status_code for r in responses] == [200] * 6
    for index, record in enumerate(records):
        assert (await db_manager.get_feedback(record))["note"] == f"note {index}"
