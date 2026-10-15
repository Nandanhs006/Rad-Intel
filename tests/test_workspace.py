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
    _tokens.clear()
    # A fresh job coordinator, so a test cannot inherit a lock from another.
    monkeypatch.setattr(feedback_routes, "job", TrainingJob(tmp_path / "training.log"))
    yield path
    # Drain anything the background writer still holds, while this test's
    # database is still the one it is bound to.
    activity.flush(2.0)
    ratelimit.reset()


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://test")


_tokens: dict[str, str] = {}


def _signed_in(user: dict, mode: str | None = None) -> dict[str, str]:
    """
    A cookie jar for an account, bypassing the two-step sign-in.

    The default mode matches the door that account would really use: an
    administrator signs in through the administrator form, everybody else
    with a Rad-Intel ID. Pass `mode="clinician"` for an administrator who
    came in the other way, which is a case worth testing.

    Memoised per account and mode. An account is signed in on one device at a
    time, so issuing a second session silently invalidates the first -- a
    test calling this twice for the same person means "the same device
    again", not "a second one". Use session.issue directly for that.
    """
    mode = mode or ("admin" if user["role"] == "admin" else "clinician")
    key = f"{user['id']}:{mode}"
    if key not in _tokens:
        _tokens[key] = session.issue(user["id"], mode=mode)
    return {session.COOKIE_NAME: _tokens[key]}


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
    assert set(body["delivery"]) >= {"code_channel", "email_configured"}
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
        lambda code, name, email=None: (issued.append(code),
                                        (True, "test", "sent"))[1],
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



async def test_the_session_cookie_goes_when_the_browser_does(monkeypatch):
    """No Max-Age and no Expires, so closing the browser ends the session."""
    user = _make_clinician()
    async with _client() as client:
        res = await _log_in(client, user, monkeypatch)
    cookies = [h for k, h in res.headers.multi_items() if k.lower() == "set-cookie"]
    sess = next(h for h in cookies if h.startswith(f"{session.COOKIE_NAME}="))
    assert "HttpOnly" in sess and "SameSite=lax" in sess
    assert "Max-Age" not in sess and "expires" not in sess.lower()

    # The device marker is the opposite: it has to outlive the browser, or
    # every return looks like a new machine.
    device = next(h for h in cookies if h.startswith(f"{session.DEVICE_COOKIE_NAME}="))
    assert f"Max-Age={session.DEVICE_COOKIE_MAX_AGE}" in device
    assert "HttpOnly" in device

async def test_a_code_that_cannot_be_delivered_refuses_the_sign_in(monkeypatch):
    """Never fall back to showing the code: that is no second factor at all."""
    user = _make_clinician()
    monkeypatch.setattr(
        "rad_intel.api.routes.auth.notify.send_verification_code",
        lambda code, name, email=None: (False, "none",
                                        "Nothing is configured to send codes."),
    )
    async with _client() as client:
        res = await client.post("/api/v1/auth/code", json={
            "email": user["email"], "radintel_id": user["radintel_id"],
        })
    assert res.status_code == 503
    assert "code" not in res.json()
    assert "configured" in res.json()["detail"]


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


# --- mail that did not arrive ----------------------------------------------

async def test_the_console_counts_requests_that_never_reached_the_inbox():
    """
    A misconfigured SMTP password fails quietly from the applicant's side.

    They are told the administrator will be in touch, which is true whether
    or not the email went out, so the only place the failure can surface is
    the console.
    """
    admin = _make_admin()
    accounts.create_access_request("Dr Hopeful", "hope@clinic.org", "Some Clinic",
                                   phone="+91 99999 11111")
    async with _client() as client:
        res = await client.get("/api/v1/admin/overview", cookies=_signed_in(admin))
    assert res.json()["access_requests_unemailed"] == 1

    # Once it has gone out, it stops being counted.
    accounts.mark_request_notified(accounts.list_access_requests()[0]["id"])
    async with _client() as client:
        res = await client.get("/api/v1/admin/overview", cookies=_signed_in(admin))
    assert res.json()["access_requests_unemailed"] == 0


async def test_a_test_email_says_plainly_that_nothing_is_configured():
    admin = _make_admin()
    async with _client() as client:
        res = await client.post("/api/v1/admin/test-email", cookies=_signed_in(admin))
    assert res.status_code == 503
    assert "SMTP_HOST" in res.json()["detail"]


async def test_a_test_email_is_sent_when_mail_works(monkeypatch):
    admin = _make_admin()
    sent: list[tuple] = []
    monkeypatch.setattr("rad_intel.api.routes.admin.notify.email_configured", lambda: True)
    monkeypatch.setattr("rad_intel.api.routes.admin.notify.send_email",
                        lambda subject, body, to=None: (sent.append((subject, body)),
                                                        (True, "Emailed you."))[1])
    async with _client() as client:
        res = await client.post("/api/v1/admin/test-email", cookies=_signed_in(admin))
    assert res.status_code == 200
    assert len(sent) == 1


async def test_a_missed_request_can_be_emailed_later(monkeypatch):
    admin = _make_admin()
    accounts.create_access_request("Dr Hopeful", "hope@clinic.org", "Some Clinic",
                                   phone="+91 99999 11111")
    request_id = accounts.list_access_requests()[0]["id"]

    monkeypatch.setattr("rad_intel.api.routes.admin.notify.send_access_request_email",
                        lambda *args: (True, "Emailed you."))
    async with _client() as client:
        res = await client.post(f"/api/v1/admin/requests/{request_id}/resend",
                                cookies=_signed_in(admin))
        after = await client.get("/api/v1/admin/overview", cookies=_signed_in(admin))
    assert res.status_code == 200
    assert after.json()["access_requests_unemailed"] == 0


async def test_a_failed_resend_does_not_claim_the_request_was_delivered():
    admin = _make_admin()
    accounts.create_access_request("Dr Hopeful", "hope@clinic.org", "Some Clinic",
                                   phone="+91 99999 11111")
    request_id = accounts.list_access_requests()[0]["id"]
    async with _client() as client:
        res = await client.post(f"/api/v1/admin/requests/{request_id}/resend",
                                cookies=_signed_in(admin))
    assert res.status_code == 502
    assert accounts.list_access_requests()[0]["notified"] == 0


async def test_only_an_administrator_can_test_or_resend_mail():
    user = _make_clinician()
    async with _client() as client:
        for method, path in (("post", "/api/v1/admin/test-email"),
                             ("post", "/api/v1/admin/requests/any/resend")):
            signed_out = await client.request(method, path)
            as_clinician = await client.request(method, path, cookies=_signed_in(user))
            assert signed_out.status_code == 401, path
            assert as_clinician.status_code == 403, path


# --- how a code is delivered -----------------------------------------------

def test_an_emailed_code_does_not_reveal_the_whole_address(monkeypatch):
    from rad_intel.auth import notify

    monkeypatch.delenv("RAD_INTEL_OTP_CHANNEL", raising=False)
    monkeypatch.setenv("SMTP_HOST", "smtp.example.org")
    monkeypatch.setenv("SMTP_FROM", "study@example.org")
    monkeypatch.setattr(notify, "send_email", lambda **kwargs: (True, "sent"))

    ok, channel, message = notify.send_verification_code(
        "Known-Code1!", "Dr A", "asha.rao@hospital.org")
    assert ok and channel == "email"
    assert "asha.rao" not in message, "the full address must not be echoed back"
    assert "as" in message and "@hospital.org" in message


def test_a_code_is_never_in_the_message_shown_to_the_user(monkeypatch):
    from rad_intel.auth import notify

    monkeypatch.setenv("RAD_INTEL_OTP_ECHO", "1")
    for name in ("SMTP_HOST", "RAD_INTEL_OTP_CHANNEL"):
        monkeypatch.delenv(name, raising=False)
    _, _, message = notify.send_verification_code("Secret-Code1", "Dr A", "a@h.org")
    assert "Secret-Code1" not in message


# --- the email channel, over a real SMTP conversation -----------------------

class _SMTPSink:
    """
    A throwaway SMTP server, enough to accept one message.

    Worth the lines: stubbing send_email proves the routing but not the
    sending, and the first version of this caught a real bug -- STARTTLS was
    demanded unconditionally, so any server that did not advertise it failed.
    """

    def __init__(self, offer_starttls: bool = False):
        import socket
        import threading

        self.received: list[str] = []
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self._offer_starttls = offer_starttls
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        try:
            conn, _ = self._sock.accept()
        except OSError:
            return
        stream = conn.makefile("rwb")
        conn.sendall(b"220 localhost ESMTP\r\n")
        in_data, body = False, []
        while True:
            line = stream.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace")
            if in_data:
                if text.strip() == ".":
                    in_data = False
                    self.received.append("".join(body))
                    conn.sendall(b"250 OK\r\n")
                else:
                    body.append(text)
                continue
            verb = text.split()[0].upper() if text.split() else ""
            if verb == "EHLO":
                extra = b"250-STARTTLS\r\n" if self._offer_starttls else b""
                conn.sendall(b"250-localhost\r\n" + extra + b"250 HELP\r\n")
            elif verb == "DATA":
                in_data = True
                conn.sendall(b"354 End with .\r\n")
            elif verb == "QUIT":
                conn.sendall(b"221 Bye\r\n")
                break
            else:
                conn.sendall(b"250 OK\r\n")
        conn.close()

    def close(self) -> None:
        self._sock.close()


@pytest.fixture
def smtp_sink(monkeypatch):
    sink = _SMTPSink()
    monkeypatch.setenv("SMTP_HOST", "127.0.0.1")
    monkeypatch.setenv("SMTP_PORT", str(sink.port))
    monkeypatch.setenv("SMTP_FROM", "study@example.org")
    monkeypatch.delenv("SMTP_USER", raising=False)
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    monkeypatch.setenv("RAD_INTEL_OTP_CHANNEL", "email")
    monkeypatch.delenv("RAD_INTEL_OTP_ECHO", raising=False)
    yield sink
    sink.close()


def test_an_emailed_code_reaches_the_mailbox_and_not_the_browser(smtp_sink):
    import time

    from rad_intel.auth import notify

    ok, channel, shown = notify.send_verification_code(
        "Kx7!mQ2@Rb9Z", "Dr Asha Rao", "asha.rao@kmc.test")
    assert ok and channel == "email"

    for _ in range(40):
        if smtp_sink.received:
            break
        time.sleep(0.05)
    assert smtp_sink.received, "nothing reached the mail server"
    delivered = smtp_sink.received[0]

    assert "Kx7!mQ2@Rb9Z" in delivered, "the code must reach the mailbox"
    assert "asha.rao@kmc.test" in delivered
    assert "Kx7!mQ2@Rb9Z" not in shown, "the code must never go back to the browser"
    assert "asha.rao" not in shown, "the full address must not be echoed back"


def test_a_password_is_never_sent_over_an_untrusted_plaintext_connection(smtp_sink,
                                                                         monkeypatch):
    """
    The sink offers no STARTTLS. Without credentials that is an acceptable
    local relay; with them the connection would carry a password in the
    clear, so it must be refused rather than downgraded.
    """
    from rad_intel.auth import notify

    monkeypatch.setenv("SMTP_USER", "study@example.org")
    monkeypatch.setenv("SMTP_PASSWORD", "hunter2-but-longer")
    sent, detail = notify.send_email("subject", "body", "someone@example.org")
    assert not sent
    assert "STARTTLS" in detail
    assert not smtp_sink.received, "nothing may be sent over the clear connection"


# --- getting in when mail is not working ------------------------------------

# --- getting back in when nothing can be delivered --------------------------

async def test_a_code_minted_on_the_console_logs_in(monkeypatch):
    """
    The locked-out path: no mail working, a code minted on the server with
    manage_clinicians.py, and a session at the end of it.
    """
    admin = _make_admin()
    # Minted on the server through the administrator door, so the session it
    # opens carries the console.
    code, _ = otp.issue_code(admin["id"], channel="console", mode="admin")

    async with _client() as client:
        # No /auth/code call at all: delivery is the thing that is broken.
        res = await client.post("/api/v1/auth/login",
                                json={"email": admin["email"], "code": code})
        assert res.status_code == 200
        assert (await client.get("/api/v1/admin/overview")).status_code == 200


async def test_a_console_code_still_obeys_every_other_rule():
    admin = _make_admin()
    code, _ = otp.issue_code(admin["id"], channel="console", mode="admin")
    async with _client() as client:
        first = await client.post("/api/v1/auth/login",
                                  json={"email": admin["email"], "code": code})
        second = await client.post("/api/v1/auth/login",
                                   json={"email": admin["email"], "code": code})
    assert first.status_code == 200
    assert second.status_code == 401, "a console code is single use like any other"


# --- one device, thirty days, through the HTTP surface ----------------------

async def _log_in(client, user, monkeypatch, code="Known-Code1!"):
    """
    Drive the real two-step sign-in with delivery stubbed out.

    Ages any code already issued past the resend cooldown first: these tests
    sign the same person in twice within a second to check what happens to
    their other device, which is not something real use does.
    """
    import sqlite3

    with sqlite3.connect(otp.DB_PATH) as conn:
        conn.execute("UPDATE login_codes SET created_at = datetime('now', '-70 seconds');")
    monkeypatch.setattr(
        "rad_intel.api.routes.auth.notify.send_verification_code",
        lambda c, name, email=None: (True, "test", "sent"))
    monkeypatch.setattr(otp, "generate_code", lambda: code)
    await client.post("/api/v1/auth/code", json={
        "email": user["email"], "radintel_id": user["radintel_id"]})
    return await client.post("/api/v1/auth/login",
                             json={"email": user["email"], "code": code})


async def _log_in_as_admin(client, admin, monkeypatch, password="a-long-enough-password",
                           code="Admin-Code12"):
    """The administrator door: email and password, then the code."""
    import sqlite3

    with sqlite3.connect(otp.DB_PATH) as conn:
        conn.execute("UPDATE login_codes SET created_at = datetime('now', '-70 seconds');")
    monkeypatch.setattr(
        "rad_intel.api.routes.auth.notify.send_verification_code",
        lambda c, name, email=None: (True, "test", "sent"))
    monkeypatch.setattr(otp, "generate_code", lambda: code)
    await client.post("/api/v1/auth/admin/code",
                      json={"email": admin["email"], "password": password})
    return await client.post("/api/v1/auth/login",
                             json={"email": admin["email"], "code": code})


async def test_a_second_device_signs_the_first_one_out(monkeypatch):
    user = _make_clinician()

    async with _client() as laptop, _client() as phone:
        first = await _log_in(laptop, user, monkeypatch, "Laptop-Code1")
        assert first.status_code == 200
        assert first.json()["displaced_other_device"] is False
        assert (await laptop.get("/api/v1/feedback")).status_code == 200

        second = await _log_in(phone, user, monkeypatch, "Phone-Code12")
        assert second.status_code == 200
        assert second.json()["displaced_other_device"] is True, \
            "the person signing in must be told they displaced a device"

        assert (await phone.get("/api/v1/feedback")).status_code == 200
        assert (await laptop.get("/api/v1/feedback")).status_code == 401, \
            "the device that was already open must be signed out"


async def test_the_displacement_is_written_to_the_activity_log(monkeypatch):
    admin = _make_admin()
    user = _make_clinician()
    async with _client() as first, _client() as second:
        await _log_in(first, user, monkeypatch, "Laptop-Code1")
        await _log_in(second, user, monkeypatch, "Phone-Code12")
    async with _client() as client:
        log = await client.get("/api/v1/admin/activity", cookies=_signed_in(admin))
    actions = [e["action"] for e in log.json()["entries"]]
    assert "login.displaced_other_device" in actions


async def test_logging_out_requires_a_fresh_code_to_come_back(monkeypatch):
    user = _make_clinician()
    async with _client() as client:
        await _log_in(client, user, monkeypatch, "Known-Code1!")
        assert (await client.get("/api/v1/feedback")).status_code == 200
        await client.post("/api/v1/auth/logout")
        assert (await client.get("/api/v1/feedback")).status_code == 401
        # The code that opened the session is spent, so it cannot be reused.
        again = await client.post("/api/v1/auth/login",
                                  json={"email": user["email"], "code": "Known-Code1!"})
        assert again.status_code == 401


async def test_a_session_says_when_it_will_ask_for_a_code_again(monkeypatch):
    user = _make_clinician()
    async with _client() as client:
        await _log_in(client, user, monkeypatch)
        me = await client.get("/api/v1/auth/me")
    body = me.json()
    assert body["authenticated"] is True
    assert body["session_expires_at"], "the server still keeps a ceiling"


async def test_an_administrator_sees_one_device_per_account_and_its_expiry():
    admin = _make_admin()
    user = _make_clinician()
    _signed_in(user)
    async with _client() as client:
        res = await client.get("/api/v1/admin/sessions", cookies=_signed_in(admin))
    rows = res.json()["sessions"]
    assert len(rows) == 2, "one each, not one per browser they have ever used"
    assert all(row["expires_at"] for row in rows)


async def test_closing_a_session_forces_that_person_to_use_a_code_again():
    admin = _make_admin()
    user = _make_clinician()
    victim = _signed_in(user)
    async with _client() as client:
        listing = await client.get("/api/v1/admin/sessions", cookies=_signed_in(admin))
        row = next(r for r in listing.json()["sessions"] if r["user_id"] == user["id"])
        closed = await client.delete(f"/api/v1/admin/sessions/{row['id']}",
                                     cookies=_signed_in(admin))
        assert closed.status_code == 200
        assert (await client.get("/api/v1/feedback", cookies=victim)).status_code == 401
    # The account itself is untouched: they can log in again with a new code.
    assert accounts.get_by_id(user["id"])["status"] == "active"


async def test_the_expiry_is_known_from_the_moment_of_signing_in(monkeypatch):
    """
    Not only from /me afterwards: the interface shows "signed in until …" as
    soon as the workspace appears, and the first thing it has to work from is
    the login response.
    """
    user = _make_clinician()
    async with _client() as client:
        res = await _log_in(client, user, monkeypatch)
    assert res.json()["session_expires_at"], "login must say when it runs out"


# --- which door you came in by ----------------------------------------------

async def test_an_administrator_using_the_clinician_door_gets_no_console():
    """
    A Rad-Intel ID is written down and handed over. It must not reach the
    controls that read and delete everybody's data, even when the account it
    belongs to is an administrator.
    """
    admin = _make_admin()
    as_clinician = _signed_in(admin, mode="clinician")

    async with _client() as client:
        assert (await client.get("/api/v1/feedback", cookies=as_clinician)).status_code == 200
        for path in ("/api/v1/admin/overview", "/api/v1/admin/users",
                     "/api/v1/admin/activity", "/api/v1/admin/sessions"):
            res = await client.get(path, cookies=as_clinician)
            assert res.status_code == 403, path
            assert "log back in through Administrator" in res.json()["detail"]


async def test_the_same_account_through_the_administrator_door_gets_it():
    admin = _make_admin()
    async with _client() as client:
        res = await client.get("/api/v1/admin/overview", cookies=_signed_in(admin))
    assert res.status_code == 200


async def test_the_session_says_which_door_was_used(monkeypatch):
    admin = _make_admin()
    async with _client() as clinician_door:
        body = (await _log_in(clinician_door, admin, monkeypatch)).json()
        assert body["session_mode"] == "clinician"
        assert body["is_admin_session"] is False
        assert body["role"] == "admin", "the account is still an administrator"

    async with _client() as admin_door:
        body = (await _log_in_as_admin(admin_door, admin, monkeypatch)).json()
        assert body["session_mode"] == "admin"
        assert body["is_admin_session"] is True


async def test_a_clinician_session_cannot_change_the_administrator_password():
    admin = _make_admin()
    async with _client() as client:
        res = await client.post("/api/v1/auth/password",
                                cookies=_signed_in(admin, mode="clinician"),
                                json={"current_password": "a-long-enough-password",
                                      "new_password": "an-even-longer-password"})
    assert res.status_code == 403
    assert accounts.identify_admin(admin["email"], "a-long-enough-password"), \
        "the password must be unchanged"


async def test_the_mode_cannot_be_claimed_at_the_second_step(monkeypatch):
    """
    The door is recorded when the code is issued, where the first factor was
    actually checked, so submitting the code cannot upgrade the session.
    """
    admin = _make_admin()
    async with _client() as client:
        # Asked for through the clinician door...
        await _log_in(client, admin, monkeypatch, code="Clinic-Code1")
        # ...and no amount of asking at /login makes it an admin session.
        me = (await client.get("/api/v1/auth/me")).json()
        assert me["is_admin_session"] is False
        assert (await client.get("/api/v1/admin/users")).status_code == 403


# --- the same machine, twice -------------------------------------------------

async def test_closing_the_browser_and_returning_is_not_a_new_device(monkeypatch):
    """
    The complaint this fixes: logging in again on the same machine reported
    that another device had been signed out. One client, with its cookie jar
    kept, is one device however many times the session cookie goes.
    """
    user = _make_clinician()
    async with _client() as same_machine:
        first = await _log_in(same_machine, user, monkeypatch, "Code-One12!")
        assert first.json()["displaced_other_device"] is False
        assert session.DEVICE_COOKIE_NAME in same_machine.cookies

        # Closing the browser drops the session cookie and keeps the device one.
        del same_machine.cookies[session.COOKIE_NAME]

        again = await _log_in(same_machine, user, monkeypatch, "Code-Two12!")
        assert again.status_code == 200
        assert again.json()["displaced_other_device"] is False, \
            "the same machine must not be reported as a displaced device"


async def test_a_genuinely_different_machine_still_is(monkeypatch):
    user = _make_clinician()
    async with _client() as laptop, _client() as phone:
        await _log_in(laptop, user, monkeypatch, "Code-One12!")
        second = await _log_in(phone, user, monkeypatch, "Code-Two12!")
    assert second.json()["displaced_other_device"] is True


async def test_logging_out_keeps_the_device_marker(monkeypatch):
    """Or every return after a deliberate sign-out looks like a new machine."""
    user = _make_clinician()
    async with _client() as client:
        await _log_in(client, user, monkeypatch, "Code-One12!")
        await client.post("/api/v1/auth/logout")
        assert session.DEVICE_COOKIE_NAME in client.cookies
        back = await _log_in(client, user, monkeypatch, "Code-Two12!")
    assert back.json()["displaced_other_device"] is False


# --- what a failed sign-in says ---------------------------------------------

async def test_a_failed_sign_in_says_only_that_it_failed():
    """
    Identical, and short. A helpful message that is specific about one half
    of a guess tells an attacker the other half was the problem.
    """
    _make_clinician()
    async with _client() as client:
        wrong_id = await client.post("/api/v1/auth/code", json={
            "email": "a@hospital.org", "radintel_id": "RADZZZZZZZZZ"})
        no_account = await client.post("/api/v1/auth/code", json={
            "email": "nobody@nowhere.io", "radintel_id": "RADZZZZZZZZZ"})
        wrong_password = await client.post("/api/v1/auth/admin/code", json={
            "email": "a@hospital.org", "password": "guess"})
    bodies = {r.json()["detail"] for r in (wrong_id, no_account, wrong_password)}
    assert bodies == {"Invalid credentials."}


# --- where the images live ---------------------------------------------------

def test_the_store_can_be_moved_off_the_project_directory(monkeypatch, tmp_path):
    """
    Hosting needs this: a small host gives a container a filesystem that is
    wiped on redeploy, so the radiographs have to sit on a mounted volume.
    """
    import importlib

    monkeypatch.setenv("RAD_INTEL_FEEDBACK_DIR", str(tmp_path / "elsewhere"))
    moved = importlib.reload(importlib.import_module("rad_intel.api.routes.feedback"))
    try:
        assert moved.FEEDBACK_DIR == (tmp_path / "elsewhere").resolve()
        assert moved.IMAGE_DIR.parent == moved.FEEDBACK_DIR
        # A path recorded from outside the project stays absolute, and comes
        # back through the containment check.
        moved._ensure_dirs()
        image = moved.IMAGE_DIR / "abc.jpg"
        image.write_bytes(b"x")
        recorded = moved._store_path(image)
        assert pathlib.Path(recorded).is_absolute()
        assert moved._resolve_image(recorded) == image.resolve()
    finally:
        monkeypatch.delenv("RAD_INTEL_FEEDBACK_DIR")
        importlib.reload(moved)


def test_a_tampered_path_cannot_reach_a_file_outside_the_store():
    from rad_intel.api.routes import feedback as fb

    for attempt in ("../rad_intel.db", "/etc/passwd", "feedback_store/../rad_intel.db"):
        assert fb._resolve_image(attempt) is None, attempt
    assert fb._resolve_image(None) is None


# --- what an anonymous visitor can read -------------------------------------

async def test_the_analysis_audit_log_is_not_public():
    """
    The analyse form invites a patient identifier and a clinical history, so
    this table is the one place holding free text somebody typed about a
    patient. It used to be readable by anyone who asked.
    """
    async with _client() as client:
        listing = await client.get("/api/v1/history")
        detail = await client.get("/api/v1/history/any-id")
    assert listing.status_code == 401
    assert detail.status_code == 401


async def test_a_clinician_cannot_read_it_either():
    """
    Those rows include runs by members of the public, who never saw the
    workspace's consent panel and did not agree to be visible to a group of
    named clinicians. Only the administrator answers for them.
    """
    user = _make_clinician()
    async with _client() as client:
        res = await client.get("/api/v1/history", cookies=_signed_in(user))
    assert res.status_code == 403


async def test_an_administrator_can_read_it():
    admin = _make_admin()
    async with _client() as client:
        res = await client.get("/api/v1/history", cookies=_signed_in(admin))
    assert res.status_code == 200
    assert "records" in res.json()


async def test_no_website_can_make_a_credentialed_request_to_this_api():
    """
    allow_origins=["*"] with allow_credentials=True makes Starlette echo back
    whichever Origin asked. SameSite=Lax kept the cookie off those requests,
    so it was not exploitable as it stood -- but it was one cookie attribute
    away, and the interface is same-origin and needs no CORS at all.
    """
    user = _make_clinician()
    async with _client() as client:
        res = await client.get("/api/v1/feedback", cookies=_signed_in(user),
                               headers={"Origin": "https://evil.example"})
    allowed = res.headers.get("access-control-allow-origin")
    assert allowed != "https://evil.example"
    assert allowed != "*"
    assert res.headers.get("access-control-allow-credentials") != "true"


# --- a report belongs to the browser that made it ---------------------------

async def test_a_report_cannot_be_fetched_with_its_id_alone(dummy_cxr_bytes):
    """
    The id was the only thing between a stranger and somebody else's report,
    and a report carries the patient reference and clinical history whoever
    ran it typed in. An id is a capability: it travels in URLs, in history,
    in a shared link, in a log. "Unguessable" is not "nobody else can have
    it".
    """
    async with _client() as mine:
        res = await mine.post("/api/v1/analyze",
                              files={"file": ("x.png", dummy_cxr_bytes, "image/png")},
                              data={"patient_id": "PT-PRIVATE"})
        assert res.status_code == 200
        analysis_id = res.json()["analysis_id"]
        # The browser that ran it can have it.
        assert (await mine.get(f"/api/v1/analyze/{analysis_id}/pdf")).status_code == 200

    # Somebody else holding the same id cannot.
    async with _client() as stranger:
        assert (await stranger.get(f"/api/v1/analyze/{analysis_id}/pdf")).status_code == 404


async def test_a_grant_cookie_cannot_be_edited_to_add_an_id():
    from rad_intel.api import grants

    real = grants.grant(None, "analysis-one")
    assert grants.holds(real, "analysis-one")

    ids, _, signature = real.rpartition(".")
    forged = f"{ids},analysis-two.{signature}"
    assert not grants.holds(forged, "analysis-two"), "the list must be signed as a whole"
    assert not grants.holds(forged, "analysis-one"), "a tampered cookie is worth nothing"
    assert grants.read("nonsense") == []
    assert grants.read(None) == []


async def test_the_grant_list_does_not_grow_without_bound():
    from rad_intel.api import grants

    cookie = None
    for n in range(grants.MAX_GRANTS + 10):
        cookie = grants.grant(cookie, f"analysis-{n}")
    assert len(grants.read(cookie)) == grants.MAX_GRANTS
    assert grants.holds(cookie, f"analysis-{grants.MAX_GRANTS + 9}"), "newest kept"
    assert not grants.holds(cookie, "analysis-0"), "oldest dropped"


# --- what the service tells the world about itself --------------------------

async def test_the_api_map_is_not_published():
    """
    The interactive docs are a complete directory of the service, including
    every administrator route. Useful while building, of no use to a visitor.
    """
    async with _client() as client:
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert (await client.get(path)).status_code == 404, path


async def test_every_response_carries_the_browser_side_protections():
    async with _client() as client:
        res = await client.get("/api/v1/health")
    headers = {k.lower(): v for k, v in res.headers.items()}
    assert headers["x-frame-options"] == "DENY"
    assert headers["x-content-type-options"] == "nosniff"
    assert "strict-origin" in headers["referrer-policy"]
    csp = headers["content-security-policy"]
    assert "frame-ancestors 'none'" in csp
    assert "default-src 'self'" in csp


# --- releasing a model to the public ----------------------------------------

@pytest.fixture
def fake_checkpoint(tmp_path, monkeypatch):
    """A versioned checkpoint on disk, with scores, in a scratch weights dir."""
    import torch

    from rad_intel.models import registry

    weights = tmp_path / "weights"
    weights.mkdir()
    path = weights / "best_hybrid_v2_model.pt"
    torch.save({
        "model_type": "hybrid",
        "optimal_threshold": 0.5574,
        "model_state_dict": {},
        "metrics_after": {"accuracy": 0.9295, "auroc": 0.9741, "precision": 0.94,
                          "sensitivity": 0.95, "specificity": 0.8761, "f1": 0.945},
        "provenance": {"reviewed_cases": 12, "created_at": "2026-10-15T00:00:00+00:00",
                       "base_checkpoint": "best_hybrid_model.pt"},
    }, path)
    monkeypatch.setattr(registry, "WEIGHTS_DIR", weights)
    monkeypatch.setattr(registry, "REGISTRY_PATH", weights / "released_models.json")
    monkeypatch.setattr("rad_intel.api.routes.feedback.WEIGHTS_DIR", weights)
    return path


async def test_releasing_needs_the_password_every_time(fake_checkpoint):
    admin = _make_admin()
    async with _client() as client:
        wrong = await client.post("/api/v1/admin/models/release", cookies=_signed_in(admin),
                                  json={"file": fake_checkpoint.name, "password": "nope"})
        assert wrong.status_code == 403
        assert "Nothing was released" in wrong.json()["detail"]

        right = await client.post("/api/v1/admin/models/release", cookies=_signed_in(admin),
                                  json={"file": fake_checkpoint.name,
                                        "password": "a-long-enough-password"})
    assert right.status_code == 200
    assert right.json()["released"] == "hybrid_v2"


async def test_a_released_version_appears_on_the_public_model_list(fake_checkpoint):
    from rad_intel.models import registry

    registry.release(fake_checkpoint.name, released_by="boss@study.org",
                     metrics={"accuracy": 0.9295, "auroc": 0.9741})
    async with _client() as client:
        res = await client.get("/api/v1/models")
    names = {m["name"]: m for m in res.json()["available_models"]}
    assert "hybrid_v2" in names, "a released version must be offered to everybody"
    entry = names["hybrid_v2"]
    assert entry["is_released"] is True
    assert "acc" in entry["label"] and "AUC 0.974" in entry["label"]
    assert entry["metrics"]["accuracy"] == pytest.approx(0.9295)
    # The built-in models are still there and still unreleased.
    assert names["hybrid"]["is_released"] is False


async def test_a_version_with_no_scores_cannot_be_released(tmp_path, monkeypatch):
    """Releasing a model nobody can judge is worse than not releasing one."""
    import torch

    from rad_intel.models import registry

    weights = tmp_path / "weights"
    weights.mkdir()
    path = weights / "best_hybrid_v9_model.pt"
    torch.save({"model_type": "hybrid", "model_state_dict": {}}, path)
    monkeypatch.setattr(registry, "WEIGHTS_DIR", weights)
    monkeypatch.setattr(registry, "REGISTRY_PATH", weights / "released_models.json")
    monkeypatch.setattr("rad_intel.api.routes.feedback.WEIGHTS_DIR", weights)

    admin = _make_admin()
    async with _client() as client:
        res = await client.post("/api/v1/admin/models/release", cookies=_signed_in(admin),
                                json={"file": path.name, "password": "a-long-enough-password"})
    assert res.status_code == 400
    assert "no scores" in res.json()["detail"]


async def test_a_base_checkpoint_is_not_releasable(fake_checkpoint):
    """They are built in, always offered, and not something to publish or retract."""
    from rad_intel.models import registry

    for name in ("best_hybrid_model.pt", "../rad_intel.db", "/etc/passwd"):
        with pytest.raises(ValueError):
            registry.release(name, released_by="boss@study.org", metrics={"accuracy": 1.0})


async def test_withdrawing_removes_it_from_the_list_but_keeps_the_file(fake_checkpoint):
    from rad_intel.models import registry

    registry.release(fake_checkpoint.name, released_by="boss@study.org",
                     metrics={"accuracy": 0.9})
    admin = _make_admin()
    async with _client() as client:
        res = await client.post("/api/v1/admin/models/withdraw", cookies=_signed_in(admin),
                                json={"file": "hybrid_v2",
                                      "password": "a-long-enough-password"})
        listing = await client.get("/api/v1/models")
    assert res.status_code == 200
    assert "hybrid_v2" not in {m["name"] for m in listing.json()["available_models"]}
    assert fake_checkpoint.is_file(), "the checkpoint itself must be kept"


async def test_only_an_administrator_session_can_release(fake_checkpoint):
    user = _make_clinician()
    admin = _make_admin()
    body = {"file": fake_checkpoint.name, "password": "a-long-enough-password"}
    async with _client() as client:
        anon = await client.post("/api/v1/admin/models/release", json=body)
        clinician = await client.post("/api/v1/admin/models/release",
                                      cookies=_signed_in(user), json=body)
        wrong_door = await client.post("/api/v1/admin/models/release",
                                       cookies=_signed_in(admin, mode="clinician"),
                                       json=body)
    assert anon.status_code == 401
    assert clinician.status_code == 403
    assert wrong_door.status_code == 403


async def test_an_unknown_model_name_is_refused_clearly(dummy_cxr_bytes):
    """The field accepts any string now, so this is where a bad one stops."""
    async with _client() as client:
        res = await client.post("/api/v1/predict",
                                files={"file": ("x.png", dummy_cxr_bytes, "image/png")},
                                data={"model_name": "hybrid_v99"})
    assert res.status_code == 400
    assert "no model called 'hybrid_v99'" in res.json()["detail"]
