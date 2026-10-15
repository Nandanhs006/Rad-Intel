"""
Reviewed radiographs outliving the disk they were uploaded to.

Nothing here talks to Google. What is being checked is the behaviour around
it, which is where the decisions are:

* a radiograph goes to Drive as well as to the disk, and a review is still
  saved when Drive is unreachable -- a clinician's work must not depend on a
  third party being up;
* a server whose disk was wiped gets the pixels back from Drive the first
  time somebody asks for that case, and writes them down again;
* deleting a review deletes the durable copy too, or "deleted" is a lie;
* the connection asks for exactly one Google scope, the one that can only
  see what this app itself made.
"""

import re

import httpx
import pytest

from rad_intel.api.routes import feedback as feedback_routes
from rad_intel.storage import drive

from tests.test_workspace import (  # noqa: F401  (fixtures come with them)
    _client,
    _make_admin,
    _make_clinician,
    _queue_case_with_image,
    _signed_in,
    isolated_state,
)


@pytest.fixture
def connected(monkeypatch):
    """Drive configured, and a fake one standing in for Google."""
    monkeypatch.setenv("GOOGLE_DRIVE_CLIENT_ID", "id")
    monkeypatch.setenv("GOOGLE_DRIVE_CLIENT_SECRET", "secret")
    monkeypatch.setenv("GOOGLE_DRIVE_REFRESH_TOKEN", "refresh")

    class Fake:
        def __init__(self):
            self.files: dict[str, bytes] = {}
            self.uploads: list[str] = []
            self.deleted: list[str] = []
            self.fail = False

        def upload(self, path, name, mime="image/jpeg"):
            if self.fail:
                raise drive.DriveError("Drive is having a day.")
            file_id = f"drive-{len(self.files) + 1}"
            self.files[file_id] = path.read_bytes()
            self.uploads.append(name)
            return file_id

        def download(self, file_id, into):
            if file_id not in self.files:
                return False
            into.parent.mkdir(parents=True, exist_ok=True)
            into.write_bytes(self.files[file_id])
            return True

        def delete(self, file_id):
            self.deleted.append(file_id)
            return self.files.pop(file_id, None) is not None

        def check(self):
            return {"configured": True, "working": True, "folder_id": "folder-1",
                    "folder_name": drive.FOLDER_NAME, "folder_url": "https://x/",
                    "detail": "ok"}

    fake = Fake()
    monkeypatch.setattr(drive, "store", fake)
    monkeypatch.setattr(feedback_routes.drive, "store", fake)
    return fake


# --- the scope --------------------------------------------------------------

def test_the_connection_asks_for_one_narrow_permission():
    """
    `drive.file` is per-file access to files the app itself created. It is not
    a promise to behave: the API will not return anything else, which is why
    the same Drive can hold the administrator's own documents and the trained
    checkpoints without this being able to see them.
    """
    assert drive.SCOPE == "https://www.googleapis.com/auth/drive.file"

    import scripts.connect_drive as connect

    assert connect.SCOPE == drive.SCOPE
    # And nothing anywhere asks for the broad one.
    for module in (drive.__file__, connect.__file__):
        text = open(module, encoding="utf-8").read()
        assert "auth/drive'" not in text and 'auth/drive"' not in text
        assert "drive.readonly" not in text


def test_nothing_is_attempted_when_drive_is_not_set_up(monkeypatch):
    for name in ("GOOGLE_DRIVE_CLIENT_ID", "GOOGLE_DRIVE_CLIENT_SECRET",
                 "GOOGLE_DRIVE_REFRESH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    assert drive.configured() is False
    status = drive.DriveStore().check()
    assert status["configured"] is False
    assert "disk" in status["detail"]
    with pytest.raises(drive.DriveError, match="connect_drive"):
        drive.DriveStore()._access_token()


def test_a_revoked_credential_says_what_to_do(monkeypatch):
    monkeypatch.setenv("GOOGLE_DRIVE_CLIENT_ID", "id")
    monkeypatch.setenv("GOOGLE_DRIVE_CLIENT_SECRET", "secret")
    monkeypatch.setenv("GOOGLE_DRIVE_REFRESH_TOKEN", "stale")
    monkeypatch.setattr(httpx, "post",
                        lambda *a, **k: httpx.Response(400, json={"error": "invalid_grant"}))

    with pytest.raises(drive.DriveError) as exc:
        drive.DriveStore()._access_token()
    # The commonest cause by far, and invisible otherwise: an OAuth app left
    # in Testing has its refresh token expired after seven days.
    assert "Testing" in str(exc.value)
    assert "seven days" in str(exc.value)


# --- through the workspace --------------------------------------------------

async def test_a_review_still_saves_when_drive_is_down(connected, dummy_cxr_bytes,
                                                       monkeypatch):
    """A clinician's work does not depend on a third party being up."""
    connected.fail = True
    user = _make_clinician()
    async with _client() as client:
        res = await client.post(
            "/api/v1/feedback/submit", cookies=_signed_in(user),
            files={"file": ("x.png", dummy_cxr_bytes, "image/png")},
            data={"verdict": "correct"},
        )
    assert res.status_code == 200, res.text

    from rad_intel.storage.database import db_manager

    record = await db_manager.get_feedback(res.json()["id"])
    assert record["drive_file_id"] is None
    assert len(await db_manager.feedback_without_drive_copy()) == 1, \
        "and it is remembered as still owed"


async def test_a_wiped_disk_gets_the_radiograph_back(connected):
    """
    The point of the whole thing.

    The database survives a redeploy on a mounted volume; a container's own
    filesystem does not. This is the case being opened on a server that came
    up without the pixels.
    """
    from rad_intel.storage.database import db_manager

    doctor = _make_clinician()
    record_id = await _queue_case_with_image(doctor, "a" * 64)
    record = await db_manager.get_feedback(record_id)

    local = feedback_routes._resolve_image(record["image_path"])
    assert local and local.is_file()
    file_id = connected.upload(local, local.name)
    await db_manager.set_drive_file_id(record_id, file_id)

    # The disk goes.
    local.unlink()
    assert feedback_routes._resolve_image(record["image_path"]) is None

    async with _client() as client:
        res = await client.get(f"/api/v1/feedback/image/{record_id}",
                               cookies=_signed_in(doctor))
    assert res.status_code == 200
    assert res.content == connected.files[file_id]
    # And it was written back down, so the next request is local again.
    assert (feedback_routes.IMAGE_DIR / f"{'a' * 64}.jpg").is_file()


async def test_a_radiograph_that_is_in_neither_place_is_a_404(connected):
    from rad_intel.storage.database import db_manager

    doctor = _make_clinician()
    record_id = await _queue_case_with_image(doctor, "b" * 64)
    record = await db_manager.get_feedback(record_id)
    feedback_routes._resolve_image(record["image_path"]).unlink()

    async with _client() as client:
        res = await client.get(f"/api/v1/feedback/image/{record_id}",
                               cookies=_signed_in(doctor))
    assert res.status_code == 404


async def test_deleting_a_review_deletes_the_durable_copy(connected):
    from rad_intel.storage.database import db_manager

    doctor = _make_clinician()
    record_id = await _queue_case_with_image(doctor, "c" * 64)
    record = await db_manager.get_feedback(record_id)
    local = feedback_routes._resolve_image(record["image_path"])
    file_id = connected.upload(local, local.name)
    await db_manager.set_drive_file_id(record_id, file_id)

    async with _client() as client:
        res = await client.delete(f"/api/v1/feedback/{record_id}",
                                  cookies=_signed_in(doctor))
    assert res.status_code == 200
    assert file_id in connected.deleted
    assert file_id not in connected.files, "'deleted' has to mean deleted"


async def test_the_backfill_pays_off_what_submitting_could_not(connected):
    """Radiographs reviewed before Drive was connected, or while it was down."""
    from rad_intel.storage.database import db_manager

    admin = _make_admin()
    doctor = _make_clinician()
    for digest in ("d" * 64, "e" * 64, "f" * 64):
        await _queue_case_with_image(doctor, digest)
    assert len(await db_manager.feedback_without_drive_copy()) == 3

    async with _client() as client:
        before = await client.get("/api/v1/feedback/drive", cookies=_signed_in(admin))
        res = await client.post("/api/v1/feedback/drive/backfill",
                                cookies=_signed_in(admin))
    assert before.json()["awaiting_backup"] == 3
    body = res.json()
    assert body["copied"] == 3 and body["remaining"] == 0
    assert "now in Drive" in body["detail"]
    assert len(connected.uploads) == 3


async def test_the_backfill_stops_at_the_first_failure_and_says_so(connected):
    """One failure is usually all of them: no credentials, or no room."""
    from rad_intel.storage.database import db_manager

    admin = _make_admin()
    doctor = _make_clinician()
    for digest in ("1" * 64, "2" * 64):
        await _queue_case_with_image(doctor, digest)
    connected.fail = True

    async with _client() as client:
        res = await client.post("/api/v1/feedback/drive/backfill",
                                cookies=_signed_in(admin))
    body = res.json()
    assert body["copied"] == 0 and body["failed"] == 1
    assert "free space" in body["detail"]
    assert len(await db_manager.feedback_without_drive_copy()) == 2


async def test_only_an_administrator_sees_where_the_radiographs_are_kept(connected):
    """It names a folder in a personal Drive."""
    doctor = _make_clinician()
    async with _client() as client:
        status = await client.get("/api/v1/feedback/drive", cookies=_signed_in(doctor))
        sweep = await client.post("/api/v1/feedback/drive/backfill",
                                  cookies=_signed_in(doctor))
        anon = await client.get("/api/v1/feedback/drive")
    assert status.status_code == 403
    assert sweep.status_code == 403
    assert anon.status_code == 401


async def test_a_tampered_path_cannot_make_a_restore_write_anywhere_else(connected,
                                                                         monkeypatch):
    """The restore path is rebuilt from the digest, not taken from the row."""
    record = {
        "id": "abcdefgh-0000",
        "image_path": "../../../../tmp/escaped.jpg",
        "image_sha256": "../../etc/passwd",
        "drive_file_id": "drive-1",
    }
    connected.files["drive-1"] = b"\xff\xd8\xff\xd9"
    assert await feedback_routes._image_for(record) is None

    record["image_sha256"] = "9" * 64
    restored = await feedback_routes._image_for(record)
    assert restored is not None
    assert restored.parent == feedback_routes.IMAGE_DIR.resolve() \
        or restored.parent == feedback_routes.IMAGE_DIR
    assert re.fullmatch(r"9{64}\.jpg", restored.name)
