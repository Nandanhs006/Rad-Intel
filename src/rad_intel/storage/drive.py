"""
Google Drive as the place reviewed radiographs actually live.

The problem this solves is that a small host's disk is not permanent. Most
of them hand a container a filesystem that is wiped on every redeploy, and
even with a mounted volume the images exist in exactly one place, on a
machine rented by the month. The radiographs clinicians contribute are the
one thing in this study that cannot be regenerated: the model can be trained
again, the database can be rebuilt, a lost radiograph is lost.

So the server keeps a working copy on its own disk, because serving a
thumbnail should not be a round trip to Google, and every image is also put
in the administrator's Drive. If the disk goes, the images are still there;
if a file is asked for and is not on disk, it is fetched back from Drive and
written down again. Drive is the record, the disk is the cache.

What this can and cannot see
----------------------------
It holds one scope, `drive.file`, which Google defines as per-file access to
files the app itself created. Not "files it promises only to touch" -- the
API will not return anything else, so the same Drive can hold the
administrator's own documents, the trained checkpoints, everything, and this
can neither read nor list any of it. It makes one folder, puts radiographs in
it, and that folder is the entire world it has.

It also means the files belong to the administrator's own account rather than
to a service account, which is what "accessible forever and from wherever"
actually requires: they are in a Drive a person owns and can open in a
browser, not in storage attached to a credential that stops existing when the
project does.

Setting it up is `scripts/connect_drive.py`, once, and three environment
variables after that.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

TOKEN_URL = "https://oauth2.googleapis.com/token"
FILES_URL = "https://www.googleapis.com/drive/v3/files"
UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files"

# Per-file access to files this app created, and nothing else in the Drive.
SCOPE = "https://www.googleapis.com/auth/drive.file"

FOLDER_NAME = "Rad-Intel reviewed radiographs"
FOLDER_MIME = "application/vnd.google-apps.folder"

TIMEOUT = 60.0
# A token is good for an hour; renewed a little early so a request never
# arrives with one that expired between the check and the call.
TOKEN_MARGIN = 120


class DriveError(RuntimeError):
    """Something went wrong talking to Drive. The message is for a person."""


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def configured() -> bool:
    return bool(_env("GOOGLE_DRIVE_CLIENT_ID")
                and _env("GOOGLE_DRIVE_CLIENT_SECRET")
                and _env("GOOGLE_DRIVE_REFRESH_TOKEN"))


class DriveStore:
    """
    The folder of reviewed radiographs, and the token that reaches it.

    One of these for the process. The access token and the folder id are
    cached here because both cost a round trip and neither changes often;
    both are rediscovered rather than stored on disk, so there is no third
    place for them to go stale.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at: float = 0.0
        self._folder_id: str | None = None

    # --- credentials -------------------------------------------------------

    def _access_token(self) -> str:
        import httpx

        with self._lock:
            if self._token and time.time() < self._expires_at - TOKEN_MARGIN:
                return self._token

        if not configured():
            raise DriveError(
                "Google Drive is not set up on this server. Run "
                "scripts/connect_drive.py once and put the three values it prints "
                "into the environment."
            )

        try:
            response = httpx.post(TOKEN_URL, timeout=TIMEOUT, data={
                "client_id": _env("GOOGLE_DRIVE_CLIENT_ID"),
                "client_secret": _env("GOOGLE_DRIVE_CLIENT_SECRET"),
                "refresh_token": _env("GOOGLE_DRIVE_REFRESH_TOKEN"),
                "grant_type": "refresh_token",
            })
        except Exception as exc:
            raise DriveError(f"Could not reach Google to refresh the token ({exc}).")

        if response.status_code != 200:
            # The refresh token is the one credential here that can be revoked
            # from the outside -- by changing the Google password, by removing
            # the app's access, or by leaving the OAuth app in Testing mode,
            # where Google expires it after seven days.
            raise DriveError(
                "Google refused the stored Drive credentials. Run "
                "scripts/connect_drive.py again to reconnect, and check the OAuth "
                "app is published rather than left in Testing — Google expires a "
                "testing app's token after seven days."
            )

        payload = response.json()
        with self._lock:
            self._token = payload["access_token"]
            self._expires_at = time.time() + float(payload.get("expires_in", 3600))
            return self._token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token()}"}

    # --- the folder --------------------------------------------------------

    def folder_id(self) -> str:
        """
        The folder radiographs go in, made if it is not there yet.

        A fixed id can be pinned with GOOGLE_DRIVE_FOLDER_ID, which is what to
        do when the folder already exists -- after restoring a server, say.
        Otherwise it is looked up by name and created once. The lookup only
        ever sees folders this app made, so it cannot adopt one of the
        administrator's own.
        """
        import httpx

        pinned = _env("GOOGLE_DRIVE_FOLDER_ID")
        if pinned:
            return pinned
        with self._lock:
            if self._folder_id:
                return self._folder_id

        query = (f"mimeType='{FOLDER_MIME}' and name='{FOLDER_NAME}' and trashed=false")
        found = httpx.get(FILES_URL, timeout=TIMEOUT, headers=self._headers(),
                          params={"q": query, "fields": "files(id,name)",
                                  "pageSize": 1})
        if found.status_code == 200 and found.json().get("files"):
            folder_id = found.json()["files"][0]["id"]
        else:
            made = httpx.post(FILES_URL, timeout=TIMEOUT, headers=self._headers(),
                              json={"name": FOLDER_NAME, "mimeType": FOLDER_MIME})
            if made.status_code not in (200, 201):
                raise DriveError(f"Could not create the Drive folder "
                                 f"({made.status_code}).")
            folder_id = made.json()["id"]
            logger.info("Created the Drive folder %r.", FOLDER_NAME)

        with self._lock:
            self._folder_id = folder_id
        return folder_id

    # --- files -------------------------------------------------------------

    def upload(self, path: Path, name: str, mime: str = "image/jpeg") -> str:
        """Put a file in the folder. Returns its Drive file id."""
        import httpx

        metadata = {"name": name, "parents": [self.folder_id()]}
        files = {
            "metadata": (None, json.dumps(metadata), "application/json; charset=UTF-8"),
            "file": (name, path.read_bytes(), mime),
        }
        response = httpx.post(UPLOAD_URL, timeout=TIMEOUT, headers=self._headers(),
                              params={"uploadType": "multipart", "fields": "id"},
                              files=files)
        if response.status_code not in (200, 201):
            raise DriveError(f"Drive refused the upload ({response.status_code}). "
                             f"Check the account still has room.")
        return response.json()["id"]

    def download(self, file_id: str, into: Path) -> bool:
        """
        Fetch a file back to disk. False when Drive no longer has it.

        This is what makes the disk a cache rather than the record: a server
        that came up with an empty volume gets the radiograph back the first
        time somebody asks to see it.
        """
        import httpx

        response = httpx.get(f"{FILES_URL}/{file_id}", timeout=TIMEOUT,
                             headers=self._headers(), params={"alt": "media"},
                             follow_redirects=True)
        if response.status_code == 404:
            return False
        if response.status_code != 200:
            raise DriveError(f"Drive would not return the file "
                             f"({response.status_code}).")
        into.parent.mkdir(parents=True, exist_ok=True)
        # Written beside and renamed, so a half-finished download is never
        # mistaken for the image itself.
        partial = into.with_suffix(into.suffix + ".partial")
        partial.write_bytes(response.content)
        partial.replace(into)
        return True

    def delete(self, file_id: str) -> bool:
        """Remove a file. False when it was already gone."""
        import httpx

        response = httpx.delete(f"{FILES_URL}/{file_id}", timeout=TIMEOUT,
                                headers=self._headers())
        return response.status_code in (200, 204, 404)

    def check(self) -> dict:
        """A summary for the administrator's console. Carries no credentials."""
        if not configured():
            return {"configured": False,
                    "detail": ("Reviewed radiographs are kept on this server's disk "
                               "only. If the disk is wiped they are gone. Run "
                               "scripts/connect_drive.py to put them in Drive as well.")}
        try:
            folder = self.folder_id()
        except DriveError as exc:
            return {"configured": True, "working": False, "detail": str(exc)}
        return {
            "configured": True,
            "working": True,
            "folder_id": folder,
            "folder_name": FOLDER_NAME,
            "folder_url": f"https://drive.google.com/drive/folders/{folder}",
            "detail": (f"Reviewed radiographs are copied to “{FOLDER_NAME}” in your "
                       f"Drive. This connection can see that folder and nothing else "
                       f"in your account."),
        }


store = DriveStore()
