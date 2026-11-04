"""Durable storage for reviewed radiographs in the administrator's Drive.

The server keeps a local working copy, but the backup is deliberately
separate. An unavailable backup must never make a clinician lose a completed
review; the route records it locally and an administrator can backfill it.

The preferred connector is a Google Apps Script web app that executes as the
administrator. It writes only to the one folder chosen in that script, so
there is no Google OAuth refresh token on the server to expire. The old
``drive.file`` OAuth connector remains as a fallback for existing installs.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TOKEN_URL = "https://oauth2.googleapis.com/token"
FILES_URL = "https://www.googleapis.com/drive/v3/files"
UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files"

# Per-file access to files this app created, and nothing else in the Drive.
SCOPE = "https://www.googleapis.com/auth/drive.file"
FOLDER_NAME = "Rad-Intel reviewed radiographs"
FOLDER_MIME = "application/vnd.google-apps.folder"
TIMEOUT = 60.0
TOKEN_MARGIN = 120


class DriveError(RuntimeError):
    """Something went wrong talking to durable radiograph storage."""


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _apps_script_configured() -> bool:
    return bool(_env("GOOGLE_DRIVE_APPS_SCRIPT_URL")
                and _env("GOOGLE_DRIVE_APPS_SCRIPT_SECRET"))


def _oauth_configured() -> bool:
    return bool(_env("GOOGLE_DRIVE_CLIENT_ID")
                and _env("GOOGLE_DRIVE_CLIENT_SECRET")
                and _env("GOOGLE_DRIVE_REFRESH_TOKEN"))


def configured() -> bool:
    """True when either supported connector is fully configured."""
    return _apps_script_configured() or _oauth_configured()


class DriveStore:
    """The administrator's backup folder and the connector that reaches it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at = 0.0
        self._folder_id: str | None = None

    @property
    def _uses_apps_script(self) -> bool:
        return _apps_script_configured()

    def _apps_script_call(self, action: str, **payload: Any) -> dict[str, Any]:
        """Call the account-owned Apps Script gateway.

        Every short-lived request is HMAC-signed with a secret known only to
        this server and the script. Apps Script sends ContentService output
        through a Google redirect, hence ``follow_redirects=True``.
        """
        import httpx

        request: dict[str, Any] = {
            "action": action,
            "timestamp": int(time.time()),
            "nonce": secrets.token_hex(16),
            **payload,
        }
        canonical = json.dumps(request, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=True)
        request["signature"] = base64.b64encode(hmac.new(
            _env("GOOGLE_DRIVE_APPS_SCRIPT_SECRET").encode("utf-8"),
            canonical.encode("utf-8"), hashlib.sha256,
        ).digest()).decode("ascii")
        try:
            response = httpx.post(
                _env("GOOGLE_DRIVE_APPS_SCRIPT_URL"), json=request,
                timeout=TIMEOUT, follow_redirects=True,
            )
        except Exception as exc:
            raise DriveError(f"Could not reach the Google Drive backup ({exc}).") from exc
        if response.status_code != 200:
            raise DriveError(f"The Google Drive backup returned HTTP {response.status_code}.")
        try:
            result = response.json()
        except ValueError as exc:
            raise DriveError("The Google Drive backup returned an invalid response.") from exc
        if not isinstance(result, dict) or result.get("ok") is not True:
            detail = result.get("error") if isinstance(result, dict) else None
            raise DriveError(str(detail or "The Google Drive backup rejected the request."))
        return result

    # --- legacy OAuth -------------------------------------------------------

    def _access_token(self) -> str:
        """Return a legacy OAuth access token, refreshing it only when needed."""
        import httpx

        with self._lock:
            if self._token and time.time() < self._expires_at - TOKEN_MARGIN:
                return self._token
        if not _oauth_configured():
            raise DriveError("Google Drive is not configured.")
        try:
            response = httpx.post(TOKEN_URL, timeout=TIMEOUT, data={
                "client_id": _env("GOOGLE_DRIVE_CLIENT_ID"),
                "client_secret": _env("GOOGLE_DRIVE_CLIENT_SECRET"),
                "refresh_token": _env("GOOGLE_DRIVE_REFRESH_TOKEN"),
                "grant_type": "refresh_token",
            })
        except Exception as exc:
            raise DriveError(f"Could not reach Google to refresh the token ({exc}).") from exc
        if response.status_code != 200:
            raise DriveError(
                "Google refused the stored Drive credentials. Reconnect with "
                "scripts/connect_drive.py, or configure the Apps Script backup in DEPLOY.md."
            )
        payload = response.json()
        token = payload.get("access_token")
        if not token:
            raise DriveError("Google returned no access token for the Drive connection.")
        with self._lock:
            self._token = token
            self._expires_at = time.time() + float(payload.get("expires_in", 3600))
            return token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token()}"}

    # --- folder -------------------------------------------------------------

    def folder_id(self) -> str:
        """Return the backup folder id, discovering it only for legacy OAuth."""
        import httpx

        if self._uses_apps_script:
            folder_id = self._apps_script_call("status").get("folder_id")
            if not isinstance(folder_id, str) or not folder_id:
                raise DriveError("The Google Drive backup did not identify its folder.")
            return folder_id
        with self._lock:
            if self._folder_id:
                return self._folder_id
        query = f"mimeType='{FOLDER_MIME}' and name='{FOLDER_NAME}' and trashed=false"
        found = httpx.get(FILES_URL, timeout=TIMEOUT, headers=self._headers(),
                          params={"q": query, "fields": "files(id,name)", "pageSize": 1})
        if found.status_code == 200 and found.json().get("files"):
            folder_id = found.json()["files"][0]["id"]
        else:
            made = httpx.post(FILES_URL, timeout=TIMEOUT, headers=self._headers(),
                              json={"name": FOLDER_NAME, "mimeType": FOLDER_MIME})
            if made.status_code not in (200, 201):
                raise DriveError(f"Could not create the Drive folder ({made.status_code}).")
            folder_id = made.json()["id"]
            logger.info("Created the Drive folder %r.", FOLDER_NAME)
        with self._lock:
            self._folder_id = folder_id
        return folder_id

    # --- files --------------------------------------------------------------

    def upload(self, path: Path, name: str, mime: str = "image/jpeg") -> str:
        """Put a file in durable storage and return its provider-specific id."""
        import httpx

        if self._uses_apps_script:
            result = self._apps_script_call(
                "upload", name=name, mime=mime,
                data_b64=base64.b64encode(path.read_bytes()).decode("ascii"),
            )
            file_id = result.get("file_id")
            if not isinstance(file_id, str) or not file_id:
                raise DriveError("The Google Drive backup did not return a file id.")
            return file_id
        metadata = {"name": name, "parents": [self.folder_id()]}
        files = {
            "metadata": (None, json.dumps(metadata), "application/json; charset=UTF-8"),
            "file": (name, path.read_bytes(), mime),
        }
        response = httpx.post(UPLOAD_URL, timeout=TIMEOUT, headers=self._headers(),
                              params={"uploadType": "multipart", "fields": "id"}, files=files)
        if response.status_code not in (200, 201):
            raise DriveError(f"Drive refused the upload ({response.status_code}). Check free space.")
        return response.json()["id"]

    def download(self, file_id: str, into: Path) -> bool:
        """Fetch a file back to disk. False means it no longer exists."""
        import httpx

        if self._uses_apps_script:
            result = self._apps_script_call("download", file_id=file_id)
            if result.get("found") is False:
                return False
            encoded = result.get("data_b64")
            if not isinstance(encoded, str):
                raise DriveError("The Google Drive backup returned no file data.")
            try:
                content = base64.b64decode(encoded, validate=True)
            except binascii.Error as exc:
                raise DriveError("The Google Drive backup returned invalid file data.") from exc
        else:
            response = httpx.get(f"{FILES_URL}/{file_id}", timeout=TIMEOUT,
                                 headers=self._headers(), params={"alt": "media"},
                                 follow_redirects=True)
            if response.status_code == 404:
                return False
            if response.status_code != 200:
                raise DriveError(f"Drive would not return the file ({response.status_code}).")
            content = response.content
        into.parent.mkdir(parents=True, exist_ok=True)
        partial = into.with_suffix(into.suffix + ".partial")
        partial.write_bytes(content)
        partial.replace(into)
        return True

    def delete(self, file_id: str) -> bool:
        """Remove a durable copy. Apps Script moves it to the owner's Drive bin."""
        import httpx

        if self._uses_apps_script:
            return bool(self._apps_script_call("delete", file_id=file_id).get("deleted"))
        response = httpx.delete(f"{FILES_URL}/{file_id}", timeout=TIMEOUT,
                                headers=self._headers())
        return response.status_code in (200, 204, 404)

    def check(self) -> dict[str, Any]:
        """A credential-free summary for the administrator console."""
        if not configured():
            return {"configured": False, "detail": (
                "Reviewed radiographs currently exist only on this server. Configure "
                "the durable Drive backup in Step 4b of DEPLOY.md.")}
        try:
            folder = self.folder_id()
        except DriveError as exc:
            return {"configured": True, "working": False, "detail": str(exc)}
        connector = "Apps Script" if self._uses_apps_script else "OAuth"
        return {
            "configured": True, "working": True, "folder_id": folder,
            "folder_name": FOLDER_NAME,
            "folder_url": f"https://drive.google.com/drive/folders/{folder}",
            "detail": (f"Reviewed radiographs are copied to “{FOLDER_NAME}” in your "
                       f"Drive through the {connector} backup."),
        }


store = DriveStore()
