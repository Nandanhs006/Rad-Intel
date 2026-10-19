r"""
Connect Rad-Intel to your Google Drive. Run this once, on your own machine.

What it does
------------
Asks Google, in your browser, for permission to put files in your Drive --
and only for the one narrow permission it actually needs. It prints three
values; you put those in the server's environment and never run this again.

What it is asking for
---------------------
One scope: `drive.file`. Google defines it as per-file access to files the
app itself created. It is not a promise -- the API will not return anything
else. Your own documents, your photographs, the trained checkpoints you keep
in the same Drive: this connection cannot read them, cannot list them, cannot
tell they exist. It makes one folder and that folder is all it has.

Before you run it
-----------------
1. Open console.cloud.google.com and make a project (any name).
2. APIs & Services -> Library -> search "Google Drive API" -> Enable.
3. APIs & Services -> OAuth consent screen:
     User type: External
     Fill in the app name, your email, your email again. Save.
     Publish app -> Confirm.
   *Publishing matters.* An app left in Testing has its refresh token
   expired by Google after seven days, and uploads start failing a week
   after you deploy. `drive.file` is not a sensitive scope, so publishing
   needs no review and no verification -- it is one button.
4. APIs & Services -> Credentials -> Create credentials ->
   OAuth client ID -> Desktop app. Copy the client ID and client secret.

Then
----
    python scripts/connect_drive.py --client-id XXX --client-secret YYY
"""
from __future__ import annotations

import argparse
import http.server
import secrets
import socket
import sys
import threading
import urllib.parse
import webbrowser

import httpx

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPE = "https://www.googleapis.com/auth/drive.file"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class _Catcher(http.server.BaseHTTPRequestHandler):
    """Catches the one redirect Google sends back, and says so in the tab."""

    code: str | None = None
    state: str = ""
    error: str | None = None

    @classmethod
    def _record_callback(cls, path: str) -> bool:
        """Record Google's redirect; return False for unrelated browser requests."""
        parsed = urllib.parse.urlparse(path)
        # A browser commonly asks for /favicon.ico immediately after Google
        # returns to /. It is not another OAuth response and must not replace
        # the successful callback with a missing-state error.
        if parsed.path != "/":
            return False

        query = urllib.parse.parse_qs(parsed.query)
        if query.get("state", [""])[0] != cls.state:
            cls.error = "The reply did not match the request. Nothing was saved."
        elif "error" in query:
            cls.error = query["error"][0]
        else:
            cls.code = query.get("code", [None])[0]
        return True

    def do_GET(self):  # noqa: N802
        if not _Catcher._record_callback(self.path):
            self.send_response(204)
            self.end_headers()
            return

        done = _Catcher.code is not None
        body = (
            "<!doctype html><meta charset='utf-8'>"
            "<body style=\"margin:0;display:grid;place-items:center;height:100vh;"
            "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;"
            "background:#f1f5f9;color:#0f172a;\">"
            "<div style='text-align:center;max-width:420px;padding:28px;'>"
            f"<div style='font-size:19px;font-weight:650;margin-bottom:8px;'>"
            f"{'Connected.' if done else 'Not connected.'}</div>"
            f"<div style='font-size:14px;line-height:1.6;color:#64748b;'>"
            f"{'Go back to your terminal.' if done else (_Catcher.error or 'Try again.')}"
            "</div></div></body>"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--client-id", required=True)
    ap.add_argument("--client-secret", required=True)
    args = ap.parse_args()

    port = _free_port()
    redirect = f"http://127.0.0.1:{port}/"
    _Catcher.state = secrets.token_urlsafe(24)

    url = AUTH_URL + "?" + urllib.parse.urlencode({
        "client_id": args.client_id,
        "redirect_uri": redirect,
        "response_type": "code",
        "scope": SCOPE,
        # Both are needed to be handed a refresh token rather than only an
        # access token that dies in an hour.
        "access_type": "offline",
        "prompt": "consent",
        "state": _Catcher.state,
    })

    server = http.server.HTTPServer(("127.0.0.1", port), _Catcher)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print("\nOpening your browser. Sign in as the account whose Drive should hold")
    print("the reviewed radiographs, and allow the one permission it asks for.")
    print(f"\nIf nothing opens, paste this in yourself:\n\n  {url}\n")
    webbrowser.open(url)

    while _Catcher.code is None and _Catcher.error is None:
        import time
        time.sleep(0.3)
    server.shutdown()

    if _Catcher.error:
        sys.exit(f"\nGoogle said: {_Catcher.error}\nNothing was saved.")

    response = httpx.post(TOKEN_URL, timeout=30, data={
        "code": _Catcher.code,
        "client_id": args.client_id,
        "client_secret": args.client_secret,
        "redirect_uri": redirect,
        "grant_type": "authorization_code",
    })
    if response.status_code != 200:
        sys.exit(f"\nGoogle refused to issue a token ({response.status_code}):\n"
                 f"{response.text}")

    refresh = response.json().get("refresh_token")
    if not refresh:
        sys.exit("\nGoogle returned no refresh token. This happens when the account "
                 "has already granted this app before: remove Rad-Intel at "
                 "myaccount.google.com/permissions and run this again.")

    print("\n" + "=" * 72)
    print("Connected. Put these three in the server's environment, then restart it.")
    print("=" * 72 + "\n")
    print(f"GOOGLE_DRIVE_CLIENT_ID={args.client_id}")
    print(f"GOOGLE_DRIVE_CLIENT_SECRET={args.client_secret}")
    print(f"GOOGLE_DRIVE_REFRESH_TOKEN={refresh}")
    print("\nTreat the last two like passwords. Anything holding them can write to")
    print("the one folder this app made in your Drive — and to nothing else in it.")
    print("\nTo disconnect later: myaccount.google.com/permissions.\n")


if __name__ == "__main__":
    main()
