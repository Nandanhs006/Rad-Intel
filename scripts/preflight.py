r"""
Is this server actually ready? Run it on the server, before you tell anyone.

Everything that can quietly be wrong at the same time, checked in one go and
reported in plain language. Each line is one of:

    OK       fine, nothing to do
    FIX      this will hurt somebody — the site is not ready
    CHECK    probably fine, but only you can say

It exits non-zero if anything says FIX, so it can also be a deploy gate.

    python scripts/preflight.py
    python scripts/preflight.py --email      also tries to send you a test email
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

OK, FIX, CHECK = "OK", "FIX", "CHECK"
_COLOUR = {OK: "\033[32m", FIX: "\033[31m", CHECK: "\033[33m"}
_RESET = "\033[0m"

results: list[tuple[str, str, str, str]] = []


def say(status: str, area: str, finding: str, action: str = "") -> None:
    results.append((status, area, finding, action))


def env(name: str) -> str:
    return os.environ.get(name, "").strip()


# --- the checks -------------------------------------------------------------

def check_python() -> None:
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 11):
        say(FIX, "Python", f"Running {major}.{minor}.",
            "This needs 3.11 or newer. Set the host's Python version.")
    else:
        say(OK, "Python", f"{major}.{minor}")


def check_torch() -> None:
    try:
        import torch
    except Exception as exc:
        say(FIX, "PyTorch", f"Not importable ({exc}).",
            "The install did not finish. Check the build log.")
        return

    say(OK, "PyTorch", f"{torch.__version__} on {'cuda' if torch.cuda.is_available() else 'cpu'}")
    # A CUDA build on a host with no GPU is two or three gigabytes of wheels
    # that will never run. It is not wrong, it is just expensive and slow to
    # deploy, and the CPU wheels are a flag away.
    if "+cu" in torch.__version__ and not torch.cuda.is_available():
        say(CHECK, "PyTorch", "A CUDA build is installed but there is no GPU.",
            "Install from requirements-deploy.txt to get the CPU wheels and save "
            "a few GB of image.")


def check_model() -> None:
    from rad_intel.api.dependencies import model_manager

    checkpoint = model_manager.active_checkpoint("hybrid")
    if not checkpoint or not Path(checkpoint).is_file():
        say(FIX, "Model", "No trained checkpoint was found.",
            "Every analysis will fail. Put best_hybrid_model.pt where "
            "MODEL_WEIGHTS_PATH points, or in weights/.")
        return
    size = Path(checkpoint).stat().st_size / (1024 * 1024)
    if size < 50:
        say(FIX, "Model", f"{Path(checkpoint).name} is only {size:.0f} MB.",
            "That is not a whole checkpoint — the upload was probably truncated.")
        return
    say(OK, "Model", f"{Path(checkpoint).name} ({size:.0f} MB)")


def check_secret_key() -> None:
    if env("RAD_INTEL_SECRET_KEY"):
        if len(env("RAD_INTEL_SECRET_KEY")) < 32:
            say(FIX, "Secret key", "Set, but short.",
                "Use 64 hex characters: python -c \"import secrets; "
                "print(secrets.token_hex(32))\"")
        else:
            say(OK, "Secret key", "Set from the environment.")
        return
    say(FIX, "Secret key", "RAD_INTEL_SECRET_KEY is not set.",
        "A random one is being written to .radintel_key instead. On a host that "
        "wipes its disk, that is a new key on every deploy — every stored phone "
        "number becomes unreadable and everyone is logged out. Set it and keep a "
        "copy somewhere safe.")


def check_cookies() -> None:
    if env("RAD_INTEL_COOKIE_SECURE") in ("1", "true", "yes"):
        say(OK, "Session cookie", "Marked secure; it will not travel over plain HTTP.")
    else:
        say(FIX, "Session cookie", "RAD_INTEL_COOKIE_SECURE is not set.",
            "Set it to 1. Without it the session cookie can be sent over plain "
            "HTTP, which is how a session gets stolen on a shared network.")


def check_proxy() -> None:
    if env("RAD_INTEL_TRUSTED_PROXY") in ("1", "true", "yes"):
        say(OK, "Caller address", "X-Forwarded-For is trusted.")
    else:
        say(CHECK, "Caller address", "X-Forwarded-For is ignored.",
            "Right on a machine people reach directly. Wrong behind Railway, "
            "Render, Cloudflare or any load balancer: every request then looks "
            "like it came from the proxy, so one person can use up everybody's "
            "rate limit and the activity log records the wrong address. Set "
            "RAD_INTEL_TRUSTED_PROXY=1 if you are behind one.")


def check_docs_and_cors() -> None:
    if env("RAD_INTEL_ENABLE_DOCS"):
        say(FIX, "API docs", "/docs is switched on.",
            "It lists every endpoint to anyone who asks. Empty "
            "RAD_INTEL_ENABLE_DOCS.")
    else:
        say(OK, "API docs", "/docs is off.")

    origins = env("RAD_INTEL_CORS_ORIGINS")
    if "*" in origins:
        say(FIX, "CORS", "A wildcard origin is configured.",
            "With credentials allowed, that lets any site make requests as a "
            "logged-in clinician. Name the origins, or leave it empty.")
    elif origins:
        say(OK, "CORS", f"Limited to {origins}")
    else:
        say(OK, "CORS", "No cross-origin access, which is what a single site wants.")


def check_mail(send: bool) -> None:
    from rad_intel.auth import notify

    channel = notify.channel_name()
    if channel == "server-log":
        say(FIX, "Email", "Codes are being written to the server log.",
            "Nobody can sign in unless they can read your logs. Configure SMTP, "
            "and empty RAD_INTEL_OTP_CHANNEL.")
        return
    if channel == "none":
        say(FIX, "Email", "No way to send a sign-in code.",
            "Set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD and SMTP_FROM. "
            "For Gmail the password is a 16-character App Password.")
        return

    say(OK, "Email", f"Configured; the administrator is {notify.ADMIN_EMAIL}.")
    if notify.echo_enabled():
        say(FIX, "Email", "RAD_INTEL_OTP_ECHO is on.",
            "Codes are being written to the log as well as emailed. Empty it.")
    if send:
        sent, detail = notify.send_test_email()
        say(OK if sent else FIX, "Email", detail,
            "" if sent else "Fix this before anyone tries to sign in.")


def check_storage() -> None:
    from rad_intel.api.routes.feedback import FEEDBACK_DIR
    from rad_intel.config import settings

    try:
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        probe = FEEDBACK_DIR / ".preflight"
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
    except Exception as exc:
        say(FIX, "Review store", f"{FEEDBACK_DIR} is not writable ({exc}).",
            "Set RAD_INTEL_FEEDBACK_DIR to a directory on the mounted volume.")
        return

    inside_project = str(FEEDBACK_DIR).startswith(str(settings.BASE_DIR))
    if inside_project and not env("RAD_INTEL_FEEDBACK_DIR"):
        say(CHECK, "Review store", f"{FEEDBACK_DIR}",
            "That is inside the code directory. On a host that rebuilds the "
            "container, everything in it is lost on the next deploy. Point "
            "RAD_INTEL_FEEDBACK_DIR at the mounted volume.")
    else:
        say(OK, "Review store", str(FEEDBACK_DIR))


def check_database() -> None:
    from rad_intel.auth import accounts

    try:
        people = accounts.list_clinicians()
    except Exception as exc:
        say(FIX, "Database", f"Could not be read ({exc}).",
            "Check DATABASE_URL and that the volume is mounted.")
        return

    admins = [p for p in people if p.get("role") == "admin" and p["status"] == "active"]
    if not admins:
        say(FIX, "Administrator", "No active administrator account on this server.",
            'Create one: python scripts/manage_clinicians.py add --role admin '
            '--email you@example.org --name "Your Name" --institution "Your '
            'Institution" --phone "+91 ..."')
    else:
        say(OK, "Administrator", f"{len(admins)} account(s); "
                                 f"{len(people)} people in total.")


def check_drive() -> None:
    from rad_intel.storage import drive

    if not drive.configured():
        say(CHECK, "Radiograph backup", "Not connected to Drive.",
            "Reviewed radiographs will exist only on this disk, and most hosts "
            "wipe it on redeploy. A radiograph a clinician uploaded cannot be "
            "made again. See Step 4b in DEPLOY.md.")
        return
    status = drive.store.check()
    if status.get("working"):
        say(OK, "Radiograph backup", f"Copied to “{status['folder_name']}” in Drive.")
    else:
        say(FIX, "Radiograph backup", status.get("detail", "Drive is not answering."),
            "Reconnect with scripts/connect_drive.py.")


def check_reporting() -> None:
    if env("GEMINI_API_KEY"):
        say(OK, "Reports", "A language model is configured.")
    else:
        say(CHECK, "Reports", "No GEMINI_API_KEY.",
            "Reports still work — they fall back to the built-in rule-based "
            "writer. They are just plainer.")


# --- main -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--email", action="store_true",
                    help="also send a real test email to the administrator")
    args = ap.parse_args()

    print("\nRad-Intel preflight\n" + "=" * 70)

    for check in (check_python, check_torch, check_model, check_secret_key,
                  check_cookies, check_proxy, check_docs_and_cors, check_storage,
                  check_database, check_drive, check_reporting):
        try:
            check()
        except Exception as exc:
            say(FIX, check.__name__.replace("check_", "").title(),
                f"The check itself failed: {type(exc).__name__}: {exc}")
    try:
        check_mail(args.email)
    except Exception as exc:
        say(FIX, "Email", f"The check itself failed: {exc}")

    colour = sys.stdout.isatty()
    for status, area, finding, action in results:
        tag = f"{_COLOUR[status]}{status:<5}{_RESET}" if colour else f"{status:<5}"
        print(f"\n{tag} {area}\n      {finding}")
        if action:
            for line in _wrapped(action):
                print(f"      → {line}")

    problems = [r for r in results if r[0] == FIX]
    worth_a_look = [r for r in results if r[0] == CHECK]
    print("\n" + "=" * 70)
    if problems:
        print(f"{len(problems)} thing(s) to fix before anyone uses this.")
    elif worth_a_look:
        print(f"Nothing broken. {len(worth_a_look)} thing(s) only you can decide.")
    else:
        print("Ready.")
    print()
    sys.exit(1 if problems else 0)


def _wrapped(text: str, width: int = 64) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width) or [""]


if __name__ == "__main__":
    main()
