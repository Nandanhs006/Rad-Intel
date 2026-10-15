"""
Delivery of verification codes and administrator notifications.

Two channels, both optional, both configured only through the environment:

* **A phone**, for one-time sign-in codes. WhatsApp when a WhatsApp sender is
  configured, otherwise SMS. Twilio's REST API is called directly over
  urllib rather than through their SDK, so this adds no dependency.
* **Email**, for telling the administrator that somebody has asked for access.
  Plain SMTP over STARTTLS.

Both fail closed. If a code cannot be delivered, the sign-in is refused and
the code is cancelled -- never shown in the response, never written to a log,
never handed to the browser. A system that falls back to showing the code
on screen when its SMS provider is down has a second factor in name only.

The one exception is deliberate and has to be switched on by hand:
RAD_INTEL_OTP_ECHO=1 writes the code to the *server's own* log, so the
workspace can be demonstrated on a laptop with no SMS account. It is off
unless set, it is loud about itself, and it never puts the code anywhere the
network can see.
"""
from __future__ import annotations

import base64
import logging
import os
import smtplib
import urllib.error
import urllib.parse
import urllib.request
from email.message import EmailMessage

logger = logging.getLogger(__name__)

HTTP_TIMEOUT = 10

# Where access requests are sent. The study administrator's own address.
ADMIN_EMAIL = os.environ.get("RAD_INTEL_ADMIN_EMAIL", "poorabng@gmail.com").strip()


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


# --- capability reporting --------------------------------------------------

def sms_configured() -> bool:
    return bool(_env("TWILIO_ACCOUNT_SID") and _env("TWILIO_AUTH_TOKEN")
                and (_env("TWILIO_FROM") or _env("TWILIO_WHATSAPP_FROM")))


def email_configured() -> bool:
    return bool(_env("SMTP_HOST") and _env("SMTP_FROM"))


def echo_enabled() -> bool:
    """Whether codes may be written to the server log. Off unless explicitly set."""
    return _env("RAD_INTEL_OTP_ECHO") in ("1", "true", "yes", "on")


def channel_name() -> str:
    if _env("TWILIO_WHATSAPP_FROM"):
        return "whatsapp"
    if _env("TWILIO_FROM"):
        return "sms"
    return "server-log" if echo_enabled() else "none"


def delivery_status() -> dict[str, object]:
    """A summary for the administrator's console. Contains no credentials."""
    return {
        "code_channel": channel_name(),
        "sms_configured": sms_configured(),
        "email_configured": email_configured(),
        "echo_to_server_log": echo_enabled(),
        "admin_email": ADMIN_EMAIL,
    }


# --- phone ----------------------------------------------------------------

def _twilio_send(to: str, body: str) -> tuple[bool, str]:
    sid, token = _env("TWILIO_ACCOUNT_SID"), _env("TWILIO_AUTH_TOKEN")
    whatsapp_from, sms_from = _env("TWILIO_WHATSAPP_FROM"), _env("TWILIO_FROM")

    if whatsapp_from:
        sender, recipient = whatsapp_from, to if to.startswith("whatsapp:") else f"whatsapp:{to}"
    else:
        sender, recipient = sms_from, to

    payload = urllib.parse.urlencode({"To": recipient, "From": sender, "Body": body}).encode()
    request = urllib.request.Request(
        f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
        data=payload, method="POST",
    )
    credentials = base64.b64encode(f"{sid}:{token}".encode()).decode("ascii")
    request.add_header("Authorization", f"Basic {credentials}")
    request.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            if 200 <= response.status < 300:
                return True, f"Sent via {'WhatsApp' if whatsapp_from else 'SMS'}."
            return False, f"Message provider returned HTTP {response.status}."
    except urllib.error.HTTPError as exc:
        # The provider's body can name the account; keep it out of the log.
        logger.error("Verification code delivery rejected with HTTP %s.", exc.code)
        return False, f"Message provider rejected the request (HTTP {exc.code})."
    except Exception as exc:
        logger.error("Verification code delivery failed: %s", type(exc).__name__)
        return False, "Could not reach the message provider."


def send_verification_code(phone: str | None, code: str, full_name: str) -> tuple[bool, str, str]:
    """
    Deliver a sign-in code. Returns (delivered, channel, message for the user).

    The returned message is shown to whoever is signing in, so it says where
    the code went -- never what it is.
    """
    if not phone:
        return False, "none", (
            "No phone number is registered for this account, so a verification code "
            "cannot be sent. Contact the study administrator."
        )

    body = (f"Rad-Intel sign-in code for {full_name}: {code}\n"
            f"Valid for 10 minutes. One use only. If you did not request it, ignore "
            f"this message and tell the study administrator.")

    if sms_configured():
        ok, detail = _twilio_send(phone, body)
        channel = channel_name()
        if ok:
            tail = "".join(ch for ch in phone if ch.isdigit())[-4:]
            return True, channel, f"Verification code sent to the number ending {tail}."
        return False, channel, f"The code could not be delivered. {detail}"

    if echo_enabled():
        logger.warning(
            "RAD_INTEL_OTP_ECHO is on. Sign-in code for %s: %s  "
            "(no message provider configured; this appears in the server log only)",
            full_name, code,
        )
        return True, "server-log", (
            "No message provider is configured, so the code was written to the server "
            "log instead of being sent to your phone. This is a local test mode."
        )

    return False, "none", (
        "Verification codes cannot be sent: this server has no message provider "
        "configured. Contact the study administrator."
    )


# --- email ----------------------------------------------------------------

def send_email(subject: str, body: str, to: str | None = None) -> tuple[bool, str]:
    """Send a plain-text email. Returns (sent, detail)."""
    recipient = (to or ADMIN_EMAIL).strip()
    if not email_configured():
        return False, "No SMTP server is configured, so no email was sent."

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = _env("SMTP_FROM")
    message["To"] = recipient
    message.set_content(body)

    host, port = _env("SMTP_HOST"), int(_env("SMTP_PORT") or 587)
    user, password = _env("SMTP_USER"), _env("SMTP_PASSWORD")

    try:
        if port == 465:
            server = smtplib.SMTP_SSL(host, port, timeout=HTTP_TIMEOUT)
        else:
            server = smtplib.SMTP(host, port, timeout=HTTP_TIMEOUT)
        with server:
            if port != 465:
                server.starttls()
            if user and password:
                server.login(user, password)
            server.send_message(message)
        return True, f"Emailed {recipient}."
    except Exception as exc:
        logger.error("Could not send email to the administrator: %s", type(exc).__name__)
        return False, f"Email could not be sent ({type(exc).__name__})."


def send_access_request_email(full_name: str, email: str, institution: str,
                              phone: str | None, registration: str | None,
                              message: str | None, reference: str) -> tuple[bool, str]:
    """Tell the administrator that somebody has asked to join the study."""
    body = (
        "Somebody has requested access to the Rad-Intel clinician review workspace.\n\n"
        f"  Name         : {full_name}\n"
        f"  Institution  : {institution}\n"
        f"  Email        : {email}\n"
        f"  Phone        : {phone or 'not given'}\n"
        f"  Registration : {registration or 'not given'}\n"
        f"  Reference    : {reference}\n\n"
        f"  Message      : {message or '(none)'}\n\n"
        "Verify their registration independently before granting access. Nothing has\n"
        "been granted: no account exists until you create one.\n\n"
        "Approve from the administrator console in the workspace, or on the command line:\n\n"
        f'  python scripts/manage_clinicians.py add --email "{email}" \\\n'
        f'      --name "{full_name}" --institution "{institution}" --phone "{phone or ""}"\n\n'
        "That prints the Rad-Intel ID to pass to them yourself.\n"
    )
    return send_email(
        subject=f"Rad-Intel access request — {full_name} ({institution})",
        body=body,
    )
