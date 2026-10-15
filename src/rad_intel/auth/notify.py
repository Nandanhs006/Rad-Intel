"""
Delivery of verification codes and administrator notifications.

Codes go by **SMS or email**, never over WhatsApp. The phone number on a
clinician's record has two separate jobs: it is where their sign-in code is
texted, and it is how the administrator reaches them personally — which is
what the "WhatsApp preferred" note on the request form is about. Pushing a
one-time code through a chat account somebody may be signed into on a shared
laptop is a different thing entirely, so that route does not exist here.

Three channels, resolved in this order:

* **SMS**, when a provider is configured. Twilio for numbers outside India;
  Fast2SMS for Indian ones, because India's TCCCPR rules put domestic
  application-to-person SMS behind DLT registration and Twilio will not carry
  it without that paperwork.
* **Email**, when SMTP is configured and SMS is not. A code sent to the
  address on the account is still a second factor — the first factor is the
  Rad-Intel ID, which is a different secret held in a different place — and
  it costs nothing and needs no regulatory registration. This is the
  reasonable default for a study.
* **The server's own log**, only when RAD_INTEL_OTP_ECHO is set by hand, so
  the workspace can be demonstrated on a laptop with nothing configured.

Everything fails closed. If a code cannot be delivered the sign-in is refused:
the code is cancelled, never returned in the response and never shown on
screen. A system that displays the code when its provider is down has a
second factor in name only.
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


# --- what is configured ----------------------------------------------------

def twilio_configured() -> bool:
    return bool(_env("TWILIO_ACCOUNT_SID") and _env("TWILIO_AUTH_TOKEN")
                and _env("TWILIO_FROM"))


def fast2sms_configured() -> bool:
    return bool(_env("FAST2SMS_API_KEY"))


def sms_configured() -> bool:
    return twilio_configured() or fast2sms_configured()


def email_configured() -> bool:
    return bool(_env("SMTP_HOST") and _env("SMTP_FROM"))


def echo_enabled() -> bool:
    """Whether codes may be written to the server log. Off unless explicitly set."""
    return _env("RAD_INTEL_OTP_ECHO") in ("1", "true", "yes", "on")


def channel_name() -> str:
    """
    Which channel a code would go out on right now.

    RAD_INTEL_OTP_CHANNEL pins it to "sms" or "email" when both are available;
    otherwise the best configured option wins.
    """
    preference = _env("RAD_INTEL_OTP_CHANNEL").lower()
    if preference in ("log", "server-log", "console"):
        # Chosen deliberately, so it wins over a configured provider. The only
        # people who can read a code are the ones who can already read the
        # server's terminal, which on a laptop demonstration is the point.
        return "server-log"
    if preference == "sms" and sms_configured():
        return "sms"
    if preference == "email" and email_configured():
        return "email"
    if sms_configured():
        return "sms"
    if email_configured():
        return "email"
    return "server-log" if echo_enabled() else "none"


def sms_provider() -> str:
    """Fast2SMS is preferred when both are set: it is the one that works in India."""
    if fast2sms_configured():
        return "fast2sms"
    return "twilio" if twilio_configured() else "none"


def delivery_status() -> dict[str, object]:
    """A summary for the administrator's console. Contains no credentials."""
    return {
        "code_channel": channel_name(),
        "sms_provider": sms_provider(),
        "sms_configured": sms_configured(),
        "email_configured": email_configured(),
        "echo_to_server_log": echo_enabled(),
        "admin_email": ADMIN_EMAIL,
    }


# --- SMS -------------------------------------------------------------------

def _twilio_send(to: str, body: str) -> tuple[bool, str]:
    """Twilio. Fine outside India; needs DLT registration for Indian numbers."""
    sid, token = _env("TWILIO_ACCOUNT_SID"), _env("TWILIO_AUTH_TOKEN")
    payload = urllib.parse.urlencode(
        {"To": to, "From": _env("TWILIO_FROM"), "Body": body}
    ).encode()

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
                return True, "Sent by SMS."
            return False, f"The SMS provider returned HTTP {response.status}."
    except urllib.error.HTTPError as exc:
        # The provider's body can name the account; keep it out of the log.
        logger.error("Twilio rejected a verification code with HTTP %s.", exc.code)
        return False, f"The SMS provider rejected the request (HTTP {exc.code})."
    except Exception as exc:
        logger.error("Twilio could not be reached: %s", type(exc).__name__)
        return False, "Could not reach the SMS provider."


def _fast2sms_send(to: str, body: str) -> tuple[bool, str]:
    """
    Fast2SMS. Indian numbers only, given as ten digits with no country code.

    The Quick route carries arbitrary text, which this needs: the dedicated
    OTP route uses a fixed numeric template and would not carry a twelve
    character code with letters and symbols in it.
    """
    digits = "".join(ch for ch in to if ch.isdigit())
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    if len(digits) != 10:
        return False, ("Fast2SMS only sends to Indian numbers, as ten digits. "
                       f"The number on this account is not one of those.")

    payload = urllib.parse.urlencode({
        "route": _env("FAST2SMS_ROUTE") or "q",
        "message": body,
        "language": "english",
        "flash": "0",
        "numbers": digits,
    }).encode()

    request = urllib.request.Request("https://www.fast2sms.com/dev/bulkV2",
                                     data=payload, method="POST")
    request.add_header("authorization", _env("FAST2SMS_API_KEY"))
    request.add_header("Content-Type", "application/x-www-form-urlencoded")

    def _reason(raw: str) -> str:
        """
        Fast2SMS explains itself in the body, including on a 400.

        Worth pulling out: its refusals are account-state problems with
        specific remedies -- an unverified website, or the free credit not
        being usable through the API until one payment has been made -- and
        "the provider refused the message" sends an administrator looking in
        entirely the wrong place.
        """
        try:
            import json
            return str(json.loads(raw).get("message") or raw)[:300]
        except Exception:
            return raw[:300]

    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            raw = response.read().decode("utf-8", "replace")
            if 200 <= response.status < 300 and '"return":true' in raw.replace(" ", ""):
                return True, "Sent by SMS."
            logger.error("Fast2SMS refused a message: %s", _reason(raw))
            return False, f"Fast2SMS refused it: {_reason(raw)}"
    except urllib.error.HTTPError as exc:
        reason = _reason(exc.read().decode("utf-8", "replace"))
        logger.error("Fast2SMS rejected a message (HTTP %s): %s", exc.code, reason)
        return False, f"Fast2SMS refused it: {reason}"
    except Exception as exc:
        logger.error("Fast2SMS could not be reached: %s", type(exc).__name__)
        return False, "Could not reach the SMS provider."


def _send_sms(to: str, body: str) -> tuple[bool, str]:
    provider = sms_provider()
    if provider == "fast2sms":
        return _fast2sms_send(to, body)
    if provider == "twilio":
        return _twilio_send(to, body)
    return False, "No SMS provider is configured."


# --- verification codes ----------------------------------------------------

def _code_body(code: str, full_name: str) -> str:
    return (f"Rad-Intel sign-in code for {full_name}: {code}\n"
            f"Valid for 10 minutes. One use only. If you did not ask for it, ignore "
            f"this and tell the study administrator.")


def send_verification_code(phone: str | None, code: str, full_name: str,
                           email: str | None = None) -> tuple[bool, str, str]:
    """
    Deliver a sign-in code. Returns (delivered, channel, message for the user).

    The returned message is shown to whoever is signing in, so it says where
    the code went — never what it is.
    """
    channel = channel_name()
    body = _code_body(code, full_name)

    if channel == "sms":
        if not phone:
            return False, "sms", (
                "No phone number is registered for this account, so a code cannot be "
                "texted to you. Contact the study administrator."
            )
        sent, detail = _send_sms(phone, body)
        if sent:
            tail = "".join(ch for ch in phone if ch.isdigit())[-4:]
            return True, "sms", f"Code texted to the number ending {tail}."
        # The provider's own words can describe the study's billing or
        # verification state, which is the administrator's business and not
        # the business of whoever is standing at the login form.
        logger.error("A sign-in code could not be texted: %s", detail)
        return False, "sms", (
            "The code could not be texted to you. This is a problem with the study's "
            "message provider, not with anything you did — tell the study "
            "administrator, who can see the reason."
        )

    if channel == "email":
        if not email:
            return False, "email", (
                "No email address is registered for this account. Contact the study "
                "administrator."
            )
        sent, detail = send_email(
            subject="Your Rad-Intel sign-in code",
            body=body + ("\n\nThis code was emailed because the study is not using SMS "
                         "for sign-in codes. It is still single use and still expires "
                         "in ten minutes.\n"),
            to=email,
        )
        if sent:
            local, _, domain = email.partition("@")
            masked = f"{local[:2]}{'*' * max(1, len(local) - 2)}@{domain}"
            return True, "email", f"Code emailed to {masked}. Check your spam folder too."
        return False, "email", f"The code could not be sent. {detail}"

    if channel == "server-log":
        logger.warning(
            "RAD_INTEL_OTP_ECHO is on. Sign-in code for %s: %s  "
            "(nothing is configured to deliver it; this is the server log only)",
            full_name, code,
        )
        return True, "server-log", (
            "Nothing is configured to send codes, so it was written to the server log "
            "instead. This is a local test mode."
        )

    return False, "none", (
        "Verification codes cannot be sent: this server has no SMS provider and no "
        "email configured. Contact the study administrator."
    )


def send_test_sms(phone: str, full_name: str) -> tuple[bool, str]:
    """
    Text the administrator, reporting the provider's own words on failure.

    The counterpart to the test email: somewhere the real reason an SMS was
    refused is visible, rather than only in the server's log.
    """
    if not sms_configured():
        return False, ("No SMS provider is configured. Set FAST2SMS_API_KEY (India) or "
                       "the TWILIO_* values, and RAD_INTEL_OTP_CHANNEL=sms.")
    if not phone:
        return False, "No phone number is on your account to text."
    return _send_sms(phone, f"Rad-Intel delivery test for {full_name}. "
                            f"If you are reading this, sign-in codes will arrive.")


# --- email -----------------------------------------------------------------

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

    # A password and, on the email channel, a one-time code both travel in
    # this connection, so neither may cross a network in the clear. STARTTLS
    # is used whenever the server offers it and required whenever anything
    # secret is at stake; the one exception is an unauthenticated relay on
    # this same machine, where there is no network to listen on.
    local_relay = host in ("localhost", "127.0.0.1", "::1")

    try:
        if port == 465:
            server = smtplib.SMTP_SSL(host, port, timeout=HTTP_TIMEOUT)
        else:
            server = smtplib.SMTP(host, port, timeout=HTTP_TIMEOUT)
        with server:
            if port != 465:
                server.ehlo_or_helo_if_needed()
                if server.has_extn("starttls"):
                    server.starttls()
                    server.ehlo()
                elif user or password or not local_relay:
                    return False, (
                        f"{host} does not offer STARTTLS, so the connection would be "
                        f"in the clear. Refusing to send a password or a sign-in code "
                        f"over it. Use port 587 or 465 on a server that supports TLS."
                    )
            if user and password:
                server.login(user, password)
            server.send_message(message)
        return True, f"Emailed {recipient}."
    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP rejected the credentials.")
        return False, ("The mail server rejected the username or password. For Gmail "
                       "this must be a 16-character App Password, not your account "
                       "password and not a backup code.")
    except Exception as exc:
        logger.error("Could not send email: %s", type(exc).__name__)
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
