"""
Email: sign-in codes, and telling the administrator about access requests.

One channel, by choice. SMS was supported and is gone: in India domestic
application-to-person messaging sits behind DLT registration under TRAI's
TCCCPR rules, providers will not carry it without that paperwork, and the one
that offers a way round it refuses its own free credit through the API until
a payment has been made. None of that bought anything email does not already
do here, and a second delivery path is a second thing that can quietly stop
working.

A code is never sent over WhatsApp either. The phone number on a clinician's
record is how the administrator reaches them personally; a one-time code does
not belong in a chat account that may be open on a shared laptop.

Delivery fails closed. If a code cannot be sent the sign-in is refused: the
code is cancelled, never returned in the response and never shown on screen.
A system that displays the code when its mail server is down has a second
factor in name only. The single exception has to be switched on by hand --
RAD_INTEL_OTP_CHANNEL=log writes codes to the server's own terminal, so the
workspace can be demonstrated on a laptop with nothing configured, and so an
administrator whose mail has broken has a way back in.
"""
from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage

from rad_intel.auth import mail

logger = logging.getLogger(__name__)

SMTP_TIMEOUT = 15

# Where access requests are sent. The study administrator's own address.
ADMIN_EMAIL = os.environ.get("RAD_INTEL_ADMIN_EMAIL", "poorabng@gmail.com").strip()


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


# --- what is configured ----------------------------------------------------

def email_configured() -> bool:
    return bool(_env("SMTP_HOST") and _env("SMTP_FROM"))


def echo_enabled() -> bool:
    """Whether codes may be written to the server log. Off unless explicitly set."""
    return _env("RAD_INTEL_OTP_ECHO") in ("1", "true", "yes", "on")


def channel_name() -> str:
    """
    Which channel a code would go out on right now.

    RAD_INTEL_OTP_CHANNEL=log pins it to the server's terminal and beats a
    working mail server, so an administrator can take that decision rather
    than having it depend on what happens to be configured.
    """
    if _env("RAD_INTEL_OTP_CHANNEL").lower() in ("log", "server-log", "console"):
        return "server-log"
    if email_configured():
        return "email"
    return "server-log" if echo_enabled() else "none"


def delivery_status() -> dict[str, object]:
    """A summary for the administrator's console. Contains no credentials."""
    return {
        "code_channel": channel_name(),
        "email_configured": email_configured(),
        "echo_to_server_log": echo_enabled(),
        "admin_email": ADMIN_EMAIL,
    }


# --- verification codes ----------------------------------------------------

def send_verification_code(code: str, full_name: str,
                           email: str | None = None) -> tuple[bool, str, str]:
    """
    Deliver a sign-in code. Returns (delivered, channel, message for the user).

    The returned message is shown to whoever is signing in, so it says where
    the code went -- never what it is.
    """
    channel = channel_name()
    # Derived, not written out. This said "10 minutes" while codes were
    # actually good for twelve hours, because the number lived in two places
    # and only one of them was changed.
    from rad_intel.auth.otp import CODE_TTL_SECONDS

    minutes = CODE_TTL_SECONDS // 60
    window = f"{minutes} minutes" if minutes < 120 else f"{minutes // 60} hours"

    if channel == "email":
        if not email:
            return False, "email", (
                "No email address is registered for this account. Contact the study "
                "administrator."
            )
        subject = "Your Rad-Intel sign-in code"
        body, html = mail.render(
            subject,
            f"Valid for {window}. One use only.",
            [
                mail.heading(f"Sign-in code for {full_name}"),
                mail.para("Type this into the page you are signing in from. It is the "
                          "second half of your sign-in; the first half was your "
                          "Rad-Intel ID or your password."),
                mail.code(code, caption=f"Valid for {window}, and for one sign-in only."),
                mail.callout(
                    "Nobody from the study will ever ask you for this code — not by "
                    "email, not on the phone, not over WhatsApp. If you did not just "
                    "try to sign in, ignore this message and tell the study "
                    "administrator, because somebody else has your Rad-Intel ID.",
                    tone="red",
                ),
                mail.small("Asking for another code cancels this one."),
            ],
        )
        sent, detail = send_email(subject=subject, body=body, to=email, html=html)
        if sent:
            local, _, domain = email.partition("@")
            masked = f"{local[:2]}{'*' * max(1, len(local) - 2)}@{domain}"
            return True, "email", f"Code emailed to {masked}. Check your spam folder too."
        logger.error("A sign-in code could not be emailed: %s", detail)
        return False, "email", (
            "The code could not be emailed. This is a problem with the study's mail "
            "settings, not with anything you did — tell the study administrator, who "
            "can see the reason."
        )

    if channel == "server-log":
        logger.warning(
            "Sign-in code for %s: %s  (RAD_INTEL_OTP_CHANNEL=log, or nothing is "
            "configured to deliver it; this is the server's own log only)",
            full_name, code,
        )
        return True, "server-log", (
            "Codes are being written to the server's terminal rather than emailed. "
            "This is a local test mode."
        )

    return False, "none", (
        "Verification codes cannot be sent: this server has no mail configured. "
        "Contact the study administrator."
    )


# --- email -----------------------------------------------------------------

def send_email(subject: str, body: str, to: str | None = None,
               html: str | None = None) -> tuple[bool, str]:
    """
    Send an email. Returns (sent, detail).

    `body` is the plain-text version and is always sent. `html` is an
    alternative, not a replacement: a client that will not or cannot render
    it falls back to the text, which says the same thing because both were
    built from the same blocks in mail.py.
    """
    recipient = (to or ADMIN_EMAIL).strip()
    if not email_configured():
        return False, "No SMTP server is configured, so no email was sent."

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = _env("SMTP_FROM")
    message["To"] = recipient
    message.set_content(body)
    if html:
        message.add_alternative(html, subtype="html")

    host, port = _env("SMTP_HOST"), int(_env("SMTP_PORT") or 587)
    user, password = _env("SMTP_USER"), _env("SMTP_PASSWORD")

    # A password and a one-time code both travel in this connection, so
    # neither may cross a network in the clear. STARTTLS is used whenever the
    # server offers it and required whenever anything secret is at stake; the
    # one exception is an unauthenticated relay on this same machine, where
    # there is no network to listen on.
    local_relay = host in ("localhost", "127.0.0.1", "::1")

    try:
        if port == 465:
            server = smtplib.SMTP_SSL(host, port, timeout=SMTP_TIMEOUT)
        else:
            server = smtplib.SMTP(host, port, timeout=SMTP_TIMEOUT)
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


def send_test_email(full_name: str = "") -> tuple[bool, str]:
    """
    Prove that mail works, from the administrator console.

    Worth its own message rather than a line of text: this is the one email
    an administrator deliberately looks at, so it is also where they find out
    whether their mail client renders what everybody else is going to get.
    """
    subject = "Rad-Intel: email delivery is working"
    body, html = mail.render(
        subject,
        "Access requests and sign-in codes will reach this address.",
        [
            mail.heading("Mail is working"),
            mail.para("You sent this to yourself from the Rad-Intel administrator "
                      "console. Reading it means two things are true."),
            mail.facts([
                ("Access requests", f"will arrive at {ADMIN_EMAIL}"),
                ("Sign-in codes", "can reach clinicians at their own addresses"),
            ]),
            mail.callout(
                "Mail is the second factor. If it stops working, nobody can sign in "
                "— including you. The way back in is the command line: "
                "scripts/manage_clinicians.py code --email … --admin prints a code "
                "that the “I already have a code” link accepts.",
                tone="amber",
            ),
            mail.small("This is also what the branding looks like in your own mail "
                       "client. What clinicians receive uses the same template."),
        ],
    )
    return send_email(subject=subject, body=body, html=html)


def send_access_request_email(full_name: str, email: str, institution: str,
                              phone: str | None, registration: str | None,
                              message: str | None, reference: str) -> tuple[bool, str]:
    """Tell the administrator that somebody has asked to join the study."""
    subject = f"Rad-Intel access request — {full_name} ({institution})"
    blocks = [
        mail.heading("Somebody has asked to join the study"),
        mail.para("They filled in the request form on the public site. No account "
                  "exists and nothing has been granted."),
        mail.facts([
            ("Name", full_name),
            ("Institution", institution),
            ("Email", email),
            ("Phone", phone or "not given"),
            ("Registration", registration or "not given"),
            ("Reference", reference),
        ]),
    ]
    if (message or "").strip():
        blocks += [mail.heading("What they wrote"), mail.para(message.strip())]

    blocks += [
        mail.callout(
            "Verify their medical registration yourself, against the register rather "
            "than against anything in this message. Everything above was typed by "
            "whoever filled in the form.",
            tone="amber",
        ),
        mail.divider(),
        mail.heading("To grant access"),
        mail.para("Open the administrator console, find this request, and press “Use "
                  "these details”. Or, from a terminal on the server:"),
        mail.command(
            f'python scripts/manage_clinicians.py add \\\n'
            f'  --email "{email}" \\\n'
            f'  --name "{full_name}" \\\n'
            f'  --institution "{institution}" \\\n'
            f'  --phone "{phone or ""}"'
        ),
        mail.small("Either way you are given a Rad-Intel ID to hand over. Send it to "
                   "them separately from the link to the site — the ID and their "
                   "mailbox are the two halves of their sign-in, and keeping them in "
                   "different places is the whole point."),
    ]
    body, html = mail.render(
        subject, f"{full_name} at {institution}. Nothing has been granted yet.", blocks)
    return send_email(subject=subject, body=body, html=html)
