"""
What the study's email looks like, and what it must never contain.

The rules worth holding:

* every message renders as both text and HTML, from the same blocks, so a
  client that shows one is not reading an older email than a client that
  shows the other;
* anything a stranger typed is escaped before it reaches the HTML, because an
  access request carries a name and a free-text message straight from a
  public form into the administrator's mailbox;
* nothing is fetched from anywhere when the message is opened -- no remote
  image, no tracking pixel, no redirect. A sign-in code should not report
  when it was read;
* a sign-in code appears in the message, and the code's own value never
  appears anywhere it should not.
"""

import re

import pytest

from rad_intel.auth import mail, notify


def flat(text: str) -> str:
    """One long line, so an assertion is not defeated by where a line wrapped."""
    return " ".join(text.replace(">", " ").split())


@pytest.fixture
def outbox(monkeypatch):
    """Capture what would be sent, with mail configured but nothing dialled."""
    sent: list[dict] = []

    def capture(subject, body, to=None, html=None):
        sent.append({"subject": subject, "body": body, "to": to, "html": html})
        return True, "captured"

    monkeypatch.setenv("SMTP_HOST", "smtp.example.org")
    monkeypatch.setenv("SMTP_FROM", "study@example.org")
    monkeypatch.delenv("RAD_INTEL_OTP_CHANNEL", raising=False)
    monkeypatch.setattr(notify, "send_email", capture)
    return sent


# --- the shape of a message -------------------------------------------------

def test_every_block_renders_as_text_and_as_a_page():
    blocks = [
        mail.heading("A heading"),
        mail.para("A paragraph."),
        mail.code("AB12CD34EF56", caption="Valid for 30 minutes."),
        mail.facts([("Name", "Dr A"), ("Institution", "A Hospital")]),
        mail.callout("Something to be careful about.", tone="red"),
        mail.command("python scripts/manage_clinicians.py add --email a@b.org"),
        mail.button("Open the workspace", "https://example.org/"),
        mail.small("A quieter note."),
        mail.divider(),
    ]
    text, html = mail.render("Subject line", "Preview line", blocks)

    for expected in ("A heading", "A paragraph.", "AB12CD34EF56", "Dr A",
                     "A Hospital", "Something to be careful about.",
                     "manage_clinicians.py", "https://example.org/",
                     "A quieter note."):
        assert expected in text, expected
        assert expected in html, expected

    assert html.lower().startswith("<!doctype html>")
    assert "Preview line" in html, "the inbox preview line has to be in the body"
    assert "Preview line" not in text, "it is a mail-client artefact, not content"


def test_no_style_attribute_is_broken_by_a_quote_inside_it():
    """
    This went wrong once and was invisible from the code.

    The font stack held a double-quoted family name, and every one of these
    sits inside style="...". The attribute closed at that quote, the rest of
    the declaration was dropped, and the whole message rendered in the mail
    client's default serif at default sizes. It looked like no styling had
    been applied at all.
    """
    _, html = mail.render("Subject", "Preview", [
        mail.heading("H"), mail.para("P"), mail.code("ABCDEF123456"),
        mail.facts([("a", "b")]), mail.callout("c"), mail.command("d"),
        mail.button("e", "https://example.org/"), mail.small("f"), mail.divider(),
    ])
    # A style attribute that names a font must still carry what was written
    # after it. If a quote closed the attribute early, it will not.
    styled = [a for a in re.findall(r'style="([^"]*)"', html) if "font-family" in a]
    assert styled, "the message is styled at all"
    for attribute in styled:
        assert "font-size" in attribute or "line-height" in attribute, attribute
    # And no stray double quote survives anywhere inside one.
    assert all('"' not in a for a in re.findall(r'style="([^"]*)"', html))


def test_a_message_fetches_nothing_when_it_is_opened():
    """
    No remote image, no tracking pixel, no click-through redirect.

    A sign-in code that reports when it was read is a sign-in code that tells
    somebody whether an address is live and when its owner is at their desk.
    """
    _, html = mail.render("Subject", "Preview", [
        mail.para("Body"), mail.code("ABCDEF123456"),
        mail.button("Go", "https://example.org/"),
    ])
    assert "<img" not in html.lower()
    assert "<script" not in html.lower()
    assert "background-image" not in html.lower()
    assert "url(" not in html.lower()
    # Exactly one link, the one that was asked for.
    assert re.findall(r'href="([^"]+)"', html) == ["https://example.org/"]


def test_what_a_stranger_typed_cannot_become_markup():
    """
    An access request carries a name and a free-text message from a public
    form into the administrator's mailbox.
    """
    nasty = '<script>alert(1)</script> "quoted" & <b>bold</b>'
    _, html = mail.render("Subject", "Preview", [
        mail.para(nasty),
        mail.facts([(nasty, nasty)]),
        mail.callout(nasty),
        mail.button(nasty, 'https://example.org/"onmouseover="alert(1)'),
    ])
    assert "<script>" not in html
    assert "<b>bold</b>" not in html
    assert "&lt;script&gt;" in html
    # The URL is quoted, so it cannot close the attribute and start another.
    assert 'onmouseover="alert' not in html


def test_the_plain_text_is_not_an_afterthought():
    text, _ = mail.render("Subject", "Preview", [
        mail.facts([("Name", "Dr A"), ("Institution", "A Very Long Hospital Name")]),
    ])
    lines = [line for line in text.splitlines() if "Dr A" in line or "Hospital" in line]
    # Aligned, so it reads as a table rather than a run-on.
    assert all(" : " in line for line in lines)
    assert "research and education system" in text, "the disclaimer is in both"


# --- the messages the study actually sends ----------------------------------

def test_a_sign_in_code_arrives_as_a_code_and_a_warning(outbox):
    ok, channel, shown = notify.send_verification_code("AB12CD34EF56", "Dr A",
                                                       "a@hospital.org")
    assert ok and channel == "email"
    message = outbox[0]

    assert message["to"] == "a@hospital.org"
    assert "AB12CD34EF56" in message["body"]
    assert "AB12CD34EF56" in message["html"]
    assert "Dr A" in message["html"]
    # The thing that stops the commonest attack on a code like this.
    assert "ever ask you for this code" in flat(message["body"])
    assert "ever ask you for this code" in message["html"]
    # And it never says what the first factor was.
    assert "a@hospital.org" not in message["html"].replace('a@hospital.org"', "")


def test_the_code_is_not_in_the_subject_or_the_inbox_preview(outbox):
    """Both are shown in a notification, on a lock screen, over a shoulder."""
    notify.send_verification_code("SECRETCODE12", "Dr A", "a@hospital.org")
    message = outbox[0]
    assert "SECRETCODE12" not in message["subject"]

    preheader = message["html"].split("mso-hide:all;\">", 1)[1].split("<", 1)[0]
    assert "SECRETCODE12" not in preheader


def test_an_access_request_says_nothing_has_been_granted(outbox):
    ok, _ = notify.send_access_request_email(
        full_name="Dr Hopeful", email="hope@clinic.org", institution="Some Clinic",
        phone="+91 99999 11111", registration="KMC-12345",
        message="I would like to take part.", reference="abc12345",
    )
    assert ok
    message = outbox[0]
    assert message["to"] is None, "it goes to the administrator's own address"
    for expected in ("Dr Hopeful", "Some Clinic", "hope@clinic.org", "KMC-12345",
                     "abc12345", "I would like to take part."):
        assert expected in flat(message["body"]), expected
        assert expected in message["html"], expected

    assert "nothing has been granted" in flat(message["body"])
    assert "Verify their medical registration yourself" in message["html"]
    # The command to run is there, with their details already in it.
    assert "manage_clinicians.py add" in flat(message["body"])


def test_an_access_request_with_no_message_does_not_show_an_empty_section(outbox):
    notify.send_access_request_email(
        full_name="Dr Quiet", email="q@clinic.org", institution="Clinic",
        phone=None, registration=None, message="   ", reference="ref00001",
    )
    assert "What they wrote" not in outbox[0]["html"]
    assert "not given" in flat(outbox[0]["body"]), "an absent field says so"


def test_the_test_email_shows_the_administrator_what_everyone_else_gets(outbox):
    ok, _ = notify.send_test_email("Study Admin")
    assert ok
    message = outbox[0]
    assert "working" in message["subject"]
    assert notify.ADMIN_EMAIL in message["html"]
    # It is also where the way back in is written down.
    assert "manage_clinicians.py code" in flat(message["body"])
