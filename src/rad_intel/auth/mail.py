"""
What Rad-Intel's email looks like.

Every message the study sends is built here, out of blocks, and rendered
twice: once as HTML for a mail client that will show it, once as plain text
for one that will not. Both come from the same list of blocks, so the two
cannot drift apart -- which is the usual way a "plain text version" ends up
describing an older email than the one people actually read.

The HTML is written the way email has to be written rather than the way a web
page is: tables for layout, every style inline, no external stylesheet, no
remote images, nothing that depends on JavaScript. Mail clients are twenty
years behind browsers and several of them rewrite what they are given.
Nothing here is decoration for its own sake -- the shape exists so that the
one thing a message is about (a code, a name to verify) is the thing the eye
lands on.

There are no tracking pixels and no click-through redirects. A sign-in code
should not tell anyone when it was read.
"""
from __future__ import annotations

from html import escape

# The site's own palette, since this is the same product arriving by post.
ORANGE = "#f97316"
ORANGE_DARK = "#ea580c"
INK = "#0f172a"
MUTED = "#64748b"
LIGHT = "#94a3b8"
PAGE = "#f1f5f9"
CARD = "#ffffff"
BORDER = "#e2e8f0"
AMBER_BG = "#fff7ed"
AMBER_BORDER = "#fed7aa"
RED_BG = "#fef2f2"
RED_BORDER = "#fecaca"
RED_INK = "#b91c1c"
GREEN_BG = "#ecfdf5"
GREEN_BORDER = "#a7f3d0"
GREEN_INK = "#047857"

# Single quotes inside, deliberately. Every one of these goes into a
# style="..." attribute, and a double quote in the font stack closes that
# attribute early -- which silently drops the rest of the declaration and
# leaves the mail rendering in the client's default serif.
FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, "
        "Arial, sans-serif")
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Courier New', monospace"

PRODUCT = "Rad-Intel"
TAGLINE = "Pneumonia detection from chest radiographs"
# Said in every message, because a reader two hops removed from the study has
# no other way to know it.
DISCLAIMER = ("Rad-Intel is a research and education system. It is not a medical "
              "device and is not for clinical use.")

_TONES = {
    "amber": (AMBER_BG, AMBER_BORDER, "#9a3412"),
    "red": (RED_BG, RED_BORDER, RED_INK),
    "green": (GREEN_BG, GREEN_BORDER, GREEN_INK),
    "grey": ("#f8fafc", BORDER, MUTED),
}


# --- blocks -----------------------------------------------------------------
#
# A message is a list of these. Each knows how to be a paragraph of text and
# how to be a piece of a page, and nothing else in this module has to know
# which kind it is looking at.

def heading(text: str) -> dict:
    return {"kind": "heading", "text": text}


def para(text: str) -> dict:
    return {"kind": "para", "text": text}


def code(value: str, caption: str | None = None) -> dict:
    """The one thing the message exists to deliver. Big, spaced, selectable."""
    return {"kind": "code", "value": value, "caption": caption}


def facts(rows: list[tuple[str, str]]) -> dict:
    """Label/value pairs, e.g. the details on an access request."""
    return {"kind": "facts", "rows": rows}


def callout(text: str, tone: str = "amber") -> dict:
    return {"kind": "callout", "text": text, "tone": tone}


def command(text: str) -> dict:
    """Something to paste into a terminal."""
    return {"kind": "command", "text": text}


def button(label: str, url: str) -> dict:
    return {"kind": "button", "label": label, "url": url}


def small(text: str) -> dict:
    return {"kind": "small", "text": text}


def divider() -> dict:
    return {"kind": "divider"}


# --- plain text -------------------------------------------------------------

def _wrap(text: str, width: int = 72, indent: str = "") -> str:
    import textwrap

    return "\n".join(
        textwrap.fill(line, width=width, initial_indent=indent,
                      subsequent_indent=indent) if line.strip() else ""
        for line in text.splitlines()
    )


def to_text(subject: str, blocks: list[dict]) -> str:
    out: list[str] = [subject, "=" * min(len(subject), 72), ""]
    for block in blocks:
        kind = block["kind"]
        if kind == "heading":
            out += [block["text"], "-" * min(len(block["text"]), 72), ""]
        elif kind == "para":
            out += [_wrap(block["text"]), ""]
        elif kind == "code":
            if block.get("caption"):
                out += [_wrap(block["caption"])]
            out += ["", f"    {block['value']}", ""]
        elif kind == "facts":
            width = max((len(label) for label, _ in block["rows"]), default=0)
            out += [f"  {label.ljust(width)} : {value}" for label, value in block["rows"]]
            out += [""]
        elif kind == "callout":
            out += [_wrap(block["text"], width=68, indent="  > "), ""]
        elif kind == "command":
            out += [*(f"    {line}" for line in block["text"].splitlines()), ""]
        elif kind == "button":
            out += [f"{block['label']}: {block['url']}", ""]
        elif kind == "small":
            out += [_wrap(block["text"]), ""]
        elif kind == "divider":
            out += ["-" * 72, ""]

    out += ["-" * 72, _wrap(DISCLAIMER), f"{PRODUCT} — {TAGLINE}"]
    return "\n".join(out).rstrip() + "\n"


# --- html -------------------------------------------------------------------

def _html_blocks(blocks: list[dict]) -> str:
    parts: list[str] = []
    for block in blocks:
        kind = block["kind"]

        if kind == "heading":
            parts.append(
                f'<h2 style="margin:0 0 10px;font-family:{FONT};font-size:17px;'
                f'line-height:1.35;font-weight:650;color:{INK};'
                f'letter-spacing:-0.01em;">{escape(block["text"])}</h2>'
            )

        elif kind == "para":
            parts.append(
                f'<p style="margin:0 0 16px;font-family:{FONT};font-size:15px;'
                f'line-height:1.6;color:{INK};">{escape(block["text"])}</p>'
            )

        elif kind == "small":
            parts.append(
                f'<p style="margin:0 0 14px;font-family:{FONT};font-size:13px;'
                f'line-height:1.6;color:{MUTED};">{escape(block["text"])}</p>'
            )

        elif kind == "code":
            caption = (
                f'<p style="margin:0 0 10px;font-family:{FONT};font-size:13px;'
                f'line-height:1.5;color:{MUTED};">{escape(block["caption"])}</p>'
                if block.get("caption") else ""
            )
            parts.append(
                f'{caption}'
                f'<table role="presentation" cellpadding="0" cellspacing="0" border="0" '
                f'width="100%" style="margin:0 0 18px;"><tr>'
                f'<td align="center" style="background:{AMBER_BG};'
                f'border:1px solid {AMBER_BORDER};border-radius:12px;padding:22px 16px;">'
                f'<div style="font-family:{MONO};font-size:27px;line-height:1.2;'
                f'font-weight:700;letter-spacing:3px;color:{INK};'
                f'word-break:break-all;">{escape(block["value"])}</div>'
                f'</td></tr></table>'
            )

        elif kind == "facts":
            rows = "".join(
                f'<tr>'
                f'<td style="padding:9px 14px 9px 0;font-family:{FONT};font-size:13px;'
                f'line-height:1.5;color:{MUTED};white-space:nowrap;vertical-align:top;'
                f'border-bottom:1px solid {BORDER};">{escape(label)}</td>'
                f'<td style="padding:9px 0;font-family:{FONT};font-size:14px;'
                f'line-height:1.5;color:{INK};font-weight:550;vertical-align:top;'
                f'border-bottom:1px solid {BORDER};">{escape(value)}</td>'
                f'</tr>'
                for label, value in block["rows"]
            )
            parts.append(
                f'<table role="presentation" cellpadding="0" cellspacing="0" border="0" '
                f'width="100%" style="margin:0 0 20px;border-collapse:collapse;">'
                f'{rows}</table>'
            )

        elif kind == "callout":
            bg, border, ink = _TONES.get(block["tone"], _TONES["amber"])
            parts.append(
                f'<table role="presentation" cellpadding="0" cellspacing="0" border="0" '
                f'width="100%" style="margin:0 0 18px;"><tr>'
                f'<td style="background:{bg};border:1px solid {border};'
                f'border-left:3px solid {border};border-radius:10px;padding:14px 16px;'
                f'font-family:{FONT};font-size:14px;line-height:1.55;color:{ink};">'
                f'{escape(block["text"])}</td></tr></table>'
            )

        elif kind == "command":
            parts.append(
                f'<table role="presentation" cellpadding="0" cellspacing="0" border="0" '
                f'width="100%" style="margin:0 0 18px;"><tr>'
                f'<td style="background:#0f172a;border-radius:10px;padding:14px 16px;">'
                f'<pre style="margin:0;font-family:{MONO};font-size:12px;line-height:1.6;'
                f'color:#cbd5e1;white-space:pre-wrap;word-break:break-word;">'
                f'{escape(block["text"])}</pre></td></tr></table>'
            )

        elif kind == "button":
            parts.append(
                f'<table role="presentation" cellpadding="0" cellspacing="0" border="0" '
                f'style="margin:0 0 20px;"><tr>'
                f'<td align="center" style="background:{ORANGE};border-radius:10px;">'
                f'<a href="{escape(block["url"], quote=True)}" '
                f'style="display:inline-block;padding:12px 26px;font-family:{FONT};'
                f'font-size:15px;font-weight:600;color:#ffffff;text-decoration:none;">'
                f'{escape(block["label"])}</a></td></tr></table>'
            )

        elif kind == "divider":
            parts.append(
                f'<div style="height:1px;background:{BORDER};margin:4px 0 20px;"></div>'
            )

    return "".join(parts)


def to_html(subject: str, preheader: str, blocks: list[dict]) -> str:
    """
    The message as a page.

    `preheader` is the line a mail client shows beside the subject in the
    inbox list. It is in the body, hidden, because there is nowhere else to
    put it -- and padded, or the client fills the rest of the preview with
    whatever text comes next.
    """
    hidden_preheader = (
        f'<div style="display:none;max-height:0;overflow:hidden;opacity:0;'
        f'mso-hide:all;">{escape(preheader)}'
        f'{"&#847;&zwnj;&nbsp;" * 60}</div>'
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light">
<meta name="supported-color-schemes" content="light">
<title>{escape(subject)}</title>
</head>
<body style="margin:0;padding:0;background:{PAGE};-webkit-font-smoothing:antialiased;">
{hidden_preheader}
<table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%"
       style="background:{PAGE};">
  <tr>
    <td align="center" style="padding:28px 14px 40px;">

      <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="600"
             style="width:100%;max-width:600px;">

        <!-- wordmark -->
        <tr><td style="padding:0 4px 16px;">
          <table role="presentation" cellpadding="0" cellspacing="0" border="0"><tr>
            <td style="background:{ORANGE};border-radius:9px;width:30px;height:30px;"
                align="center" valign="middle">
              <span style="font-family:{FONT};font-size:14px;font-weight:700;
                           color:#ffffff;line-height:30px;">R</span>
            </td>
            <td style="padding-left:10px;font-family:{FONT};font-size:17px;
                       font-weight:700;color:{INK};letter-spacing:-0.02em;">
              Rad&#8209;Intel
            </td>
          </tr></table>
        </td></tr>

        <!-- the card -->
        <tr><td style="background:{CARD};border:1px solid {BORDER};
                       border-top:3px solid {ORANGE};border-radius:16px;
                       padding:30px 30px 22px;">
          {_html_blocks(blocks)}
        </td></tr>

        <!-- footer -->
        <tr><td style="padding:20px 10px 0;">
          <p style="margin:0 0 6px;font-family:{FONT};font-size:12px;line-height:1.6;
                    color:{MUTED};">{escape(DISCLAIMER)}</p>
          <p style="margin:0;font-family:{FONT};font-size:12px;line-height:1.6;
                    color:{LIGHT};">{PRODUCT} &mdash; {escape(TAGLINE)}</p>
        </td></tr>

      </table>
    </td>
  </tr>
</table>
</body>
</html>
"""


def render(subject: str, preheader: str, blocks: list[dict]) -> tuple[str, str]:
    """A message, as (plain text, html). Both from the same blocks."""
    return to_text(subject, blocks), to_html(subject, preheader, blocks)
