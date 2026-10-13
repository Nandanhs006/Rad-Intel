"""
Grounded question answering for the educational assistant.

The flow is: classify the question, retrieve supporting entries from the
curated knowledge base, ask the language model to answer *from those entries
only*, then screen what comes back. If the model is unavailable or its answer
fails screening, the knowledge-base extract is returned directly. That
deterministic path is always available, so the feature degrades to "slightly
drier answers" rather than to an error.

This is kept entirely separate from the radiograph pipeline. Nothing here
touches a model checkpoint, a tensor, or an uploaded image, and the classifier
result is only ever included when the caller explicitly passes it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from rad_intel.assistant import safety
from rad_intel.assistant.knowledge_base import KBEntry, retrieve
from rad_intel.config import settings

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Educational information only; not medical diagnosis. "
    "Seek qualified medical care for urgent or personal concerns."
)

_SYSTEM_PROMPT = """You answer general educational questions about lung health, pneumonia, \
chest radiography, and the Rad-Intel research project.

Absolute rules:
- Answer ONLY from the reference extracts supplied in the user message. If they do not \
cover the question, say so plainly and suggest what the person could ask instead.
- Never tell the user they have, might have, or probably have any condition.
- Never interpret anyone's personal radiograph, symptoms or test result.
- Never recommend, name or dose a medication, and never give a treatment plan.
- Never claim clinical certainty. Describe what is generally understood, not what is \
true of the reader.
- Do not invent statistics, study results or sources.

Style: plain English, 2 to 5 short sentences or a brief list. No preamble, no sign-off, \
no emoji. Do not restate the disclaimer; the interface already shows it."""


@dataclass
class ChatAnswer:
    answer: str
    sources: list[dict[str, str]]
    mode: str          # gemini | knowledge_base | emergency | refusal
    disclaimer: str = DISCLAIMER
    is_emergency: bool = False


def _sources_payload(entries: list[KBEntry]) -> list[dict[str, str]]:
    return [{"title": e.title, "name": e.source_name, "url": e.source_url} for e in entries]


def _kb_answer(entries: list[KBEntry]) -> str:
    """Deterministic answer assembled straight from the knowledge base."""
    if not entries:
        return (
            "I don't have a reference entry covering that. I can explain what pneumonia is, "
            "how it tends to be described, what a chest X-ray shows, what the heatmap and "
            "the metrics in this interface mean, or what this project can and cannot do."
        )
    parts = [f"**{e.title}.** {e.body}" for e in entries[:2]]
    return "\n\n".join(parts)


def _build_user_prompt(question: str, entries: list[KBEntry], app_context: str | None) -> str:
    refs = "\n\n".join(
        f"[{i + 1}] {e.title} (source: {e.source_name})\n{e.body}"
        for i, e in enumerate(entries)
    )
    ctx = ""
    if app_context:
        ctx = (
            "\n\nThe user has explicitly asked about the result currently displayed in the "
            "application. These are the recorded model outputs; describe them as model "
            "outputs, never as findings about a person:\n" + app_context
        )
    return f"REFERENCE EXTRACTS:\n{refs}{ctx}\n\nQUESTION: {question}\n\nAnswer from the extracts above."


def _gemini_answer(question: str, entries: list[KBEntry], app_context: str | None) -> str | None:
    """Ask Gemini to phrase an answer from the extracts. None if unavailable or unusable."""
    if not settings.GEMINI_API_KEY:
        return None
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        resp = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=_build_user_prompt(question, entries, app_context),
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=400,
            ),
        )
        text = (getattr(resp, "text", "") or "").strip()
        if not text:
            return None

        violations = safety.output_violations(text)
        if violations:
            logger.warning("Assistant answer rejected, forbidden phrasing: %s", violations)
            return None
        return text
    except Exception as exc:
        logger.warning("Assistant language model unavailable: %s", exc)
        return None


def answer_question(question: str, app_context: str | None = None) -> ChatAnswer:
    """
    Answer an educational question.

    `app_context` is a short, caller-supplied description of the on-screen
    result. It is only ever populated when the user explicitly asks about the
    displayed analysis, and it carries no image and no identifying field.
    """
    question = (question or "").strip()
    if not question:
        return ChatAnswer("Ask me something about lungs, pneumonia or this project.", [], "refusal")

    route = safety.classify(question)

    if route == "emergency":
        entry = retrieve("urgent warning signs seek care", k=1)
        return ChatAnswer(safety.EMERGENCY_NOTICE, _sources_payload(entry), "emergency",
                          is_emergency=True)

    if route == "personal":
        return ChatAnswer(safety.PERSONAL_NOTICE, [], "refusal")

    if route == "out_of_scope":
        return ChatAnswer(safety.SCOPE_NOTICE, [], "refusal")

    entries = retrieve(question, k=3)
    if not entries:
        return ChatAnswer(_kb_answer([]), [], "knowledge_base")

    text = _gemini_answer(question, entries, app_context)
    if text:
        return ChatAnswer(text, _sources_payload(entries), "gemini")
    return ChatAnswer(_kb_answer(entries), _sources_payload(entries), "knowledge_base")
