"""
Safety boundaries for the educational assistant.

Three checks, applied in order:

1. Emergency language in the question short-circuits everything. If someone is
   describing severe breathing difficulty or blue lips, the useful response is
   a short instruction to get help now, not a paragraph about alveoli.
2. Out-of-scope questions are declined before any model call, so the assistant
   cannot be walked away from lung health one turn at a time.
3. Generated text is screened before display. The prompt forbids diagnosis and
   prescription; this catches the cases where it does it anyway.

The screens are intentionally blunt. A false positive costs a redirect to a
clinician, which is a safe failure; a false negative puts a diagnostic-sounding
sentence in front of a user, which is not.
"""
from __future__ import annotations

import re

EMERGENCY_NOTICE = (
    "**This needs urgent medical attention, not a chatbot.**\n\n"
    "What you are describing is on the list of signs that health services say should be "
    "acted on immediately. Please contact your local emergency number or go to the "
    "nearest emergency department now. If someone is struggling to breathe, is turning "
    "blue around the lips or face, or is hard to rouse, treat it as an emergency.\n\n"
    "I am an educational tool and cannot assess anyone's condition."
)

SCOPE_NOTICE = (
    "I can only help with general questions about lung health, pneumonia, chest X-rays, "
    "the terms used in this interface, and how this project works. That one falls outside "
    "what I cover."
)

PERSONAL_NOTICE = (
    "I can't assess your situation, interpret your radiograph, or tell you what is wrong "
    "with you — I have no access to your medical history and no clinical standing. A "
    "qualified clinician is the right place for that.\n\n"
    "I can explain the general topic if that would help."
)

# Signs health services name as reasons to seek immediate care.
_EMERGENCY_PATTERNS = (
    r"\bcan'?t breathe\b", r"\bcannot breathe\b", r"\bunable to breathe\b",
    r"\bstruggling to breathe\b", r"\bgasping\b", r"\bsuffocat\w*",
    r"\bsevere(?:ly)?\s+(?:short(?:ness)? of breath|breathless|difficulty breathing)\b",
    r"\bshortness of breath\b.{0,30}\b(?:severe|worsening|sudden|bad)\b",
    r"\b(?:severe|crushing|sharp|bad)\s+chest pain\b", r"\bchest pain\b.{0,25}\b(?:severe|worse|spreading)\b",
    r"\bblue\s+(?:lips|face|fingers|nails|skin)\b", r"\blips?\s+(?:are |is |turning )?blue\b",
    r"\bcyanosis\b", r"\bturning blue\b",
    r"\bconfus(?:ed|ion)\b", r"\bunrespons\w*", r"\bpassed out\b", r"\bfaint(?:ed|ing)\b",
    r"\bcollaps(?:ed|ing)\b", r"\bunconscious\b", r"\bhard to (?:wake|rouse)\b",
    r"\bcoughing (?:up )?blood\b", r"\bblood in (?:my |the )?(?:sputum|phlegm|cough)\b",
    r"\bhaemoptysis\b", r"\bhemoptysis\b",
    r"\bgetting (?:much )?worse\b", r"\bworsening (?:fast|rapidly|quickly)\b",
    r"\bemergency\b", r"\bambulance\b", r"\bdying\b", r"\bgoing to die\b",
    r"\bbaby\b.{0,40}\b(?:not breathing|breathing fast|indrawing|grunting)\b",
    r"\b(?:not breathing|stopped breathing)\b",
)

# First-person or second-person framing that asks for a personal verdict.
_PERSONAL_PATTERNS = (
    r"\bdo i have\b", r"\bdo you think i\b", r"\bam i\b.{0,20}\b(?:sick|ill|infected|ok|okay|fine)\b",
    r"\bis (?:my|this) (?:x-?ray|scan|radiograph|image|result)\b",
    r"\bwhat'?s wrong with me\b", r"\bdiagnose me\b", r"\bmy (?:symptoms|condition|illness)\b",
    r"\bshould i (?:take|use|start|stop)\b", r"\bwhat (?:medicine|medication|antibiotic|drug|dose)\b",
    r"\bprescri\w*", r"\bhow much .{0,20}should i take\b",
    r"\bis it serious\b", r"\bdo i need (?:to see|a doctor|antibiotics|treatment)\b",
)

# Topic gate. A question must touch at least one of these to be answered.
_IN_SCOPE = (
    "lung", "lungs", "pneumonia", "chest", "x-ray", "xray", "radiograph", "radiology",
    "breath", "breathing", "respiratory", "cough", "alveol", "opacity", "consolidation",
    "infection", "infect", "fever", "oxygen", "airway", "pleural", "thorax", "thoracic",
    "bronch", "sputum", "vaccine", "vaccination", "grad-cam", "gradcam", "heatmap",
    "saliency", "lime", "model", "accuracy", "auroc", "sensitivity", "specificity",
    "threshold", "dataset", "rad-intel", "radintel", "this project", "this app",
    "this system", "this tool", "classifier", "prediction", "confidence", "cbam",
    "densenet", "swin", "resnet", "explanation", "limitation", "report", "summary",
)

# Phrases that must not appear in generated output.
_FORBIDDEN_OUTPUT = (
    r"\byou (?:have|likely have|probably have|appear to have|may have)\s+pneumonia\b",
    r"\byour (?:x-?ray|scan|radiograph|image)\s+shows\b",
    r"\bi diagnose\b", r"\byou should take\b", r"\bi recommend (?:taking|starting)\b",
    r"\bstart (?:taking )?(?:antibiotics|amoxicillin|azithromycin)\b",
    r"\bprescribe\b", r"\byou are (?:definitely|certainly) \w+",
)

_EMERGENCY_RE = re.compile("|".join(_EMERGENCY_PATTERNS), re.I)
_PERSONAL_RE = re.compile("|".join(_PERSONAL_PATTERNS), re.I)
_FORBIDDEN_RE = re.compile("|".join(_FORBIDDEN_OUTPUT), re.I)


def is_emergency(text: str) -> bool:
    """True when the question describes a recognised warning sign."""
    return bool(_EMERGENCY_RE.search(text or ""))


def is_personal_medical_request(text: str) -> bool:
    """True when the question asks for a verdict on the asker rather than a topic."""
    return bool(_PERSONAL_RE.search(text or ""))


def is_in_scope(text: str) -> bool:
    """True when the question touches lung health, radiography or this project."""
    low = (text or "").lower()
    return any(term in low for term in _IN_SCOPE)


def output_violations(text: str) -> list[str]:
    """Forbidden phrases found in generated text, for logging and for rejection."""
    return sorted({m.group(0).lower() for m in _FORBIDDEN_RE.finditer(text or "")})


def classify(question: str) -> str:
    """
    Route a question: 'emergency', 'personal', 'out_of_scope' or 'answerable'.

    Emergency wins over everything else, including out-of-scope, because a
    person in trouble should not be met with a scope lecture.
    """
    if is_emergency(question):
        return "emergency"
    if is_personal_medical_request(question):
        return "personal"
    if not is_in_scope(question):
        return "out_of_scope"
    return "answerable"
