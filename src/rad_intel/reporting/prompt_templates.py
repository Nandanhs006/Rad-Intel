"""
Prompt engineering and template definitions for clinical radiology report generation.
Adheres to American College of Radiology (ACR) reporting conventions with anti-hallucination guardrails.
"""

from rad_intel.reporting.latex_prompts import (
    LATEX_REPORT_SYSTEM_PROMPT,
    LATEX_REPORT_USER_PROMPT_TEMPLATE,
    OFFLINE_LATEX_REPORT_TEMPLATE,
)

# Backwards-compatible aliases
SYSTEM_PROMPT = LATEX_REPORT_SYSTEM_PROMPT
REPORT_USER_PROMPT_TEMPLATE = LATEX_REPORT_USER_PROMPT_TEMPLATE

__all__ = [
    "LATEX_REPORT_SYSTEM_PROMPT",
    "LATEX_REPORT_USER_PROMPT_TEMPLATE",
    "OFFLINE_LATEX_REPORT_TEMPLATE",
    "SYSTEM_PROMPT",
    "REPORT_USER_PROMPT_TEMPLATE",
]
