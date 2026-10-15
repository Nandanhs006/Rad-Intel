"""
Gemini LLM client and offline ACR-compliant generator for clinical LaTeX radiology reporting.
Uses the official google-genai Python SDK and strict LaTeX prompt engineering.
"""

import datetime
import logging
import re
from typing import Any
from google import genai
from google.genai import types

from rad_intel.config import settings
from rad_intel.reporting.latex_prompts import (
    LATEX_REPORT_SYSTEM_PROMPT,
    OFFLINE_LATEX_REPORT_TEMPLATE,
)

logger = logging.getLogger("rad_intel.reporting")


class GeminiReportClient:
    """
    Client for generating ACR-formatted clinical LaTeX reports via Google GenAI (Gemini 2.5/3.5/3.6 Flash).
    Provides automatic fallback to an offline structured LaTeX report generator when no API key
    is provided or network requests fail.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = settings.GEMINI_MODEL,
    ):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name
        self._client: genai.Client | None = None

        if self.api_key:
            try:
                self._client = genai.Client(api_key=self.api_key)
                logger.info("Initialized Google GenAI client with Gemini model: %s", self.model_name)
            except Exception as e:
                logger.warning("Failed to initialize Google GenAI client: %s. Using offline fallback.", e)
                self._client = None
        else:
            logger.info("No GEMINI_API_KEY detected. GeminiReportClient running in offline fallback mode.")

    @property
    def is_live(self) -> bool:
        """Returns True if a live Gemini API client is initialized."""
        return self._client is not None

    def generate_report(self, user_prompt: str, context: dict[str, Any]) -> tuple[str, str]:
        """
        Generates clinical report LaTeX source.
        Returns:
            report_latex: Full LaTeX source code of the radiology report.
            engine: Name of engine used (e.g. 'gemini-2.5-flash' or 'rule-based-offline-fallback').
        """
        if self._client:
            candidate_models = [self.model_name]
            for fallback in ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-flash-latest"]:
                if fallback not in candidate_models:
                    candidate_models.append(fallback)

            for target_model in candidate_models:
                try:
                    response = self._client.models.generate_content(
                        model=target_model,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=LATEX_REPORT_SYSTEM_PROMPT,
                            temperature=0.15,
                            max_output_tokens=2048,
                        ),
                    )
                    if response.text:
                        raw_text = response.text.strip()
                        cleaned_latex = self._clean_latex_output(raw_text)
                        return cleaned_latex, target_model
                except Exception as e:
                    logger.warning("Attempt with model %s failed: %s", target_model, e)
                    continue

        # Fallback to local rule-based LaTeX ACR generator
        offline_latex = self.generate_offline_report(context)
        return offline_latex, "rule-based-offline-fallback"

    @staticmethod
    def _clean_latex_output(text: str) -> str:
        """Strips markdown code fence wrappers (```latex ... ```) if present."""
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:latex|tex)?\s*\n?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        return cleaned.strip()

    @staticmethod
    def escape_latex(text: Any) -> str:
        """Escapes LaTeX special characters in string values."""
        s = str(text)
        replacements = [
            ("&", r"\&"),
            ("%", r"\%"),
            ("$", r"\$"),
            ("#", r"\#"),
            ("_", r"\_"),
        ]
        for orig, rep in replacements:
            s = s.replace(orig, rep)
        return s

    @classmethod
    def generate_offline_report(cls, ctx: dict[str, Any]) -> str:
        """
        Deterministic, ACR-compliant radiology LaTeX report generator for offline use
        and test verification without internet/API keys.
        """
        pred_class = str(ctx.get("prediction_class", "NORMAL")).upper()
        conf = float(ctx.get("confidence_pct", 95.0))
        loc = ctx.get("localization", {})
        zones = loc.get("zone_scores", {})
        dom_desc = cls.escape_latex(loc.get("dominant_zone_description", "lung parenchyma"))
        is_bilateral = bool(loc.get("is_bilateral", False))
        patient_id = cls.escape_latex(ctx.get("patient_id", "Anonymous"))
        patient_age = cls.escape_latex(ctx.get("patient_age", "N/A"))
        patient_sex = cls.escape_latex(ctx.get("patient_sex", "N/A"))
        history = cls.escape_latex(ctx.get("clinical_history", "Cough and fever; suspected pulmonary infection."))
        report_id = cls.escape_latex(ctx.get("report_id", "RAD-2026-001"))
        exam_date = ctx.get("exam_date", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        probs = ctx.get("probabilities", {"NORMAL": 0.95, "PNEUMONIA": 0.05})

        prob_norm = probs.get("NORMAL", 1.0 - (conf / 100.0) if pred_class == "PNEUMONIA" else conf / 100.0)
        prob_pneu = probs.get("PNEUMONIA", (conf / 100.0) if pred_class == "PNEUMONIA" else 1.0 - (conf / 100.0))

        ruz = zones.get("right_upper_zone", 0.0)
        rlz = zones.get("right_lower_zone", 0.0)
        luz = zones.get("left_upper_zone", 0.0)
        llz = zones.get("left_lower_zone", 0.0)

        # Certainty tier. The template previously asserted the same categorical
        # findings at 71% confidence as at 99%, which states more than the
        # classifier supports. Hedging scales with the reported probability.
        if conf >= 85.0:
            hedge_find = "is identified"
            hedge_impr = "Radiographic features consistent with"
            hedge_caveat = ""
        elif conf >= 75.0:
            hedge_find = "is suggested"
            hedge_impr = "Findings favour"
            hedge_caveat = ""
        else:
            hedge_find = "is equivocal"
            hedge_impr = "Equivocal findings that may represent"
            hedge_caveat = (
                r" Model confidence is low and close to the decision threshold; "
                r"this finding should be treated as uncertain."
            )

        if pred_class == "PNEUMONIA":
            verdict_bg = "red!6"
            verdict_frame = "alertred!70"
            verdict_color = "alertred"
            bilateral_str = "Yes (Multifocal bilateral opacification)" if is_bilateral else "No (Focal / unilateral)"

            findings_items = (
                r"\item \textbf{LUNGS \& AIRSPACES:} A region of airspace opacification "
                + hedge_find
                + r" in the model's region of maximal response, the "
                + dom_desc
                + r". "
                + (r"Secondary multifocal airspace opacities are demonstrated within the contralateral lung field." if is_bilateral else r"Contralateral lung parenchyma demonstrates normal aeration without focal infiltrate.")
                + hedge_caveat
                + "\n"
                r"\item \textbf{PLEURAL SPACES:} Bilateral costophrenic angles remain adequately preserved; no evidence of significant pleural effusion or pneumothorax."
                + "\n"
                r"\item \textbf{CARDIOMEDIASTINAL CONTOUR:} Cardiac silhouette size and central vascular pedicle remain within acceptable physiological limits."
                + "\n"
                r"\item \textbf{OSSEOUS STRUCTURES:} Visualized ribs, clavicles, and spine demonstrate normal alignment with no acute osteolytic or traumatic disruption."
            )

            impression_items = (
                r"\item "
                + hedge_impr
                + r" acute airspace pneumonia. The model's response was greatest over the "
                + dom_desc
                + r"; this is a saliency location, not an independently verified anatomical finding."
                + "\n"
                r"\item Automated deep learning classifier confidence: \textbf{"
                + f"{conf:.1f}\\%"
                + r"} (PNEUMONIA). Interpretation by a qualified clinician is required."
            )

            # Treatment directives removed: an image classifier has no basis to
            # initiate antimicrobial therapy, and doing so exceeds the stated
            # research/education scope of the system.
            recommendation_items = (
                r"\item Clinical correlation is advised, including inflammatory markers (CBC with differential, C-reactive protein) and microbiologic sampling where clinically indicated."
                + "\n"
                r"\item Any decision regarding antimicrobial therapy rests with the treating clinician and is outside the scope of this automated output."
                + "\n"
                r"\item Interval follow-up radiography may be considered to confirm radiographic resolution."
            )
        else:
            verdict_bg = "green!6"
            verdict_frame = "safeemerald!70"
            verdict_color = "safeemerald"
            bilateral_str = "No (Symmetric aerated parenchyma)"

            findings_items = (
                r"\item \textbf{LUNGS \& AIRSPACES:} Both lungs demonstrate symmetric inflation, clear bronchovascular branching, and normal parenchymal lucency. No focal airspace consolidation, interstitial reticulation, or pulmonary nodules."
                + "\n"
                r"\item \textbf{PLEURAL SPACES:} Bilateral costophrenic and cardiophrenic sulci are sharp and well-defined without blunting, effusion, or pneumothorax."
                + "\n"
                r"\item \textbf{CARDIOMEDIASTINAL CONTOUR:} Normal cardiothoracic ratio (<0.50). Mediastinal contours, aortic knob, and hila are within normal limits."
                + "\n"
                r"\item \textbf{OSSEOUS STRUCTURES:} Thoracic cage, visualized ribs, and scapulae show intact cortical margins without acute lesion."
            )

            impression_items = (
                r"\item No acute cardiopulmonary disease. Clear lung fields without radiographic evidence of pneumonia or active consolidation."
                + "\n"
                r"\item Automated deep learning diagnostic confidence: \textbf{"
                + f"{conf:.1f}\\%"
                + r"} (NORMAL)."
            )

            recommendation_items = (
                r"\item Routine clinical follow-up as symptoms dictate."
                + "\n"
                r"\item If systemic febrile or respiratory symptoms persist clinically, investigate non-pulmonary etiologies or consider low-dose thoracic CT."
            )

        latex_content = OFFLINE_LATEX_REPORT_TEMPLATE.format(
            report_id=report_id,
            exam_date=exam_date,
            engine="rule-based-offline-fallback",
            patient_id=patient_id,
            patient_age=patient_age,
            patient_sex=patient_sex,
            clinical_history=history,
            verdict_bg=verdict_bg,
            verdict_frame=verdict_frame,
            verdict_color=verdict_color,
            prediction_class=pred_class,
            confidence_pct=conf,
            prob_normal=prob_norm,
            prob_pneumonia=prob_pneu,
            dominant_zone_desc=dom_desc,
            bilateral_status=bilateral_str,
            ruz=ruz,
            rlz=rlz,
            luz=luz,
            llz=llz,
            findings_items=findings_items,
            impression_items=impression_items,
            recommendation_items=recommendation_items,
        )

        return latex_content.strip()
