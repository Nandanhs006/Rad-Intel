"""
Report generator service coordinating classification results, XAI localization,
strict LaTeX prompt generation, LLM execution, LaTeX parsing, and PDF report compilation.
"""

import datetime
import re
from typing import Any
from rad_intel.reporting.latex_prompts import LATEX_REPORT_USER_PROMPT_TEMPLATE
from rad_intel.reporting.gemini_client import GeminiReportClient
from rad_intel.reporting.pdf_generator import PDFReportGenerator, default_pdf_generator


class ClinicalReportGenerator:
    """Orchestrates end-to-end radiology LaTeX and PDF report generation."""

    def __init__(
        self,
        client: GeminiReportClient | None = None,
        pdf_generator: PDFReportGenerator | None = None,
    ):
        self.client = client or GeminiReportClient()
        self.pdf_generator = pdf_generator or default_pdf_generator

    def generate(
        self,
        prediction_class: str,
        confidence: float,
        probabilities: dict[str, float],
        localization: dict[str, Any] | None = None,
        patient_metadata: dict[str, Any] | None = None,
        generate_pdf: bool = True,
    ) -> dict[str, Any]:
        """
        Builds the strict LaTeX prompt, executes Gemini (or deterministic offline fallback),
        parses the clinical sections, and compiles the publication-ready PDF report.
        """
        loc = localization or {}
        zones = loc.get("zone_scores", {})
        meta = patient_metadata or {}
        report_id = meta.get("report_id", f"RAD-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
        exam_date = meta.get("exam_date", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

        # Format variables according to strict prompt rules
        prompt_vars = {
            "prediction_class": prediction_class.upper(),
            "confidence_pct": confidence * 100.0,
            "model_name": meta.get("model_name", "DenseNet121-Swin-CBAM Hybrid"),
            "prob_normal": probabilities.get("NORMAL", 0.0),
            "prob_pneumonia": probabilities.get("PNEUMONIA", 0.0),
            "dominant_zone_desc": loc.get("dominant_zone_description", "lung parenchyma"),
            "dominant_intensity": loc.get("dominant_intensity", 0.0),
            "bilateral_status": "Yes (Multifocal bilateral)" if loc.get("is_bilateral") else "No (Focal / unilateral)",
            "ruz": zones.get("right_upper_zone", 0.0),
            "rlz": zones.get("right_lower_zone", 0.0),
            "luz": zones.get("left_upper_zone", 0.0),
            "llz": zones.get("left_lower_zone", 0.0),
            "saliency_summary": loc.get("distribution_summary", "Unspecified opacity pattern"),
            "report_id": report_id,
            "exam_date": exam_date,
            "patient_id": meta.get("patient_id", "Anonymous"),
            "patient_age": meta.get("age", "N/A"),
            "patient_sex": meta.get("sex", "N/A"),
            "clinical_history": meta.get("history", "Evaluation for suspected respiratory infection."),
        }

        user_prompt = LATEX_REPORT_USER_PROMPT_TEMPLATE.format(**prompt_vars)

        context = {
            "prediction_class": prediction_class.upper(),
            "confidence_pct": confidence * 100.0,
            "confidence": confidence,
            "probabilities": probabilities,
            "localization": loc,
            "report_id": report_id,
            "exam_date": exam_date,
            **meta,
        }

        report_latex, engine = self.client.generate_report(user_prompt, context)

        # Parse sections from LaTeX (and markdown fallback)
        sections = self._parse_report_sections(report_latex)

        # Build clean markdown representation for UI inspection & clipboard
        markdown_representation = self._convert_to_markdown(report_latex, sections, context)

        # Compile PDF report
        pdf_base64 = None
        pdf_bytes = None
        if generate_pdf:
            try:
                context_for_pdf = {**context, "engine": engine}
                pdf_bytes = self.pdf_generator.generate_pdf_bytes(report_latex, sections, context_for_pdf)
                pdf_base64 = self.pdf_generator.generate_pdf_base64(report_latex, sections, context_for_pdf)
            except Exception as e:
                # PDF generation error fallback
                pdf_base64 = None

        return {
            "engine": engine,
            "latex_content": report_latex,
            "full_markdown": markdown_representation,
            "sections": sections,
            "pdf_base64": pdf_base64,
            "pdf_bytes": pdf_bytes,
            "report_id": report_id,
            "prediction_class": prediction_class,
            "confidence": confidence,
            "dominant_zone": loc.get("dominant_zone_description"),
        }

    @classmethod
    def _parse_report_sections(cls, report_text: str) -> dict[str, str]:
        """Extracts individual standard sections from either LaTeX or Markdown text."""
        section_names = [
            "EXAMINATION",
            "CLINICAL INDICATION",
            "TECHNIQUE",
            "FINDINGS",
            "IMPRESSION",
            "RECOMMENDATIONS",
        ]
        parsed = {}

        for sec in section_names:
            key = sec.lower().replace(" ", "_")
            content = ""

            # 1. Try LaTeX section regex
            latex_pat = rf"\\section\*?\{{\s*{sec}\s*\}}(.*?)(?=(?:\\section\*?\{{|\n\\begin\{{tcolorbox\}}|\n\\vspace|\n\\end\{{document\}}|$))"
            match = re.search(latex_pat, report_text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
            else:
                # 2. Try Markdown section regex
                md_pat = rf"(?:###|##|\*\*)\s*{sec}[:\*]*\s*\n(.*?)(?=(?:\n(?:###|##|\*\*)[^\n]+|\n---\s*|$))"
                m_md = re.search(md_pat, report_text, re.DOTALL | re.IGNORECASE)
                if m_md:
                    content = m_md.group(1).strip()

            parsed[key] = cls._clean_section_content(content)

        return parsed

    @staticmethod
    def _clean_section_content(raw: str) -> str:
        """Strips TeX environments and converts items to standard clean text."""
        if not raw:
            return ""
        s = raw
        # Remove LaTeX itemize / enumerate containers
        s = re.sub(r"\\begin\{(?:itemize|enumerate)\}(?:\[[^\]]*\])?", "", s)
        s = re.sub(r"\\end\{(?:itemize|enumerate)\}", "", s)
        # Convert LaTeX bold/italic to Markdown
        s = re.sub(r"\\textbf\{([^}]+)\}", r"**\1**", s)
        s = re.sub(r"\\textit\{([^}]+)\}", r"*\1*", s)
        # Convert \item to clean bullet points
        s = re.sub(r"\\item\s*", "- ", s)
        # Clean escaped characters
        s = re.sub(r"\\([%&#_\$\{\}])", r"\1", s)
        # Strip trailing LaTeX spacing
        s = re.sub(r"\\(?:vspace|hspace)\{[^}]*\}", "", s)
        return s.strip()

    @staticmethod
    def _convert_to_markdown(
        report_latex: str,
        sections: dict[str, str],
        context: dict[str, Any],
    ) -> str:
        """Generates clean human-readable Markdown from the structured report."""
        pred = context.get("prediction_class", "NORMAL")
        conf = context.get("confidence_pct", 95.0)
        report_id = context.get("report_id", "RAD-2026-001")
        exam_date = context.get("exam_date", "")

        md = f"""# RAD-INTEL CLINICAL RADIOLOGY REPORT
**Study ID:** {report_id} | **Date:** {exam_date}
**Primary Diagnostic Assessment:** {pred} ({conf:.1f}% confidence)

### EXAMINATION
{sections.get('examination', 'Chest Radiograph (Single Frontal View)')}

### CLINICAL INDICATION
{sections.get('clinical_indication', context.get('clinical_history', 'Suspected respiratory infection'))}

### TECHNIQUE
{sections.get('technique', 'Standard digital frontal radiographic projection')}

### FINDINGS
{sections.get('findings', 'No active focal abnormalities')}

### IMPRESSION
{sections.get('impression', 'No acute cardiopulmonary disease')}

### RECOMMENDATIONS
{sections.get('recommendations', 'Routine clinical follow-up')}

---
*Notice: This report was generated by the Rad-Intel Clinical Decision Support System and must be validated by a board-certified physician.*
"""
        return md.strip()


default_report_generator = ClinicalReportGenerator()
