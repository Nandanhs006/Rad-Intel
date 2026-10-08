"""
Clinical PDF Report Generator for Rad-Intel.
Provides dual compilation:
1. Native pdflatex / xelatex compilation when available on system PATH.
2. Built-in, high-fidelity ReportLab clinical radiology PDF renderer (zero external TeX runtime dependencies).
"""

import base64
import datetime
import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger("rad_intel.reporting")


class PDFReportGenerator:
    """
    Compiles or renders structured LaTeX radiology reports into publication-grade,
    ACR-compliant medical PDF documents.
    """

    def __init__(self, prefer_pdflatex: bool = True):
        self.prefer_pdflatex = prefer_pdflatex
        self.pdflatex_path = shutil.which("pdflatex") or shutil.which("xelatex")

    def generate_pdf_bytes(
        self,
        latex_content: str,
        parsed_sections: dict[str, str],
        context: dict[str, Any],
    ) -> bytes:
        """
        Generates the PDF report bytes. Attempts system pdflatex first if available;
        otherwise compiles through the high-fidelity ReportLab medical engine.
        """
        # Try native LaTeX compiler if present and preferred
        if self.prefer_pdflatex and self.pdflatex_path:
            try:
                pdf_bytes = self._compile_with_pdflatex(latex_content)
                if pdf_bytes:
                    logger.info("Successfully compiled PDF report using system %s", self.pdflatex_path)
                    return pdf_bytes
            except Exception as e:
                logger.warning("pdflatex compilation failed (%s). Falling back to ReportLab medical renderer.", e)

        # Built-in high-fidelity ReportLab medical PDF renderer
        return self._render_reportlab_pdf(latex_content, parsed_sections, context)

    def generate_pdf_base64(
        self,
        latex_content: str,
        parsed_sections: dict[str, str],
        context: dict[str, Any],
    ) -> str:
        """Returns the generated PDF encoded as a base64 string."""
        pdf_bytes = self.generate_pdf_bytes(latex_content, parsed_sections, context)
        return base64.b64encode(pdf_bytes).decode("utf-8")

    def save_pdf_file(
        self,
        output_path: str | Path,
        latex_content: str,
        parsed_sections: dict[str, str],
        context: dict[str, Any],
    ) -> Path:
        """Generates and writes the PDF report to disk."""
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pdf_bytes = self.generate_pdf_bytes(latex_content, parsed_sections, context)
        dest.write_bytes(pdf_bytes)
        return dest

    def _compile_with_pdflatex(self, latex_content: str) -> bytes | None:
        """Executes pdflatex in an isolated temporary directory."""
        if not self.pdflatex_path:
            return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            tex_file = Path(tmp_dir) / "report.tex"
            tex_file.write_text(latex_content, encoding="utf-8")

            cmd = [
                self.pdflatex_path,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                tmp_dir,
                str(tex_file),
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
            pdf_file = Path(tmp_dir) / "report.pdf"
            if result.returncode == 0 and pdf_file.exists():
                return pdf_file.read_bytes()
            else:
                logger.warning("pdflatex error output: %s", result.stdout.decode("utf-8", errors="ignore"))
                return None

    def _render_reportlab_pdf(
        self,
        latex_content: str,
        parsed_sections: dict[str, str],
        context: dict[str, Any],
    ) -> bytes:
        """
        Renders a publication-grade, ACR-compliant radiology report PDF using ReportLab.
        """
        buffer = io.BytesIO()

        # Document setup (0.5 inch margins for dense, professional clinical report layout)
        margin = 36  # 0.5 in points
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=margin,
            rightMargin=margin,
            topMargin=margin,
            bottomMargin=margin,
        )

        # Palette
        c_navy = colors.HexColor("#0F2850")
        c_blue = colors.HexColor("#1E3A8A")
        c_slate = colors.HexColor("#475569")
        c_dark = colors.HexColor("#1E293B")
        c_bg_light = colors.HexColor("#F8FAFC")
        c_border = colors.HexColor("#CBD5E1")

        pred_class = str(context.get("prediction_class", "NORMAL")).upper()
        conf_pct = float(context.get("confidence_pct", context.get("confidence", 0.95) * 100.0))
        is_pneumonia = pred_class == "PNEUMONIA"

        c_accent = colors.HexColor("#DC2626") if is_pneumonia else colors.HexColor("#059669")
        c_accent_bg = colors.HexColor("#FEF2F2") if is_pneumonia else colors.HexColor("#ECFDF5")
        c_accent_border = colors.HexColor("#FCA5A5") if is_pneumonia else colors.HexColor("#6EE7B7")

        # Typography / Styles
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=c_blue,
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=c_slate,
        )
        meta_label = ParagraphStyle(
            "MetaLabel",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=c_navy,
        )
        meta_val = ParagraphStyle(
            "MetaVal",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=c_dark,
        )
        sec_header = ParagraphStyle(
            "SectionHeader",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=c_blue,
            spaceBefore=6,
            spaceAfter=2,
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11.5,
            textColor=c_dark,
            spaceAfter=3,
        )
        bullet_style = ParagraphStyle(
            "Bullet",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11.5,
            textColor=c_dark,
            leftIndent=12,
            spaceAfter=2,
        )
        disclaimer_style = ParagraphStyle(
            "Disclaimer",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=7,
            leading=9,
            textColor=c_slate,
        )

        elements: list[Any] = []

        # 1. Header Banner
        report_id = str(context.get("report_id", context.get("analysis_id", "RAD-2026-001")))
        exam_date = str(context.get("exam_date", datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
        engine_name = str(context.get("engine", "Gemini 2.5 Flash / Rad-Intel CDS"))

        header_table_data = [
            [
                Paragraph("<b>RAD-INTEL CLINICAL RADIOLOGY REPORT</b>", title_style),
                Paragraph(f"<b>Report ID:</b> {report_id}<br/><b>Date:</b> {exam_date}", meta_val),
            ],
            [
                Paragraph("Automated Deep Learning Decision Support & XAI Saliency Network<br/>Department of Diagnostic & Thoracic Imaging", subtitle_style),
                Paragraph(f"<b>Engine:</b> {engine_name}<br/><b>Status:</b> Pre-Verification", meta_val),
            ],
        ]
        header_table = Table(header_table_data, colWidths=[360, 180])
        header_table.setStyle(
            TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
            ])
        )
        elements.append(header_table)
        elements.append(Spacer(1, 4))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=c_blue, spaceBefore=2, spaceAfter=6))

        # 2. Patient Demographics & Study Metadata Table
        pat_id = str(context.get("patient_id", "Anonymous"))
        pat_age = str(context.get("patient_age", context.get("age", "N/A")))
        pat_sex = str(context.get("patient_sex", context.get("sex", "N/A")))
        pat_hist = str(context.get("clinical_history", context.get("history", "Suspected acute pulmonary infection.")))

        demographics_data = [
            [
                Paragraph("<b>Patient ID:</b>", meta_label),
                Paragraph(pat_id, meta_val),
                Paragraph("<b>Modality:</b>", meta_label),
                Paragraph("Digital Chest Radiography (PA/AP)", meta_val),
            ],
            [
                Paragraph("<b>Age / Sex:</b>", meta_label),
                Paragraph(f"{pat_age} / {pat_sex}", meta_val),
                Paragraph("<b>Clinical Indication:</b>", meta_label),
                Paragraph(pat_hist, meta_val),
            ],
        ]
        demo_table = Table(demographics_data, colWidths=[75, 195, 100, 170])
        demo_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), c_bg_light),
                ("BOX", (0, 0), (-1, -1), 0.5, c_border),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        elements.append(demo_table)
        elements.append(Spacer(1, 6))

        # 3. AI Quantitative Diagnostic Assessment Card
        probs = context.get("probabilities", {"NORMAL": 0.05, "PNEUMONIA": 0.95})
        p_norm = probs.get("NORMAL", 0.0)
        p_pneu = probs.get("PNEUMONIA", 0.0)

        loc = context.get("localization", {})
        dom_zone = loc.get("dominant_zone_description", "lung parenchyma")
        is_bilat = "Yes (Multifocal bilateral opacities)" if loc.get("is_bilateral") else "No (Focal / unilateral)"

        verdict_left = f"""
        <font size="8" color="#475569"><b>PRIMARY AI DIAGNOSTIC ASSESSMENT:</b></font><br/>
        <font size="16" color="{c_accent.hexval()}"><b>{pred_class}</b></font><br/>
        <font size="9" color="#1E293B"><b>Confidence: {conf_pct:.1f}%</b></font>
        """

        verdict_right = f"""
        <b>Class Probabilities:</b> Normal: {p_norm:.3f} | Pneumonia: {p_pneu:.3f}<br/>
        <b>Dominant Localization:</b> {dom_zone}<br/>
        <b>Bilateral Status:</b> {is_bilat}
        """

        assessment_data = [
            [
                Paragraph(verdict_left, styles["Normal"]),
                Paragraph(verdict_right, body_style),
            ]
        ]
        assessment_table = Table(assessment_data, colWidths=[240, 300])
        assessment_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), c_accent_bg),
                ("BOX", (0, 0), (-1, -1), 1, c_accent_border),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ])
        )
        elements.append(assessment_table)
        elements.append(Spacer(1, 6))

        # 4. Grad-CAM Quadrant Intensity Table
        zones = loc.get("zone_scores", {})
        ruz = zones.get("right_upper_zone", 0.0)
        rlz = zones.get("right_lower_zone", 0.0)
        luz = zones.get("left_upper_zone", 0.0)
        llz = zones.get("left_lower_zone", 0.0)

        quad_data = [
            [
                Paragraph("<b>Right Upper Zone (RUZ)</b>", meta_label),
                Paragraph("<b>Right Lower Zone (RLZ)</b>", meta_label),
                Paragraph("<b>Left Upper Zone (LUZ)</b>", meta_label),
                Paragraph("<b>Left Lower Zone (LLZ)</b>", meta_label),
            ],
            [
                Paragraph(f"{ruz:.3f}", meta_val),
                Paragraph(f"{rlz:.3f}", meta_val),
                Paragraph(f"{luz:.3f}", meta_val),
                Paragraph(f"{llz:.3f}", meta_val),
            ],
        ]
        quad_table = Table(quad_data, colWidths=[135, 135, 135, 135])
        quad_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                ("BACKGROUND", (0, 1), (-1, 1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.5, c_border),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, c_border),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ])
        )
        elements.append(Paragraph("<b>Grad-CAM Explainability Saliency Distribution:</b>", meta_label))
        elements.append(Spacer(1, 2))
        elements.append(quad_table)
        elements.append(Spacer(1, 6))

        # 5. Formal ACR Sections
        exam_text = parsed_sections.get("examination", "Chest Radiograph (Single View Frontal Projection).")
        ind_text = parsed_sections.get("clinical_indication", f"Patient ID: {pat_id}; History: {pat_hist}")
        tech_text = parsed_sections.get("technique", "Standard digital frontal radiographic exposure of the thorax.")
        findings_text = parsed_sections.get("findings", "")
        impression_text = parsed_sections.get("impression", "")
        recommendations_text = parsed_sections.get("recommendations", "")

        # Examination & Technique
        elements.append(Paragraph("EXAMINATION & TECHNIQUE", sec_header))
        elements.append(Paragraph(f"<b>Exam:</b> {self._clean_text_for_pdf(exam_text)} &nbsp;|&nbsp; <b>Technique:</b> {self._clean_text_for_pdf(tech_text)}", body_style))
        elements.append(Spacer(1, 4))

        # Findings
        elements.append(Paragraph("FINDINGS", sec_header))
        findings_paragraphs = self._format_section_items(findings_text, bullet_style, body_style)
        for p in findings_paragraphs:
            elements.append(p)
        elements.append(Spacer(1, 4))

        # Impression
        elements.append(Paragraph("IMPRESSION", sec_header))
        impression_paragraphs = self._format_section_items(impression_text, bullet_style, body_style, prefix_num=True)
        for p in impression_paragraphs:
            elements.append(p)
        elements.append(Spacer(1, 4))

        # Recommendations
        elements.append(Paragraph("RECOMMENDATIONS", sec_header))
        rec_paragraphs = self._format_section_items(recommendations_text, bullet_style, body_style, prefix_num=True)
        for p in rec_paragraphs:
            elements.append(p)
        elements.append(Spacer(1, 6))

        # 6. Attestation & Regulatory Notice (KeepTogether to ensure single-page cohesion)
        attestation_elements: list[Any] = []
        attestation_elements.append(
            Table(
                [[
                    Paragraph(
                        "<b>CLINICAL DECISION SUPPORT NOTICE:</b> This report was generated by the Rad-Intel deep learning computer vision and generative informatics framework. It is intended solely to assist licensed healthcare providers. Findings must be correlated with clinical status and validated by a board-certified physician before initiating treatment.",
                        disclaimer_style,
                    )
                ]],
                colWidths=[540],
                style=[
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94A3B8")),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ],
            )
        )
        attestation_elements.append(Spacer(1, 8))

        sig_table_data = [
            [
                Paragraph("<b>Attending Radiologist Attestation:</b><br/><br/>_____________________________________<br/>MD / DO, Board Certified Radiologist", meta_val),
                Paragraph("<b>Electronic Verification:</b><br/><br/>_____________________<br/>Verified Date & Time", meta_val),
            ]
        ]
        sig_table = Table(sig_table_data, colWidths=[360, 180])
        sig_table.setStyle(
            TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ])
        )
        attestation_elements.append(sig_table)

        elements.append(KeepTogether(attestation_elements))

        # Build Document
        doc.build(elements)
        return buffer.getvalue()

    @staticmethod
    def _clean_text_for_pdf(text: str) -> str:
        """Sanitizes LaTeX and Markdown artifacts into clean HTML-like tags for ReportLab Paragraphs."""
        s = text.strip()
        # Remove LaTeX command artifacts
        s = re.sub(r"\\textbf\{([^}]+)\}", r"<b>\1</b>", s)
        s = re.sub(r"\\textit\{([^}]+)\}", r"<i>\1</i>", s)
        s = re.sub(r"\\item\s*", "", s)
        s = re.sub(r"\\section\*?\{[^}]+\}", "", s)
        s = re.sub(r"\\(?:%|&|\$|_|#|\{|\})", lambda m: m.group(0)[1:], s)
        s = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?", "", s)
        # Markdown to tags
        s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", s)
        # Escape remaining XML special chars if needed (ReportLab uses simple XML)
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        # Restore basic allowed tags
        s = s.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
        s = s.replace("&lt;i&gt;", "<i>").replace("&lt;/i&gt;", "</i>")
        s = s.replace("&lt;br/&gt;", "<br/>").replace("&lt;br&gt;", "<br/>")
        return s.strip()

    @classmethod
    def _format_section_items(
        cls,
        raw_text: str,
        bullet_style: ParagraphStyle,
        body_style: ParagraphStyle,
        prefix_num: bool = False,
    ) -> list[Paragraph]:
        """Parses multi-line/bulleted section text into clean paragraphs."""
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        paragraphs = []
        item_counter = 1

        for line in lines:
            cleaned = cls._clean_text_for_pdf(line)
            if not cleaned or cleaned in ["\\begin{itemize}", "\\end{itemize}", "\\begin{enumerate}", "\\end{enumerate}"]:
                continue

            # Strip existing leading numbers or bullets
            cleaned = re.sub(r"^(\d+[\.\)]|\-|\*|•)\s*", "", cleaned)

            if prefix_num:
                formatted = f"<b>{item_counter}.</b> {cleaned}"
                item_counter += 1
            else:
                formatted = f"• &nbsp;{cleaned}"

            paragraphs.append(Paragraph(formatted, bullet_style))

        if not paragraphs:
            paragraphs.append(Paragraph("No focal abnormalities documented.", body_style))

        return paragraphs


default_pdf_generator = PDFReportGenerator()
