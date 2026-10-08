"""
Unit tests for clinical LaTeX reporting and PDF document generation.
"""

from rad_intel.reporting.report_generator import default_report_generator
from rad_intel.reporting.pdf_generator import default_pdf_generator
from rad_intel.reporting.latex_prompts import (
    LATEX_REPORT_SYSTEM_PROMPT,
    LATEX_REPORT_USER_PROMPT_TEMPLATE,
)


def test_latex_prompt_templates():
    """Verify strict prompt template constraints."""
    assert "ACR" in LATEX_REPORT_SYSTEM_PROMPT or "American College of Radiology" in LATEX_REPORT_SYSTEM_PROMPT
    assert r"\documentclass" in LATEX_REPORT_USER_PROMPT_TEMPLATE
    assert r"FINDINGS" in LATEX_REPORT_SYSTEM_PROMPT
    assert r"IMPRESSION" in LATEX_REPORT_SYSTEM_PROMPT
    assert r"RECOMMENDATIONS" in LATEX_REPORT_SYSTEM_PROMPT


def test_report_generation_pneumonia():
    """Test clinical report generation for pneumonia diagnosis with LaTeX and PDF compilation."""
    localization = {
        "dominant_zone": "right_lower_zone",
        "dominant_zone_description": "Right lower lung field / retrocardiac & basal zone",
        "dominant_intensity": 0.85,
        "is_bilateral": False,
        "zone_scores": {
            "right_upper_zone": 0.1,
            "right_lower_zone": 0.85,
            "left_upper_zone": 0.05,
            "left_lower_zone": 0.15,
        },
    }

    patient_meta = {
        "patient_id": "PT-90210",
        "age": "58",
        "sex": "M",
        "history": "Fever, chills, productive cough for 4 days.",
    }

    res = default_report_generator.generate(
        prediction_class="PNEUMONIA",
        confidence=0.94,
        probabilities={"NORMAL": 0.06, "PNEUMONIA": 0.94},
        localization=localization,
        patient_metadata=patient_meta,
        generate_pdf=True,
    )

    # 1. Structure assertions
    assert "latex_content" in res
    assert r"\documentclass" in res["latex_content"]
    assert r"\begin{document}" in res["latex_content"]
    assert r"\end{document}" in res["latex_content"]
    assert "PNEUMONIA" in res["latex_content"]
    assert "PT-90210" in res["latex_content"]

    # 2. Section parsing assertions
    assert "sections" in res
    assert res["prediction_class"] == "PNEUMONIA"
    assert "findings" in res["sections"]
    assert len(res["sections"]["findings"]) > 0
    assert "impression" in res["sections"]
    assert len(res["sections"]["impression"]) > 0
    assert "recommendations" in res["sections"]
    assert len(res["sections"]["recommendations"]) > 0

    # 3. PDF generation assertions
    assert "pdf_base64" in res
    assert res["pdf_base64"] is not None
    assert "pdf_bytes" in res
    assert res["pdf_bytes"] is not None
    assert res["pdf_bytes"].startswith(b"%PDF-")  # Valid PDF binary signature


def test_report_generation_normal():
    """Test clinical report generation for normal radiograph."""
    localization = {
        "dominant_zone": "none",
        "dominant_zone_description": "Normal symmetric lung fields",
        "dominant_intensity": 0.05,
        "is_bilateral": False,
        "zone_scores": {
            "right_upper_zone": 0.02,
            "right_lower_zone": 0.03,
            "left_upper_zone": 0.01,
            "left_lower_zone": 0.02,
        },
    }

    res = default_report_generator.generate(
        prediction_class="NORMAL",
        confidence=0.98,
        probabilities={"NORMAL": 0.98, "PNEUMONIA": 0.02},
        localization=localization,
        generate_pdf=True,
    )

    assert res["prediction_class"] == "NORMAL"
    assert "NORMAL" in res["latex_content"]
    assert res["pdf_bytes"].startswith(b"%PDF-")


def test_pdf_report_generator_direct():
    """Direct test of the ReportLab medical PDF renderer."""
    sample_latex = r"\documentclass{article}\begin{document}Sample Report\end{document}"
    sections = {
        "examination": "Chest Radiograph Frontal",
        "clinical_indication": "Routine checkup",
        "technique": "PA digital",
        "findings": "- Clear bilateral lungs.\n- No pleural effusion.",
        "impression": "1. Normal exam.",
        "recommendations": "1. None.",
    }
    context = {
        "prediction_class": "NORMAL",
        "confidence": 0.99,
        "probabilities": {"NORMAL": 0.99, "PNEUMONIA": 0.01},
        "patient_id": "PT-DIRECT",
        "localization": {
            "dominant_zone_description": "Clear lungs",
            "is_bilateral": False,
            "zone_scores": {"right_upper_zone": 0.0, "right_lower_zone": 0.0, "left_upper_zone": 0.0, "left_lower_zone": 0.0},
        },
    }

    pdf_bytes = default_pdf_generator.generate_pdf_bytes(sample_latex, sections, context)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 1000
