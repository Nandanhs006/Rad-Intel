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


# --- when the language service lets us down ---------------------------------

def test_an_unparseable_reply_falls_back_instead_of_leaving_gaps(monkeypatch):
    """
    The real failure this came from: the configured model answered 503, the
    next one down replied with prose the parser could not read, and the empty
    sections reached the page as the caller's default strings -- which said
    "No focal findings." underneath a PNEUMONIA verdict.

    A reply that will not parse is a failed reply, so the deterministic
    offline generator produces the report instead. It derives its text from
    the prediction and therefore cannot contradict it.
    """
    monkeypatch.setattr(
        default_report_generator.client, "generate_report",
        lambda prompt, context: ("Sorry, I cannot help with that.", "gemini-2.5-flash"))

    res = default_report_generator.generate(
        prediction_class="PNEUMONIA", confidence=0.94,
        probabilities={"NORMAL": 0.06, "PNEUMONIA": 0.94},
        localization={"dominant_zone": "right_lower_zone",
                      "dominant_zone_description": "Right lower lung field",
                      "dominant_intensity": 0.85, "is_bilateral": False,
                      "zone_scores": {}},
        generate_pdf=False,
    )

    assert res["engine"] == "rule-based-offline-fallback"
    assert res["sections"]["findings"].strip()
    assert res["sections"]["impression"].strip()
    # And the text agrees with what was predicted.
    joined = (res["sections"]["findings"] + res["sections"]["impression"]).lower()
    assert "no acute cardiopulmonary disease" not in joined


def test_a_missing_section_never_reads_as_a_normal_study():
    """
    Whatever is shown for a section with no text must not be a finding.
    "No active focal abnormalities" and "No acute cardiopulmonary disease"
    are statements about a chest, and printing either because a report failed
    to generate is a false clinical claim.
    """
    from rad_intel.reporting import report_generator as rg

    assert "unavailable" in rg._UNAVAILABLE.lower()
    for forbidden in ("no active focal", "no acute cardiopulmonary",
                      "no focal findings", "clear lung"):
        assert forbidden not in rg._UNAVAILABLE.lower()

    markdown = default_report_generator._convert_to_markdown(
        "", {}, {"prediction_class": "PNEUMONIA", "confidence_pct": 94.0,
                 "report_id": "x", "exam_date": "2026-01-01"})
    lowered = markdown.lower()
    for forbidden in ("no active focal abnormalities", "no acute cardiopulmonary disease"):
        assert forbidden not in lowered, f"a gap must not assert: {forbidden}"


def test_the_page_shows_no_clinical_claim_when_a_section_is_empty():
    """The same rule on the other side of the wire."""
    import pathlib

    app_js = pathlib.Path("src/rad_intel/static/app.js").read_text()
    body = "\n".join(line for line in app_js.splitlines()
                     if not line.strip().startswith("//"))
    for forbidden in ('"No focal findings."', '"Evaluation completed."',
                      '"Clinical correlation advised."'):
        assert forbidden not in body, f"still used as a default: {forbidden}"
