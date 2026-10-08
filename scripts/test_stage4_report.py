r"""
Stage 4 Verification Script: LLM Clinical Report Generation Engine.
Run with: .\.venv\Scripts\python.exe scripts/test_stage4_report.py
"""

from rad_intel.reporting.report_generator import default_report_generator

def test_stage4():
    print("=" * 65)
    print(" RAD-INTEL :: STAGE 4 VERIFICATION (LLM Clinical Reporting)")
    print("=" * 65)

    # 1. Test Case 1: Focal Pneumonia in Right Lower Lung
    print("\n[1] Testing Report Generation for Pneumonia Case:")
    sample_localization_pna = {
        "dominant_zone": "right_lower_zone",
        "dominant_zone_description": "Right lower lung field / retrocardiac & basal zone",
        "dominant_intensity": 0.842,
        "is_bilateral": False,
        "distribution_summary": "Focal opacity predominantly in Right lower lung field",
        "zone_scores": {
            "right_upper_zone": 0.12,
            "right_lower_zone": 0.84,
            "left_upper_zone": 0.05,
            "left_lower_zone": 0.18,
        }
    }

    patient_meta = {
        "patient_id": "PT-2026-9042",
        "age": 58,
        "sex": "Female",
        "history": "Productive cough with purulent sputum, fever (38.6 C) for 4 days.",
    }

    res_pna = default_report_generator.generate(
        prediction_class="PNEUMONIA",
        confidence=0.965,
        probabilities={"NORMAL": 0.035, "PNEUMONIA": 0.965},
        localization=sample_localization_pna,
        patient_metadata=patient_meta,
    )

    print(f"    - Engine Used:    {res_pna['engine']}")
    print(f"    - Diagnosis:      {res_pna['prediction_class']} ({res_pna['confidence']*100:.1f}%)")
    print(f"    - Dominant Zone:  {res_pna['dominant_zone']}")
    print(f"    - Parsed Sections: {list(res_pna['sections'].keys())}")

    # Check key sections
    for sec_name in ["findings", "impression", "recommendations"]:
        sec_content = res_pna['sections'].get(sec_name, "")
        assert len(sec_content) > 10, f"Section {sec_name} is empty or too short."
        print(f"      * [{sec_name.upper()}]: {sec_content[:70]}...")

    assert "Right lower lung" in res_pna['full_markdown'] or "retrocardiac" in res_pna['full_markdown']
    print("    [PASSED] Pneumonia report generated and verified.")

    # 2. Test Case 2: Normal Clear Chest Radiograph
    print("\n[2] Testing Report Generation for Normal Case:")
    res_normal = default_report_generator.generate(
        prediction_class="NORMAL",
        confidence=0.982,
        probabilities={"NORMAL": 0.982, "PNEUMONIA": 0.018},
        localization={
            "dominant_zone": "none",
            "dominant_zone_description": "Clear lung parenchyma",
            "dominant_intensity": 0.05,
            "is_bilateral": False,
            "distribution_summary": "No focal consolidation identified",
            "zone_scores": {"right_upper_zone": 0.02, "right_lower_zone": 0.04, "left_upper_zone": 0.03, "left_lower_zone": 0.02}
        },
        patient_metadata={"patient_id": "PT-2026-1180", "age": 34, "sex": "Male", "history": "Routine pre-operative clearance."},
    )

    print(f"    - Engine Used:    {res_normal['engine']}")
    print(f"    - Diagnosis:      {res_normal['prediction_class']} ({res_normal['confidence']*100:.1f}%)")
    assert "clear" in res_normal['full_markdown'].lower()
    print("    [PASSED] Normal radiograph report verified.")

    print("\n[3] Sample Generated Report Markdown (First 500 chars):")
    print("-" * 50)
    print(res_pna['full_markdown'][:500] + "\n...")
    print("-" * 50)

    print("\n" + "=" * 65)
    print(" STAGE 4 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    test_stage4()
