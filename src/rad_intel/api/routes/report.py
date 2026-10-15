"""
Clinical report generation route using Gemini Flash / ACR structured LaTeX and PDF generator.
"""

import re

from fastapi import APIRouter, Depends, HTTPException, Response
from rad_intel.api.dependencies import (
    ClinicalReportGenerator,
    get_report_generator,
)
from rad_intel.api.schemas import ReportRequest, ReportResponse
from rad_intel.api.ratelimit import limit


router = APIRouter(prefix="/api/v1", tags=["Clinical Reporting"])


@router.post("/report", response_model=ReportResponse,
             dependencies=[Depends(limit("public-report", 20, 300,
                                         "reports"))])
async def generate_report(
    req: ReportRequest,
    generator: ClinicalReportGenerator = Depends(get_report_generator),
):
    """
    Translates classification output and visual saliency into a formal ACR-compliant
    radiology report formatted with strict LaTeX rules and compiled to PDF.
    """
    try:
        patient_dict = req.patient_metadata.model_dump() if req.patient_metadata else {}
        result = generator.generate(
            prediction_class=req.prediction_class,
            confidence=req.confidence,
            probabilities=req.probabilities,
            localization=req.localization,
            patient_metadata=patient_dict,
            generate_pdf=req.generate_pdf,
        )

        return ReportResponse(
            engine=result["engine"],
            prediction_class=result["prediction_class"],
            confidence=result["confidence"],
            dominant_zone=result["dominant_zone"],
            sections=result["sections"],
            full_markdown=result["full_markdown"],
            latex_content=result.get("latex_content"),
            pdf_base64=result.get("pdf_base64"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation error: {str(e)}")


@router.post("/report/pdf",
             dependencies=[Depends(limit("public-report-pdf", 20, 300,
                                         "report downloads"))])
async def export_report_pdf(
    req: ReportRequest,
    generator: ClinicalReportGenerator = Depends(get_report_generator),
):
    """
    Generates and streams an official, formatted clinical radiology report PDF file.
    """
    try:
        patient_dict = req.patient_metadata.model_dump() if req.patient_metadata else {}
        result = generator.generate(
            prediction_class=req.prediction_class,
            confidence=req.confidence,
            probabilities=req.probabilities,
            localization=req.localization,
            patient_metadata=patient_dict,
            generate_pdf=True,
        )
        pdf_bytes = result.get("pdf_bytes")
        if not pdf_bytes:
            raise HTTPException(status_code=500, detail="Failed to compile PDF report.")

        # The id is generated here, not sent in -- PatientMetadata has no
        # report_id field and Pydantic drops what it does not declare. Still
        # reduced to characters that cannot close the quoted string or start
        # a new header line, so this stays safe if that schema ever gains a
        # field somebody can fill in.
        stem = re.sub(r"[^A-Za-z0-9._-]", "", str(result.get("report_id", "CXR")))[:64]
        filename = f"Rad_Intel_Report_{stem or 'CXR'}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")
