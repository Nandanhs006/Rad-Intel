"""
Clinical report generation route using Gemini Flash / ACR structured LaTeX and PDF generator.
"""

from fastapi import APIRouter, Depends, HTTPException, Response
from rad_intel.api.dependencies import (
    ClinicalReportGenerator,
    get_report_generator,
)
from rad_intel.api.schemas import ReportRequest, ReportResponse

router = APIRouter(prefix="/api/v1", tags=["Clinical Reporting"])


@router.post("/report", response_model=ReportResponse)
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


@router.post("/report/pdf")
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

        filename = f"Rad_Intel_Report_{result.get('report_id', 'CXR')}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")
