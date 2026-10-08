"""
Comprehensive analysis route: End-to-End Pipeline (Upload -> Predict -> Explain -> Report -> Persist).
"""

import time
import uuid
from pathlib import Path
from typing import Literal
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Response
from rad_intel.api.dependencies import (
    ModelManager,
    get_model_manager,
    get_preprocessor,
    CXRPreprocessor,
    ClinicalReportGenerator,
    get_report_generator,
)
from rad_intel.models.factory import ModelType
from rad_intel.api.schemas import (
    AnalysisResponse,
    PredictionResponse,
    ExplanationResponse,
    ReportResponse,
    AnatomicalLocalization,
)
from rad_intel.config import BASE_DIR
from rad_intel.storage.database import db_manager
from rad_intel.xai.gradcam import GradCAMExplainer

router = APIRouter(prefix="/api/v1", tags=["Comprehensive Analysis"])

REPORTS_DIR = BASE_DIR / "uploads" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_xray(
    file: UploadFile = File(..., description="Chest X-Ray image file (PNG, JPEG)"),
    model_name: ModelType | None = Form(
        None, description="Model architecture (defaults to config default: hybrid)"
    ),
    patient_id: str | None = Form("Anonymous", description="Patient identifier"),
    patient_age: str | None = Form("N/A", description="Patient age"),
    patient_sex: str | None = Form("N/A", description="Patient sex"),
    patient_history: str | None = Form(
        "Evaluation for suspected respiratory infection.", description="Clinical indication / symptoms"
    ),
    manager: ModelManager = Depends(get_model_manager),
    preprocessor: CXRPreprocessor = Depends(get_preprocessor),
    report_gen: ClinicalReportGenerator = Depends(get_report_generator),
):
    """
    Coordinates the complete Rad-Intel clinical workflow in a single request:
      1. Preprocesses and standardizes the chest radiograph.
      2. Performs deep learning classification (DenseNet-Swin-CBAM hybrid).
      3. Generates Grad-CAM visual heatmap overlay & extracts anatomical zone metrics.
      4. Synthesizes a formal ACR-compliant radiology report via Gemini Flash in strict LaTeX.
      5. Compiles a publication-grade PDF report and persists findings to SQLite audit storage.
    """
    total_start = time.perf_counter()
    analysis_id = str(uuid.uuid4())

    # Sanitize default Swagger UI placeholder strings
    p_id = "Anonymous" if not patient_id or patient_id.strip() == "string" else patient_id.strip()
    p_age = "N/A" if not patient_age or patient_age.strip() == "string" else patient_age.strip()
    p_sex = "N/A" if not patient_sex or patient_sex.strip() == "string" else patient_sex.strip()
    p_hist = (
        "Evaluation for suspected respiratory infection."
        if not patient_history or patient_history.strip() == "string"
        else patient_history.strip()
    )

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # 1. Preprocessing
        tensor, rgb_processed = preprocessor.preprocess(image_bytes, device=manager.device)
        selected_model = manager._clean_model_name(model_name)

        # 2. Prediction
        pred_class, confidence, prob_dict, inf_ms = manager.predict(
            tensor, model_name=selected_model
        )

        # 3. Explainability (Grad-CAM)
        xai_start = time.perf_counter()
        target_cat = 1 if pred_class == "PNEUMONIA" else 0
        model = manager.get_model(selected_model)
        explainer = GradCAMExplainer(model=model, method="gradcam")
        xai_result = explainer.explain(
            input_tensor=tensor, original_rgb=rgb_processed, target_category=target_cat
        )
        xai_ms = (time.perf_counter() - xai_start) * 1000.0

        loc_data = xai_result["localization"]
        localization = AnatomicalLocalization(
            zone_scores=loc_data["zone_scores"],
            dominant_zone=loc_data["dominant_zone"],
            dominant_zone_description=loc_data["dominant_zone_description"],
            dominant_intensity=loc_data["dominant_intensity"],
            right_lung_intensity=loc_data["right_lung_intensity"],
            left_lung_intensity=loc_data["left_lung_intensity"],
            is_bilateral=loc_data["is_bilateral"],
            distribution_summary=loc_data["distribution_summary"],
        )

        # 4. Clinical Report & PDF Generation
        patient_meta = {
            "patient_id": p_id,
            "age": p_age,
            "sex": p_sex,
            "history": p_hist,
            "report_id": f"RAD-{analysis_id[:8].upper()}",
            "model_name": selected_model,
        }
        report_result = report_gen.generate(
            prediction_class=pred_class,
            confidence=confidence,
            probabilities=prob_dict,
            localization=loc_data,
            patient_metadata=patient_meta,
            generate_pdf=True,
        )

        # Cache PDF bytes on disk for direct file serving
        pdf_bytes = report_result.get("pdf_bytes")
        if pdf_bytes:
            pdf_path = REPORTS_DIR / f"{analysis_id}.pdf"
            pdf_path.write_bytes(pdf_bytes)

        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0

        # 5. Save record in SQLite Audit Database
        await db_manager.save_analysis(
            record_id=analysis_id,
            model_name=selected_model,
            prediction_class=pred_class,
            confidence=confidence,
            probabilities=prob_dict,
            dominant_zone=loc_data.get("dominant_zone_description"),
            is_bilateral=loc_data.get("is_bilateral", False),
            report_markdown=report_result["full_markdown"],
            engine=report_result["engine"],
            patient_id=p_id,
            execution_time_ms=total_elapsed_ms,
        )

        return AnalysisResponse(
            analysis_id=analysis_id,
            prediction=PredictionResponse(
                model_name=selected_model,
                prediction_class=pred_class,
                confidence=round(confidence, 4),
                probabilities=prob_dict,
                inference_time_ms=round(inf_ms, 2),
            ),
            explanation=ExplanationResponse(
                method="gradcam",
                target_category=target_cat,
                overlay_base64=xai_result["overlay_base64"],
                preprocessed_base64=xai_result.get("preprocessed_base64"),
                localization=localization,
                execution_time_ms=round(xai_ms, 2),
            ),
            report=ReportResponse(
                engine=report_result["engine"],
                prediction_class=pred_class,
                confidence=confidence,
                dominant_zone=loc_data.get("dominant_zone_description"),
                sections=report_result["sections"],
                full_markdown=report_result["full_markdown"],
                latex_content=report_result.get("latex_content"),
                pdf_base64=report_result.get("pdf_base64"),
            ),
            total_pipeline_time_ms=round(total_elapsed_ms, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis pipeline error: {str(e)}")


@router.get("/analyze/{analysis_id}/pdf")
async def download_analysis_pdf(analysis_id: str):
    """
    Downloads the compiled clinical radiology PDF report for the specified analysis ID.
    """
    pdf_path = REPORTS_DIR / f"{analysis_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF report not found for this analysis ID.")

    return Response(
        content=pdf_path.read_bytes(),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="Rad_Intel_Report_{analysis_id[:8].upper()}.pdf"'},
    )
