"""
Explainability (XAI) API route for Grad-CAM and LIME saliency generation.
"""

import time
from typing import Literal
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from rad_intel.api.dependencies import (
    ModelManager,
    get_model_manager,
    get_preprocessor,
    CXRPreprocessor,
)
from rad_intel.models.factory import ModelType
from rad_intel.api.schemas import ExplanationResponse, AnatomicalLocalization
from rad_intel.xai.gradcam import GradCAMExplainer
from rad_intel.xai.lime_explainer import LIMECXRExplainer

router = APIRouter(prefix="/api/v1", tags=["Explainability"])


@router.post("/explain", response_model=ExplanationResponse)
async def explain_xray(
    file: UploadFile = File(..., description="Chest X-Ray image file"),
    method: Literal["gradcam", "gradcam++", "lime"] = Form(
        "gradcam", description="Explainability algorithm to apply"
    ),
    model_name: ModelType | None = Form(None, description="Target model architecture"),
    target_category: int | None = Form(
        None, description="Category index (0=NORMAL, 1=PNEUMONIA, None=predicted class)"
    ),
    manager: ModelManager = Depends(get_model_manager),
    preprocessor: CXRPreprocessor = Depends(get_preprocessor),
):
    """
    Generates visual saliency overlays (Grad-CAM or LIME) and radiological anatomical quadrant metrics.
    """
    start_t = time.perf_counter()
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        tensor, rgb_processed = preprocessor.preprocess(image_bytes, device=manager.device)
        model = manager.get_model(model_name)

        if method in ("gradcam", "gradcam++"):
            explainer = GradCAMExplainer(model=model, method=method)
            result = explainer.explain(
                input_tensor=tensor,
                original_rgb=rgb_processed,
                target_category=target_category,
            )
            loc_data = result["localization"]
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
            overlay_b64 = result["overlay_base64"]
        else:
            # LIME
            explainer = LIMECXRExplainer(model=model, device=manager.device)
            result = explainer.explain(
                image_rgb=rgb_processed,
                target_category=target_category if target_category is not None else 1,
                num_samples=100,
                num_features=5,
            )
            localization = None
            overlay_b64 = result["overlay_base64"]

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return ExplanationResponse(
            method=method,
            target_category=target_category,
            overlay_base64=overlay_b64,
            localization=localization,
            execution_time_ms=round(elapsed_ms, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainability error: {str(e)}")
