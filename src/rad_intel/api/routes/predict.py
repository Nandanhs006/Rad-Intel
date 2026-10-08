"""
Prediction API route for chest radiograph classification.
"""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from rad_intel.api.dependencies import (
    ModelManager,
    get_model_manager,
    get_preprocessor,
    CXRPreprocessor,
)
from rad_intel.models.factory import ModelType
from rad_intel.api.schemas import PredictionResponse

router = APIRouter(prefix="/api/v1", tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_xray(
    file: UploadFile = File(..., description="Chest X-Ray image file (PNG, JPEG, or DICOM-derived)"),
    model_name: ModelType | None = Form(
        None, description="Model architecture to use (optional, defaults to config)"
    ),
    manager: ModelManager = Depends(get_model_manager),
    preprocessor: CXRPreprocessor = Depends(get_preprocessor),
):
    """
    Classifies an uploaded chest radiograph as NORMAL or PNEUMONIA.
    Returns predicted class, confidence, calibrated probabilities, and inference latency.
    """
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        tensor, _ = preprocessor.preprocess(image_bytes, device=manager.device)
        selected_model = model_name or manager.active_model_name

        pred_class, confidence, prob_dict, latency_ms = manager.predict(
            tensor, model_name=selected_model
        )

        return PredictionResponse(
            model_name=selected_model,
            prediction_class=pred_class,
            confidence=round(confidence, 4),
            probabilities=prob_dict,
            inference_time_ms=round(latency_ms, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
