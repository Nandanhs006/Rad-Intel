"""
Pydantic v2 schemas for Rad-Intel FastAPI request and response validation.
"""

from typing import Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    project_name: str
    version: str
    torch_version: str
    active_device: str
    default_model: str
    gemini_reporting_live: bool


class ModelInfo(BaseModel):
    name: str
    description: str
    is_hybrid: bool
    is_ablation: bool
    parameters_count: int


class ModelsListResponse(BaseModel):
    active_model: str
    available_models: list[ModelInfo]


class PredictionResponse(BaseModel):
    model_name: str
    prediction_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    inference_time_ms: float


class AnatomicalLocalization(BaseModel):
    zone_scores: dict[str, float]
    dominant_zone: str
    dominant_zone_description: str
    dominant_intensity: float
    right_lung_intensity: float
    left_lung_intensity: float
    is_bilateral: bool
    distribution_summary: str


class ExplanationResponse(BaseModel):
    method: str
    target_category: int | None
    overlay_base64: str
    localization: AnatomicalLocalization | None = None
    execution_time_ms: float


class PatientMetadata(BaseModel):
    patient_id: str = "Anonymous"
    age: int | str = "N/A"
    sex: str = "N/A"
    history: str = "Evaluation for suspected respiratory infection."


class ReportRequest(BaseModel):
    prediction_class: str
    confidence: float
    probabilities: dict[str, float]
    localization: dict[str, Any] | None = None
    patient_metadata: PatientMetadata | None = None
    generate_pdf: bool = True


class ReportResponse(BaseModel):
    engine: str
    prediction_class: str
    confidence: float
    dominant_zone: str | None = None
    sections: dict[str, str]
    full_markdown: str
    latex_content: str | None = None
    pdf_base64: str | None = None


class AnalysisResponse(BaseModel):
    analysis_id: str
    prediction: PredictionResponse
    explanation: ExplanationResponse
    report: ReportResponse
    total_pipeline_time_ms: float


class AnalysisHistoryItem(BaseModel):
    id: str
    created_at: str
    model_name: str
    prediction_class: str
    confidence: float
    dominant_zone: str | None = None
    is_bilateral: bool
    engine: str
    patient_id: str | None = None
    execution_time_ms: float


class AnalysisHistoryResponse(BaseModel):
    total_records: int
    records: list[AnalysisHistoryItem]
