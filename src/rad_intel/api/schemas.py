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
    # A short label for the model chooser, e.g. "92.0% acc, AUC 0.974".
    label: str | None = None
    # Set on a version an administrator released from reviewed cases, so the
    # interface can mark it apart from the ones the project reports.
    is_released: bool = False
    released_at: str | None = None
    metrics: dict[str, float] | None = None


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
    # Share of raw saliency that fell outside the thorax before masking. A high
    # value indicates the model is responding to framing/background rather than
    # lung parenchyma; surfaced rather than silently discarded.
    off_thorax_fraction: float | None = None


class ExplanationResponse(BaseModel):
    method: str
    target_category: int | None
    overlay_base64: str
    # The 224x224 preprocessed image the overlay was drawn on. The UI must show
    # this beside the heatmap, not the raw upload: the raw file has a different
    # aspect ratio and no letterbox padding, so the two panels cannot align.
    preprocessed_base64: str | None = None
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


# --- Educational assistant -------------------------------------------------

class AssistantRequest(BaseModel):
    question: str = Field(..., description="General question about lungs, pneumonia, radiography or this project")
    displayed_result: str | None = Field(
        None,
        description=(
            "Optional short text description of the result currently on screen, sent only "
            "when the user explicitly asks about it. Never an image, never identifying data."
        ),
    )


class AssistantSource(BaseModel):
    title: str
    name: str
    url: str


class AssistantResponse(BaseModel):
    answer: str
    sources: list[AssistantSource] = Field(default_factory=list)
    mode: str = Field(..., description="gemini | knowledge_base | emergency | refusal")
    is_emergency: bool = False
    disclaimer: str


class AssistantStatus(BaseModel):
    available: bool
    language_model_live: bool
    mode: str
    disclaimer: str


# --- Clinician feedback ----------------------------------------------------

class FeedbackSubmitResponse(BaseModel):
    id: str
    predicted_label: str
    predicted_probability: float
    threshold: float
    model_version: str
    preprocessing_version: str
    image_sha256: str
    status: str


class FeedbackItem(BaseModel):
    id: str
    created_at: str
    updated_at: str
    reviewer: str
    model_version: str
    preprocessing_version: str
    image_sha256: str
    image_path: str | None = None
    institution: str | None = None
    predicted_label: str
    predicted_probability: float
    threshold: float
    verdict: str
    corrected_label: str | None = None
    note: str | None = None
    status: str
    exported_in: str | None = None
    owner_id: str | None = None
    # Whether the caller may change this record. Computed per request rather
    # than stored: the same row is editable by its author and by the
    # administrator, and by nobody else.
    editable: bool = False


class FeedbackListResponse(BaseModel):
    counts: dict[str, int]
    institutions: dict[str, int] = Field(default_factory=dict)
    items: list[FeedbackItem]


class FeedbackUpdateRequest(BaseModel):
    verdict: str | None = None
    corrected_label: str | None = None
    note: str | None = None
    status: str | None = None


class FeedbackExportResponse(BaseModel):
    export_tag: str
    path: str
    record_count: int
    note: str


# --- Authentication --------------------------------------------------------
#
# Signing in is two steps. The first names an account and gets a one-time code
# sent to the phone on its record; the second exchanges that code for a
# session. A clinician's first factor is the Rad-Intel ID the administrator
# gave them, an administrator's is a password.


class ClinicianCodeRequest(BaseModel):
    email: str
    radintel_id: str


class AdminCodeRequest(BaseModel):
    email: str
    password: str


class CodeRequestResponse(BaseModel):
    sent: bool
    channel: str
    detail: str
    expires_in_seconds: int


class VerifyCodeRequest(BaseModel):
    email: str
    code: str


class AuthMeResponse(BaseModel):
    authenticated: bool
    id: str | None = None
    email: str | None = None
    full_name: str | None = None
    institution: str | None = None
    role: str | None = None
    # Which door this session was opened by, and whether it carries the
    # console. An administrator signed in with their Rad-Intel ID holds a
    # clinician session and sees the clinician workspace only.
    session_mode: str | None = None
    is_admin_session: bool = False
    radintel_id: str | None = None
    phone: str | None = None
    must_change_password: bool = False
    # True when this sign-in closed the account's session on another device.
    displaced_other_device: bool = False
    # When a fresh one-time code will next be needed.
    session_expires_at: str | None = None
    # True when this login closed a session the account had somewhere else.
    replaced_other_device: bool = False


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class AccessRequestSubmission(BaseModel):
    full_name: str
    email: str
    institution: str
    phone: str
    registration: str | None = None
    message: str | None = None


# --- Administration --------------------------------------------------------

class AdminUserCreate(BaseModel):
    full_name: str
    email: str
    institution: str
    phone: str
    role: str = "clinician"
    admin_note: str | None = None
    radintel_id: str | None = None
    password: str | None = None


class AdminUserUpdate(BaseModel):
    full_name: str | None = None
    email: str | None = None
    institution: str | None = None
    phone: str | None = None
    role: str | None = None
    status: str | None = None
    admin_note: str | None = None


class AdminProfileUpdate(BaseModel):
    full_name: str | None = None
    email: str | None = None
    institution: str | None = None
    phone: str | None = None


class ModelReleaseRequest(BaseModel):
    """Publishing a model is confirmed with the administrator's password."""
    file: str
    password: str
    note: str | None = None


class CommunityFetchRequest(BaseModel):
    """
    Fetching a community model from a share link, instead of uploading it.

    The password is here for the same reason it is on a release: this writes
    a file that can then be put in front of the public.
    """
    url: str = Field(..., min_length=12, max_length=2000)
    password: str
