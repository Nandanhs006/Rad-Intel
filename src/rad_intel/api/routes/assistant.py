"""
Educational assistant route.

Deliberately text-only. This endpoint accepts no file upload, so an uploaded
radiograph cannot reach the language service through it, and it is registered
under its own tag so the interface can keep it visually separate from the
classifier's evidence-constrained summary.
"""

from fastapi import APIRouter, HTTPException

from rad_intel.assistant.chat import DISCLAIMER, answer_question
from rad_intel.api.schemas import AssistantRequest, AssistantResponse, AssistantStatus
from rad_intel.config import settings

router = APIRouter(prefix="/api/v1/assistant", tags=["Educational Assistant"])

MAX_QUESTION_CHARS = 1000


@router.get("/status", response_model=AssistantStatus)
async def assistant_status():
    """Reports whether the language service is configured; the assistant works either way."""
    return AssistantStatus(
        available=True,
        language_model_live=bool(settings.GEMINI_API_KEY),
        mode="gemini" if settings.GEMINI_API_KEY else "knowledge_base",
        disclaimer=DISCLAIMER,
    )


@router.post("/chat", response_model=AssistantResponse)
async def assistant_chat(req: AssistantRequest):
    """Answers a general question about lung health, pneumonia, radiography or this project."""
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Question is too long; limit is {MAX_QUESTION_CHARS} characters.",
        )

    # Only populated when the user ticked "ask about the displayed result".
    app_context = req.displayed_result.strip() if req.displayed_result else None
    if app_context and len(app_context) > MAX_QUESTION_CHARS:
        app_context = app_context[:MAX_QUESTION_CHARS]

    result = answer_question(question, app_context=app_context)
    return AssistantResponse(
        answer=result.answer,
        sources=result.sources,
        mode=result.mode,
        is_emergency=result.is_emergency,
        disclaimer=result.disclaimer,
    )
