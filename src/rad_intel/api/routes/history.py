"""
History and audit log retrieval API routes.
"""

from fastapi import APIRouter, HTTPException, Query
from rad_intel.api.schemas import (
    AnalysisHistoryResponse,
    AnalysisHistoryItem,
)
from rad_intel.storage.database import db_manager

router = APIRouter(prefix="/api/v1", tags=["Audit & History"])


@router.get("/history", response_model=AnalysisHistoryResponse)
async def get_history(
    limit: int = Query(50, ge=1, le=100, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """Retrieves logged historical patient CXR analyses."""
    records = await db_manager.list_analyses(limit=limit, offset=offset)
    items = [
        AnalysisHistoryItem(
            id=r["id"],
            created_at=r["created_at"],
            model_name=r["model_name"],
            prediction_class=r["prediction_class"],
            confidence=r["confidence"],
            dominant_zone=r.get("dominant_zone"),
            is_bilateral=bool(r.get("is_bilateral", False)),
            engine=r["engine"],
            patient_id=r.get("patient_id"),
            execution_time_ms=r["execution_time_ms"],
        )
        for r in records
    ]
    return AnalysisHistoryResponse(total_records=len(items), records=items)


@router.get("/history/{record_id}")
async def get_history_detail(record_id: str):
    """Retrieves full analysis record including raw report markdown and probabilities."""
    record = await db_manager.get_analysis(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis record not found.")
    return record
