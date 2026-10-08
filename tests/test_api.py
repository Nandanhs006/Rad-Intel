"""
Unit tests for FastAPI endpoints using async httpx client.
"""

import pytest
import httpx
from rad_intel.api.main import app


@pytest.mark.asyncio
async def test_health_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/v1/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "torch_version" in data


@pytest.mark.asyncio
async def test_models_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/v1/models")
        assert res.status_code == 200
        data = res.json()
        assert len(data["available_models"]) >= 5


@pytest.mark.asyncio
async def test_predict_and_analyze_endpoint(dummy_cxr_bytes):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Predict
        files = {"file": ("test.png", dummy_cxr_bytes, "image/png")}
        res = await client.post("/api/v1/predict", files=files)
        assert res.status_code == 200
        data = res.json()
        assert "prediction_class" in data
        assert "confidence" in data

        # Analyze
        files = {"file": ("test.png", dummy_cxr_bytes, "image/png")}
        res = await client.post("/api/v1/analyze", files=files, data={"patient_id": "PT-TEST-001"})
        assert res.status_code == 200
        data = res.json()
        assert "analysis_id" in data
        assert "prediction" in data
        assert "explanation" in data
        assert "report" in data
        assert "latex_content" in data["report"]
        assert "pdf_base64" in data["report"]
        assert data["report"]["pdf_base64"] is not None

        # Test download PDF endpoint for this analysis
        analysis_id = data["analysis_id"]
        pdf_res = await client.get(f"/api/v1/analyze/{analysis_id}/pdf")
        assert pdf_res.status_code == 200
        assert pdf_res.headers["content-type"] == "application/pdf"
        assert pdf_res.content.startswith(b"%PDF-")


@pytest.mark.asyncio
async def test_report_pdf_export_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "prediction_class": "PNEUMONIA",
            "confidence": 0.95,
            "probabilities": {"NORMAL": 0.05, "PNEUMONIA": 0.95},
            "localization": {
                "dominant_zone": "right_lower_zone",
                "dominant_zone_description": "Right lower lung base",
                "dominant_intensity": 0.88,
                "is_bilateral": False,
                "zone_scores": {"right_upper_zone": 0.1, "right_lower_zone": 0.88, "left_upper_zone": 0.05, "left_lower_zone": 0.12},
            },
            "patient_metadata": {
                "patient_id": "PT-TEST-PDF",
                "age": 45,
                "sex": "F",
                "history": "Persistent productive cough.",
            },
            "generate_pdf": True,
        }

        # 1. JSON report endpoint
        res = await client.post("/api/v1/report", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "latex_content" in data
        assert "pdf_base64" in data
        assert data["pdf_base64"] is not None

        # 2. Direct binary PDF export endpoint
        res_pdf = await client.post("/api/v1/report/pdf", json=payload)
        assert res_pdf.status_code == 200
        assert res_pdf.headers["content-type"] == "application/pdf"
        assert res_pdf.content.startswith(b"%PDF-")
