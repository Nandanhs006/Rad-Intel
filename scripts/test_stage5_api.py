r"""
Stage 5 Verification Script: FastAPI End-to-End API Integration & History.
Run with: .\.venv\Scripts\python.exe scripts/test_stage5_api.py
"""

import asyncio
import io
import cv2
import numpy as np
import httpx
from PIL import Image

from rad_intel.api.main import app

def generate_sample_cxr_bytes() -> bytes:
    """Generates PNG bytes of a synthetic chest X-ray image."""
    img = np.ones((256, 256), dtype=np.uint8) * 50
    # Add simulated anatomical lung fields
    cv2.ellipse(img, (90, 130), (45, 80), 0, 0, 360, 150, -1)
    cv2.ellipse(img, (166, 130), (45, 80), 0, 0, 360, 150, -1)
    # Add focal opacity in right lung
    cv2.circle(img, (95, 160), 25, 220, -1)
    img = cv2.GaussianBlur(img, (15, 15), 0)

    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

async def test_stage5():
    print("=" * 65)
    print(" RAD-INTEL :: STAGE 5 VERIFICATION (FastAPI Endpoints)")
    print("=" * 65)

    cxr_bytes = generate_sample_cxr_bytes()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Test Root & Health Check
        print("\n[1] Testing GET /api/v1/health:")
        r_health = await client.get("/api/v1/health")
        assert r_health.status_code == 200, f"Health check failed: {r_health.text}"
        health_data = r_health.json()
        print(f"    - Status:          {health_data['status']}")
        print(f"    - PyTorch Version: {health_data['torch_version']}")
        print(f"    - Active Device:   {health_data['active_device']}")
        print(f"    - Default Model:   {health_data['default_model']}")
        print("    [PASSED] Health endpoint verified.")

        # 2. Test Models Endpoint
        print("\n[2] Testing GET /api/v1/models:")
        r_models = await client.get("/api/v1/models")
        assert r_models.status_code == 200
        models_data = r_models.json()
        print(f"    - Active Model:    {models_data['active_model']}")
        print(f"    - Model Variants:  {[m['name'] for m in models_data['available_models']]}")
        assert len(models_data["available_models"]) >= 5
        print("    [PASSED] Models inspection endpoint verified.")

        # 3. Test Predict Endpoint
        print("\n[3] Testing POST /api/v1/predict:")
        files = {"file": ("test_cxr.png", cxr_bytes, "image/png")}
        r_pred = await client.post("/api/v1/predict", files=files, data={"model_name": "hybrid"})
        assert r_pred.status_code == 200, f"Prediction failed: {r_pred.text}"
        pred_data = r_pred.json()
        print(f"    - Model:           {pred_data['model_name']}")
        print(f"    - Prediction:      {pred_data['prediction_class']} ({pred_data['confidence']*100:.1f}%)")
        print(f"    - Probabilities:   {pred_data['probabilities']}")
        print(f"    - Latency:         {pred_data['inference_time_ms']} ms")
        assert pred_data["prediction_class"] in ("NORMAL", "PNEUMONIA")
        print("    [PASSED] Predict endpoint verified.")

        # 4. Test Explain Endpoint (Grad-CAM)
        print("\n[4] Testing POST /api/v1/explain:")
        files = {"file": ("test_cxr.png", cxr_bytes, "image/png")}
        r_explain = await client.post("/api/v1/explain", files=files, data={"method": "gradcam"})
        assert r_explain.status_code == 200, f"Explain failed: {r_explain.text}"
        explain_data = r_explain.json()
        print(f"    - Method:          {explain_data['method']}")
        print(f"    - Dominant Zone:   {explain_data['localization']['dominant_zone']}")
        print(f"    - Summary:         {explain_data['localization']['distribution_summary']}")
        assert explain_data["overlay_base64"].startswith("data:image/png;base64,")
        print("    [PASSED] Explain endpoint verified.")

        # 5. Test Flagship Analyze Endpoint (End-to-End)
        print("\n[5] Testing POST /api/v1/analyze (Full Diagnostic Pipeline):")
        files = {"file": ("test_cxr.png", cxr_bytes, "image/png")}
        form_data = {
            "model_name": "hybrid",
            "patient_id": "PT-STAGE5-VERIFY",
            "patient_age": "64",
            "patient_sex": "Male",
            "patient_history": "Shortness of breath, tachypnea, low-grade fever.",
        }
        r_analyze = await client.post("/api/v1/analyze", files=files, data=form_data)
        assert r_analyze.status_code == 200, f"Analyze failed: {r_analyze.text}"
        analyze_data = r_analyze.json()
        print(f"    - Analysis ID:     {analyze_data['analysis_id']}")
        print(f"    - Total Latency:   {analyze_data['total_pipeline_time_ms']} ms")
        print(f"    - Prediction:      {analyze_data['prediction']['prediction_class']}")
        print(f"    - Overlay URI:     {analyze_data['explanation']['overlay_base64'][:35]}...")
        print(f"    - Report Engine:   {analyze_data['report']['engine']}")
        print(f"    - Report Preview:  {analyze_data['report']['sections']['impression'][:80]}...")
        assert analyze_data["analysis_id"] is not None
        print("    [PASSED] Full pipeline analyze endpoint verified.")

        # 6. Test History Endpoint
        print("\n[6] Testing GET /api/v1/history:")
        r_hist = await client.get("/api/v1/history")
        assert r_hist.status_code == 200
        hist_data = r_hist.json()
        print(f"    - Total Saved Records: {hist_data['total_records']}")
        assert hist_data["total_records"] >= 1
        latest = hist_data["records"][0]
        print(f"    - Latest Record: ID={latest['id']}, Class={latest['prediction_class']}, Patient={latest['patient_id']}")
        assert latest["id"] == analyze_data["analysis_id"]
        print("    [PASSED] History audit trail verified.")

    print("\n" + "=" * 65)
    print(" STAGE 5 VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    asyncio.run(test_stage5())
