r"""
Test uploading a real Kaggle chest X-ray to the FastAPI backend.
Run with: .\.venv\Scripts\python.exe scripts/test_sample_upload.py
"""

import asyncio
from pathlib import Path
import httpx
from rad_intel.api.main import app

async def run_upload():
    img_path = Path("sample_images/sample_pneumonia_person100_bacteria_475.jpeg")
    if not img_path.exists():
        print(f"Sample image not found at {img_path}")
        return

    print(f"[*] Processing {img_path.name} through Rad-Intel Diagnostic Pipeline...")

    with open(img_path, "rb") as f:
        file_bytes = f.read()

    files = {"file": (img_path.name, file_bytes, "image/jpeg")}
    data = {
        "model_name": "densenet121",
        "patient_id": "PT-PNA-475",
        "patient_age": "52",
        "patient_sex": "Male",
        "patient_history": "Fever, chills, productive cough with colored sputum for 5 days.",
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        r = await client.post("/api/v1/analyze", files=files, data=data, timeout=60.0)

    if r.status_code == 200:
        res = r.json()
        print("\n" + "=" * 60)
        print(" ANALYSIS RESULTS FOR REAL CHEST X-RAY")
        print("=" * 60)
        print(f"Analysis ID:            {res['analysis_id']}")
        print(f"Prediction:             {res['prediction']['prediction_class']} (Confidence: {res['prediction']['confidence']*100:.2f}%)")
        print(f"Probabilities:          {res['prediction']['probabilities']}")
        print(f"Dominant Saliency Zone: {res['explanation']['localization']['dominant_zone_description']}")
        print(f"Bilateral Opacities:    {res['explanation']['localization']['is_bilateral']}")
        print(f"Report Engine:          {res['report']['engine']}")
        print("\n--- CLINICAL REPORT IMPRESSION ---")
        print(res['report']['sections']['impression'])
        print("\n--- CLINICAL REPORT RECOMMENDATIONS ---")
        print(res['report']['sections']['recommendations'])
        print("\n" + "=" * 60)
    else:
        print(f"API Error ({r.status_code}):", r.text)

if __name__ == "__main__":
    asyncio.run(run_upload())
