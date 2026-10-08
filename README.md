# Rad-Intel: Clinical Decision Support System for Automated Pneumonia Detection

> **Data-Driven Clinical Decision Support Framework for Automated Pneumonia Detection from Chest Radiographs**  
> Built on **Python 3.14**, PyTorch, torchvision, timm, FastAPI, ReportLab, and Google GenAI (Gemini Flash).

---

## 1. System Architecture Overview

```
                      +-----------------------------+
                      |   Chest Radiograph (CXR)    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Preprocessing (CLAHE, 224p) |
                      +--------------+--------------+
                                     |
                   +-----------------+-----------------+
                   |                                   |
                   v                                   v
      +-------------------------+         +-------------------------+
      |  DenseNet121 Extractor  |         |  Swin-T Vision Transf.  |
      |   (Local Feature Maps)  |         |   (Global Context)      |
      |    (B, 1024, 7, 7)      |         |     (B, 768, 7, 7)      |
      +------------+------------+         +------------+------------+
                   |                                   |
                   +-----------------+-----------------+
                                     | (Channel Concat: 1792)
                                     v
                      +-----------------------------+
                      |  1x1 Projection Conv (512)  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   CBAM Attention Module     |
                      | (Channel + Spatial Refine)  |
                      +--------------+--------------+
                                     |
                   +-----------------+-----------------+
                   |                                   |
                   v                                   v
      +-------------------------+         +-------------------------+
      |   Classification Head   |         |   Explainability Engine |
      |   (Pool + Dense Logits) |         | (Grad-CAM & LIME Zones) |
      +------------+------------+         +------------+------------+
                   |                                   |
                   +-----------------+-----------------+
                                     |
                                     v
                      +-----------------------------+
                      | Strict LaTeX LLM Generator  |
                      | (Gemini Flash / ACR Rules)  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Publication-Grade PDF Engine|
                      | (ReportLab / pdflatex)      |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  FastAPI Endpoints + Web UI |
                      +-----------------------------+
```

---

## 2. Strict LaTeX Prompting & Clinical PDF Reporting

Rad-Intel synthesizes formal medical reports following the American College of Radiology (ACR) and RSNA structured reporting standards. Instead of loose markdown, reports are generated through **strict prompt rules for compilable LaTeX content**, which are then compiled into **publication-grade clinical PDF files**.

### Strict Prompt Rules (`prompts/latex_clinical_report_prompt.txt` / `src/rad_intel/reporting/latex_prompts.py`)
- **Strict LaTeX Syntax**: Rigid character escaping (`\%`, `\&`, `\_`, `\#`, `\$`), standard packages (`geometry`, `tcolorbox`, `tabularx`, `booktabs`), zero extraneous conversational commentary.
- **ACR Clinical Hierarchy**:
  1. Institution & Study Header (Rad-Intel CDS, Accession ID, Timestamp)
  2. Patient Demographics & Modality protocol
  3. AI Diagnostic Verdict & Confidence %
  4. Explainable AI (Grad-CAM) 4-Quadrant Saliency Distribution (RUZ, RLZ, LUZ, LLZ)
  5. `\section*{EXAMINATION}`
  6. `\section*{CLINICAL INDICATION}`
  7. `\section*{TECHNIQUE}`
  8. `\section*{FINDINGS}` (Airspaces, Pleura, Cardiomediastinum, Osseous structures)
  9. `\section*{IMPRESSION}` (Definitive clinical conclusions)
  10. `\section*{RECOMMENDATIONS}` (Actionable follow-up & cultures)
  11. Regulatory Medicolegal Verification Notice (`tcolorbox`)
  12. Attending Radiologist Attestation block
- **Anti-Hallucination Guardrails**:
  - Grounds every statement exclusively on provided quantitative predictions and Grad-CAM lung zone metrics.
  - Strictly prohibits fabricating prior imaging, unmentioned patient history, or non-existent laboratory values.
  - Normal radiographs strictly report clear lung parenchyma without consolidation or effusion.
  - Pneumonia findings accurately pinpoint consolidation to the detected dominant zone(s).

### Dual-Engine PDF Compiler (`src/rad_intel/reporting/pdf_generator.py`)
- **Native TeX Compiler**: Invokes system `pdflatex` or `xelatex` if present on the host environment.
- **ReportLab Medical Document Engine**: Self-contained, zero-dependency medical PDF renderer delivering clean hospital branding, tabular demographic layout, color-coded diagnostic badges (Emerald for Normal, Crimson for Pneumonia), Grad-CAM heat-intensity tables, itemized clinical findings, and physician signature fields.

---

## 3. Testing & Verification

You can verify each stage individually from the project root using the provided virtual environment:

### Stage 1: Environment, Dependencies & Configuration
```powershell
.\.venv\Scripts\python.exe scripts/test_stage1_env.py
```

### Stage 2: Deep Learning Model Architecture
```powershell
.\.venv\Scripts\python.exe scripts/test_stage2_model.py
```

### Stage 3: Medical Preprocessing & Explainability (XAI)
```powershell
.\.venv\Scripts\python.exe scripts/test_stage3_xai.py
```

### Stage 4: LLM Clinical LaTeX & PDF Report Generation
```powershell
.\.venv\Scripts\python.exe scripts/test_stage4_report.py
```

### Stage 5: FastAPI End-to-End API Integration
```powershell
.\.venv\Scripts\python.exe scripts/test_stage5_api.py
```

### Running the Full Automated Pytest Suite (20 Tests)
```powershell
.\.venv\Scripts\pytest.exe -v
```

---

## 4. Running the Web Application & REST API

To launch the FastAPI development server:
```powershell
.\.venv\Scripts\uvicorn.exe src.rad_intel.api.main:app --reload --host 127.0.0.1 --port 8000
```

Once running:
- **Interactive Web Interface**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive OpenAPI Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 5. REST API Endpoint Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` or `/api/v1/health` | Service health, PyTorch version, active hardware device |
| `GET` | `/api/v1/models` | List supported models, ablation variants, and parameter counts |
| `POST` | `/api/v1/predict` | Upload CXR image -> returns `NORMAL` or `PNEUMONIA` + confidence |
| `POST` | `/api/v1/explain` | Upload CXR image -> returns Grad-CAM / LIME overlay + quadrant metrics |
| `POST` | `/api/v1/report` | Translates model metrics + XAI into structured LaTeX report & base64 PDF |
| `POST` | `/api/v1/report/pdf` | Streams official binary clinical PDF report (`application/pdf`) |
| `POST` | `/api/v1/analyze` | Flagship all-in-one pipeline (Upload -> Predict -> Explain -> LaTeX -> PDF -> Persist) |
| `GET` | `/api/v1/analyze/{id}/pdf` | Downloads the generated clinical PDF report for a completed analysis |
| `GET` | `/api/v1/history` | Paginated query history of patient analyses from SQLite database |
| `GET` | `/api/v1/history/{id}` | Detailed inspection of a specific past diagnostic scan |

---

## 6. Environment Variables (`.env`)

```ini
# Gemini API Key (optional for offline testing; required for live Google GenAI reports)
GEMINI_API_KEY=

# Hardware Device: "auto", "cpu", or "cuda"
DEVICE=auto

# Default Model: "hybrid", "densenet121", "swin_t", "hybrid_no_cbam", "resnet50"
DEFAULT_MODEL=hybrid

# Port and Host
HOST=127.0.0.1
PORT=8000
```
