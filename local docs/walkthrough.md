# Rad-Intel Backend Architecture Walkthrough

The backend architecture for **Rad-Intel** (Data-Driven Clinical Decision Support Framework for Automated Pneumonia Detection) has been built using **Python 3.14.7**. It integrates deep learning, medical image preprocessing, visual explainability (XAI), radiological report generation with **Gemini 2.5 Flash**, and a high-performance **FastAPI** asynchronous service.

Per your instructions, **no frontend has been built yet**, and the backend is organized into **5 testable stages**, each with a dedicated verification script.

---

## 1. Architectural Highlights

### Deep Learning Core (`src/rad_intel/models/`)
- **DenseNet121 Branch**: Extracts local fine-grained pathological features ($B \times 1024 \times 7 \times 7$).
- **Swin Transformer Branch**: Models global contextual lung dependencies ($B \times 768 \times 7 \times 7$).
- **Feature Fusion & Projection**: Concatenates feature representations ($B \times 1792 \times 7 \times 7$) and projects to $512$ dimensions using a $1 \times 1$ convolution.
- **CBAM Attention Refinement**: Sequentially applies **Channel Attention** (exploiting AvgPool + MaxPool shared MLP inter-channel correlations) and **Spatial Attention** ($7 \times 7$ convolution over concatenated spatial maps).
- **Ablation & Baseline Registry**: Supports testing the Proposed Hybrid (35.4M params), Hybrid without CBAM (35.3M params), Standalone DenseNet121 (6.9M params), Standalone Swin-T (27.5M params), and ResNet50 (23.5M params).

### Preprocessing & Explainable AI (`src/rad_intel/preprocessing/` & `src/rad_intel/xai/`)
- **Medical Image Pipeline**: Ingests bytes/arrays/files, applies **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to enhance lung opacities, resizes to $224 \times 224$, and standardizes via ImageNet stats.
- **Grad-CAM / Grad-CAM++**: Targets the CBAM spatial attention layer to backpropagate gradients and generate class activation maps.
- **Anatomical Quadrant Localization**: Quantifies saliency across radiological lung zones:
  - Right Upper Zone (RUZ)
  - Right Lower Zone (RLZ)
  - Left Upper Zone (LUZ)
  - Left Lower Zone (LLZ)
  - Evaluates dominant focus and bilateral vs unilateral distribution.
- **LIME Superpixel Explainer**: Perturbation-based interpreter highlighting interpretable superpixels.
- **Visualizer**: Colormaps heatmaps (JET) and encodes overlays into base64 data URLs for immediate API transmission.

### LLM Clinical Report Engine (`src/rad_intel/reporting/`)
- **ACR Standard Radiology Reporting**: Generates standardized sections:
  1. `EXAMINATION`
  2. `CLINICAL INDICATION`
  3. `TECHNIQUE`
  4. `FINDINGS`
  5. `IMPRESSION`
  6. `RECOMMENDATIONS`
- **Gemini 2.5 Flash + Offline Fallback**: Direct integration with Google GenAI SDK (`google-genai`). When `GEMINI_API_KEY` is not provided, automatically engages a deterministic ACR-compliant clinical fallback generator.
- **Anti-Hallucination Guardrails**: Restricts reported diagnostic opacities strictly to model confidence scores and Grad-CAM anatomical coordinates.

### FastAPI REST Service & Audit Persistence (`src/rad_intel/api/` & `src/rad_intel/storage/`)
- **Modular Endpoints**:
  - `GET /api/v1/health`: System status, PyTorch version, active compute device.
  - `GET /api/v1/models`: Listing of available architectures, ablation models, and parameter counts.
  - `POST /api/v1/predict`: Raw image classification (`NORMAL` vs `PNEUMONIA`) with latency tracking.
  - `POST /api/v1/explain`: Grad-CAM / LIME heatmap overlays and anatomical quadrant scores.
  - `POST /api/v1/report`: Clinical report synthesis.
  - `POST /api/v1/analyze`: Coordinated end-to-end pipeline (Upload $\to$ Predict $\to$ Explain $\to$ Report $\to$ Persist).
  - `GET /api/v1/history`: Audit log of past patient analyses stored in asynchronous SQLite.

---

## 2. Stage-by-Stage Verification Results

All 5 stages were verified using Python 3.14 inside `.venv`:

| Stage | Script | Status | Verification Summary |
|---|---|---|---|
| **Stage 1: Environment & Config** | `scripts/test_stage1_env.py` | **PASSED** | Python 3.14.7, PyTorch 2.14.0+cpu, torchvision, timm, grad-cam, lime, fastapi, pydantic, google-genai |
| **Stage 2: Model Architecture** | `scripts/test_stage2_model.py` | **PASSED** | Forward pass & parameter counts for Hybrid (35.4M), Hybrid no CBAM (35.3M), DenseNet (6.9M), Swin (27.5M), ResNet50 (23.5M) |
| **Stage 3: Preprocessing & XAI** | `scripts/test_stage3_xai.py` | **PASSED** | CLAHE enhancement, Grad-CAM heatmap $[0, 1]$, anatomical zone mapping (RUZ, RLZ, LUZ, LLZ), LIME superpixels |
| **Stage 4: LLM Clinical Reporting** | `scripts/test_stage4_report.py` | **PASSED** | ACR sections parsed, dominant zone opacities reflected, anti-hallucination compliance |
| **Stage 5: FastAPI & History** | `scripts/test_stage5_api.py` | **PASSED** | Asynchronous HTTP tests across `/health`, `/models`, `/predict`, `/explain`, `/analyze`, and SQLite audit logging |
| **Full Pytest Suite** | `pytest -v tests/` | **16/16 PASSED** | All 16 automated tests passed in 14.2 seconds |

---

## 3. How to Test Each Stage Yourself

Open your terminal in `c:\digitals\Rad-Intel` and run:

```powershell
# Stage 1: Verify Python 3.14 runtime, packages & config
.\.venv\Scripts\python.exe scripts/test_stage1_env.py

# Stage 2: Verify Deep Learning models, CBAM & tensor passes
.\.venv\Scripts\python.exe scripts/test_stage2_model.py

# Stage 3: Verify CLAHE preprocessing, Grad-CAM & LIME
.\.venv\Scripts\python.exe scripts/test_stage3_xai.py

# Stage 4: Verify ACR radiology report generation
.\.venv\Scripts\python.exe scripts/test_stage4_report.py

# Stage 5: Verify FastAPI endpoints and SQLite history
.\.venv\Scripts\python.exe scripts/test_stage5_api.py

# Run the complete test suite:
.\.venv\Scripts\pytest.exe -v tests/
```

### Launching the Live Backend API Server

```powershell
.\.venv\Scripts\uvicorn.exe src.rad_intel.api.main:app --reload --host 127.0.0.1 --port 8000
```
- Interactive Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Alternative ReDoc UI: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
