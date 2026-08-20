# Tech Stack — Rad-Intel
### Data-Driven Clinical Decision Support Framework for Automated Pneumonia Detection

This document is based on the Synopsis, Objectives, Phase 1 report, and the accompanying paper. It separates what those documents actually specify from what still needs to be chosen, and gives direct recommendations, including the decision to use the Gemini API free tier for the LLM reporting module.

---

## 1. What the Documents Actually Specify

The Synopsis, Objectives, Phase 1 report, and paper describe the model architecture and functional requirements in detail, but do not name an implementation-level engineering stack. No programming language, ML framework, database, frontend, or deployment tool is mentioned anywhere in either document. What is explicitly named is the following:

| Component | Named in Documents |
|---|---|
| Local feature extraction | DenseNet |
| Global contextual learning | Swin Transformer |
| Attention refinement | CBAM (Convolutional Block Attention Module) |
| Explainability | Grad-CAM, LIME |
| Clinical report generation | An LLM-based module (provider not specified) |
| Interface | "User-friendly interface" for image upload and result display (no framework named) |

Everything below fills in the actual engineering stack needed to build this, since the documents only cover the model design, not the software layer.

---

## 2. Current Tech Stack (proposed, to match what the documents describe)

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | Standard for deep learning and the only language the rest of this stack assumes |
| Deep learning framework | PyTorch | DenseNet and Swin Transformer both have ready pretrained weights via `torchvision` and `timm`; avoids building either architecture from scratch |
| Attention module | CBAM, implemented as a small custom PyTorch module | Not available pretrained anywhere, has to be written directly, but it is short (channel + spatial attention block) |
| Explainability | `pytorch-grad-cam` for Grad-CAM, `lime` for LIME | Both are the standard, actively maintained libraries for these exact techniques |
| LLM reporting | Gemini API, free tier | As decided — see section 4 |
| Backend | FastAPI | Serves the model, handles image upload, calls the Gemini API, returns prediction + explanation + report |
| Frontend | Streamlit | Single-file app: upload, prediction, Grad-CAM heatmap, LLM report, all in one page, no separate frontend/backend split needed |
| Image processing | OpenCV, Pillow | Resizing, normalization, augmentation, as described in the report's preprocessing module |
| Data handling | NumPy, Pandas, scikit-learn | Dataset splitting, metrics, general array operations |
| Dataset | Kaggle "Chest X-Ray Images (Pneumonia)" or NIH ChestX-ray14 | Standard, freely available, matches the "structured dataset" mentioned in the Synopsis |
| Training environment | Google Colab (free T4 GPU) | Swin Transformer and DenseNet both need GPU acceleration; unlikely a student machine has one |
| Version control | Git + GitHub | Standard, free |
| Deployment | Streamlit Community Cloud or Hugging Face Spaces | Free hosting for a Streamlit or Gradio app, no server management |

---

## 3. Better or Easier Alternatives

These are direct alternatives worth considering at each layer, stated plainly rather than as vague options.

| Layer | Current Choice | Alternative | Why It May Be Better |
|---|---|---|---|
| Backbone architecture | Swin Transformer (hybrid with DenseNet) | EfficientNet or ResNet50, fine-tuned | Swin Transformer needs more data and compute than a typical student project has access to. EfficientNet/ResNet50 are lighter, faster to train, and — per the comparison table already in the report itself — competitive on this exact task |
| Attention module | CBAM | SE (Squeeze-and-Excitation) block | Simpler to implement, fewer parameters, similar performance gains, also referenced directly in the report's own literature survey |
| Explainability | Grad-CAM + LIME | Grad-CAM alone, or Grad-CAM++ | LIME is slow on images and adds implementation complexity for limited extra insight on X-rays specifically. Grad-CAM++ gives sharper heatmaps than plain Grad-CAM with no added cost. LIME can be dropped without materially weakening the explainability story |
| Backend framework | FastAPI | Flask, or skip a separate backend entirely | Flask is simpler to set up for a small project. Streamlit alone (no FastAPI) removes the backend/frontend split completely — one Python file does everything |
| Frontend | Streamlit | Gradio | Gradio is built specifically for ML demos (image in, prediction + heatmap + text out) and requires even less UI code than Streamlit |
| Frontend (if a separate one is wanted) | — | Plain HTML/CSS/JS instead of React/Next.js | React/Next.js is unnecessary overhead for a single-page upload-and-result interface; plain HTML/JS or Streamlit/Gradio is faster to build and easier to defend in a viva |
| Dataset storage / history | — | SQLite | If prediction history needs to be stored at all, SQLite is file-based and needs no server; avoid MySQL/PostgreSQL unless there's a real multi-user concurrency requirement |
| Training infrastructure | Google Colab | Kaggle Notebooks | Same free GPU access as Colab, with the dataset often already hosted on the same platform, saving a manual download/upload step |
| Deployment | Streamlit Community Cloud | Hugging Face Spaces | Both are free; Hugging Face Spaces has slightly more generous compute limits and better support for larger PyTorch models |

---

## 4. LLM Reporting Module — Gemini API (Free Tier)

Since the LLM-based reporting module is confirmed to run on the Gemini API free tier, note the current state of that tier as of August 2026:

- **Gemini 2.5 Flash** and **Gemini 2.5 Flash-Lite** remain free, with generous daily request quotas (roughly 1,500 requests/day) and a 1-million-token context window.
- **Pro-tier models (Gemini 2.5 Pro / 3.1 Pro)** were moved to paid-only starting April 2026 and are heavily rate-limited even for testing — avoid designing around them.
- For this project, **Gemini 2.5 Flash** is the right default: it is free, fast enough for generating a short clinical summary per prediction, and well within quota for a project-scale number of requests. **Flash-Lite** is a fallback if higher request volume is needed during testing/demoing.
- Access is via Google AI Studio (`google-genai` Python SDK), using a free API key — no billing setup required to get started.
- Keep in mind: free-tier prompts and responses may be used by Google for model improvement, unlike the paid tier. Worth a one-line disclosure if the project involves real patient data (in practice, this project will use public de-identified datasets, so this is a non-issue).

---

## 5. Final Recommended Stack

Putting it together, given the constraints of a student project and the confirmed choice of Gemini's free tier:

- **Language:** Python 3.10+
- **Model backbone:** DenseNet121 (pretrained) + CBAM; Swin Transformer as the "global context" half, or EfficientNet/ResNet50 if compute/time is tight
- **Explainability:** Grad-CAM (primary), LIME optional as a secondary view
- **LLM reporting:** Gemini 2.5 Flash via the Gemini API free tier
- **Framework:** PyTorch + torchvision/timm
- **Interface:** Streamlit (single app: upload, predict, heatmap, generated report)
- **Image processing:** OpenCV, Pillow
- **Data/metrics:** NumPy, Pandas, scikit-learn
- **Dataset:** Kaggle Chest X-Ray Pneumonia dataset
- **Training:** Google Colab or Kaggle Notebooks (free GPU)
- **Version control:** Git + GitHub
- **Deployment:** Streamlit Community Cloud or Hugging Face Spaces
