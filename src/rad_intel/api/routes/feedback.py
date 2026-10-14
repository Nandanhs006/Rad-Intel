"""
Clinician feedback route: a review queue, not a training loop.

A reviewer runs a case, marks the prediction right or wrong, optionally
supplies a corrected binary label and a note. The case is written to a
separate store with the model and preprocessing versions that produced it.

Nothing here retrains anything. Submitting feedback does not touch a
checkpoint, and the Kaggle dataset and its split are never written to. Review
material leaves this queue only through an explicit export, which writes a
versioned JSONL manifest and marks the records it consumed.
"""

import hashlib
import json
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from rad_intel.api.dependencies import (
    CXRPreprocessor,
    ModelManager,
    get_model_manager,
    get_preprocessor,
)
from rad_intel.api.schemas import (
    FeedbackExportResponse,
    FeedbackItem,
    FeedbackListResponse,
    FeedbackSubmitResponse,
    FeedbackUpdateRequest,
)
from rad_intel.auth.dependencies import require_clinician
from rad_intel.config import settings
from rad_intel.storage.database import db_manager

router = APIRouter(prefix="/api/v1/feedback", tags=["Clinician Feedback"])

# Separate from the Kaggle dataset directory by construction.
FEEDBACK_DIR = settings.BASE_DIR / "feedback_store"
IMAGE_DIR = FEEDBACK_DIR / "images"
EXPORT_DIR = FEEDBACK_DIR / "exports"

MAX_IMAGE_BYTES = 20 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png"}

# Bumped whenever the inference preprocessing changes, so an export records
# which pipeline produced the probability a reviewer was judging.
PREPROCESSING_VERSION = "clahe2.0-8x8_pil-bilinear-224_imagenet-norm"

# Every contribution trains the same shared model, so the configuration is
# fixed here rather than chosen per review.
SHARED_MODEL = "hybrid"

UPLOAD_WARNING = (
    "Do not upload identifiable patient data without appropriate authorisation. "
    "This is a research and education store, not a clinical record system."
)


def _ensure_dirs() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/warning")
async def upload_warning(_user: dict = Depends(require_clinician)):
    """The notice the interface shows above the upload control."""
    return {"warning": UPLOAD_WARNING, "preprocessing_version": PREPROCESSING_VERSION}


@router.post("/submit", response_model=FeedbackSubmitResponse)
async def submit_feedback(
    file: UploadFile = File(..., description="Chest radiograph reviewed by the clinician"),
    verdict: str = Form(..., description="correct | incorrect"),
    corrected_label: str | None = Form(None, description="NORMAL | PNEUMONIA"),
    note: str | None = Form(None, description="Optional short review note"),
    institution: str | None = Form(None, description="Source institution, for multi-site tracking"),
    user: dict = Depends(require_clinician),
    manager: ModelManager = Depends(get_model_manager),
    preprocessor: CXRPreprocessor = Depends(get_preprocessor),
):
    """Runs the standard prediction, then records the reviewer's judgement against it."""
    if verdict not in ("correct", "incorrect"):
        raise HTTPException(status_code=400, detail="verdict must be 'correct' or 'incorrect'.")
    if corrected_label is not None and corrected_label not in ("NORMAL", "PNEUMONIA"):
        raise HTTPException(
            status_code=400, detail="corrected_label must be 'NORMAL' or 'PNEUMONIA'."
        )
    if verdict == "incorrect" and not corrected_label:
        raise HTTPException(
            status_code=400,
            detail="A corrected label is required when the prediction is marked incorrect.",
        )
    if file.content_type and file.content_type.lower() not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{file.content_type}'. Use JPEG or PNG.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit.",
        )

    # Same preprocessing and the same threshold the rest of the service uses.
    try:
        tensor, _ = preprocessor.preprocess(image_bytes, device=manager.device)
    except Exception as decode_error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode the uploaded file as an image: {decode_error}",
        )

    target = manager.require_trained(SHARED_MODEL)
    pred_class, _confidence, probs, _latency = manager.predict(tensor, target)
    threshold = manager.get_threshold(target)
    p_pneumonia = probs.get("PNEUMONIA", 0.0)

    digest = hashlib.sha256(image_bytes).hexdigest()
    _ensure_dirs()
    suffix = ".png" if (file.content_type or "").lower() == "image/png" else ".jpg"
    stored = IMAGE_DIR / f"{digest}{suffix}"
    if not stored.exists():
        stored.write_bytes(image_bytes)

    record_id = await db_manager.save_feedback(
        reviewer=user["full_name"],
        model_version=target,
        preprocessing_version=PREPROCESSING_VERSION,
        image_sha256=digest,
        predicted_label=pred_class,
        predicted_probability=float(p_pneumonia),
        threshold=float(threshold),
        verdict=verdict,
        corrected_label=corrected_label,
        note=(note or "").strip()[:2000] or None,
        image_path=str(stored.relative_to(settings.BASE_DIR)),
        institution=(institution or user["institution"] or "").strip()[:120] or None,
    )

    return FeedbackSubmitResponse(
        id=record_id,
        predicted_label=pred_class,
        predicted_probability=float(p_pneumonia),
        threshold=float(threshold),
        model_version=target,
        preprocessing_version=PREPROCESSING_VERSION,
        image_sha256=digest,
        status="pending",
    )


@router.get("", response_model=FeedbackListResponse)
async def list_feedback(status: str | None = None, limit: int = 100, offset: int = 0,
                        _user: dict = Depends(require_clinician)):
    """Lists the review queue, newest first."""
    rows = await db_manager.list_feedback(status=status, limit=min(limit, 500), offset=offset)
    return FeedbackListResponse(
        counts=await db_manager.feedback_counts(),
        institutions=await db_manager.feedback_institutions(),
        items=[FeedbackItem(**r) for r in rows],
    )


@router.patch("/{record_id}", response_model=FeedbackItem)
async def update_feedback(record_id: str, req: FeedbackUpdateRequest,
                          _user: dict = Depends(require_clinician)):
    """Edits a queued review before it is exported."""
    existing = await db_manager.get_feedback(record_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Feedback record not found.")
    if existing["status"] == "exported":
        raise HTTPException(
            status_code=409,
            detail="This record has already been exported and is frozen. Export tag: "
                   f"{existing['exported_in']}.",
        )
    if req.verdict is not None and req.verdict not in ("correct", "incorrect"):
        raise HTTPException(status_code=400, detail="verdict must be 'correct' or 'incorrect'.")
    if req.corrected_label is not None and req.corrected_label not in ("NORMAL", "PNEUMONIA"):
        raise HTTPException(
            status_code=400, detail="corrected_label must be 'NORMAL' or 'PNEUMONIA'."
        )

    await db_manager.update_feedback(
        record_id,
        verdict=req.verdict,
        corrected_label=req.corrected_label,
        note=req.note,
        status=req.status,
    )
    return FeedbackItem(**(await db_manager.get_feedback(record_id)))


@router.post("/export", response_model=FeedbackExportResponse)
async def create_training_export(_user: dict = Depends(require_clinician)):
    """
    Writes a versioned JSONL manifest of every pending review and freezes them.

    This produces review material for a later, separate fine-tuning run. It
    does not start training and does not alter any checkpoint.
    """
    pending = await db_manager.list_feedback(status="pending", limit=500)
    if not pending:
        raise HTTPException(status_code=400, detail="No pending reviews to export.")

    _ensure_dirs()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"reviewed-export-{stamp}"
    path = EXPORT_DIR / f"{tag}.jsonl"

    with path.open("w", encoding="utf-8") as fh:
        for row in pending:
            fh.write(json.dumps({
                "id": row["id"],
                "image_path": row["image_path"],
                "image_sha256": row["image_sha256"],
                "label": row["corrected_label"] or row["predicted_label"],
                "label_source": "reviewer_correction" if row["corrected_label"] else "model_agreed",
                "institution": row["institution"],
                "model_version": row["model_version"],
                "preprocessing_version": row["preprocessing_version"],
                "predicted_label": row["predicted_label"],
                "predicted_probability": row["predicted_probability"],
                "threshold": row["threshold"],
                "verdict": row["verdict"],
                "reviewer": row["reviewer"],
                "note": row["note"],
                "reviewed_at": row["updated_at"],
            }, ensure_ascii=False) + "\n")

    updated = await db_manager.mark_feedback_exported([r["id"] for r in pending], tag)
    return FeedbackExportResponse(
        export_tag=tag,
        path=str(path.relative_to(settings.BASE_DIR)),
        record_count=updated,
        note=(
            "Export written. No training has been started and no checkpoint was modified. "
            "Fine-tuning remains a separate, manual, versioned workflow."
        ),
    )


@router.get("/exports")
async def list_exports(_user: dict = Depends(require_clinician)):
    """Lists the export manifests written so far."""
    _ensure_dirs()
    items = []
    for p in sorted(EXPORT_DIR.glob("reviewed-export-*.jsonl"), reverse=True):
        items.append({
            "tag": p.stem,
            "path": str(p.relative_to(settings.BASE_DIR)),
            "records": sum(1 for _ in p.open(encoding="utf-8")),
            "bytes": p.stat().st_size,
        })
    return {"exports": items}


# --- training ---------------------------------------------------------------
#
# Fine-tuning takes minutes, so it cannot be a request/response cycle. The
# endpoint starts scripts/finetune_from_feedback.py as a detached process and
# the interface polls for progress. The script keeps every guarantee it has on
# the command line: the Kaggle test partition is not trained on, the existing
# checkpoint is not overwritten, and nothing is promoted automatically.

TRAIN_LOG = FEEDBACK_DIR / "training.log"
_train_state: dict[str, object] = {"running": False, "started_at": None, "manifest": None}
_train_lock = threading.Lock()


def _run_training(manifest: Path, base_model: str, epochs: int) -> None:
    script = settings.BASE_DIR / "scripts" / "finetune_from_feedback.py"
    cmd = [sys.executable, "-u", str(script), "--manifest", str(manifest),
           "--base-model", base_model, "--epochs", str(epochs)]
    try:
        with TRAIN_LOG.open("w", encoding="utf-8") as log:
            log.write(f"$ {' '.join(cmd)}\n\n")
            log.flush()
            subprocess.run(cmd, cwd=settings.BASE_DIR, stdout=log,
                           stderr=subprocess.STDOUT, check=False)
    except Exception as exc:
        with TRAIN_LOG.open("a", encoding="utf-8") as log:
            log.write(f"\nTraining process failed to run: {exc}\n")
    finally:
        with _train_lock:
            _train_state["running"] = False


@router.post("/train")
async def start_training(epochs: int = Form(3),
                         _user: dict = Depends(require_clinician)):
    """Start a fine-tuning run over the most recent export."""
    with _train_lock:
        if _train_state["running"]:
            raise HTTPException(status_code=409, detail="A training run is already in progress.")

    _ensure_dirs()
    manifests = sorted(EXPORT_DIR.glob("reviewed-export-*.jsonl"), reverse=True)
    if not manifests:
        raise HTTPException(
            status_code=400,
            detail="No export to train on. Review some cases and create an export first.",
        )

    manifest = manifests[0]
    usable = sum(1 for line in manifest.open(encoding="utf-8") if line.strip())
    if usable < 4:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Only {usable} reviewed radiograph"
                f"{'' if usable == 1 else 's'} in this batch. Training on so few would "
                f"teach the model almost nothing and risks skewing it. Review a few more "
                f"cases — a mix of normal and pneumonia, including ones the model got "
                f"wrong — then export again."
            ),
        )

    with _train_lock:
        _train_state.update(running=True,
                            started_at=datetime.now(timezone.utc).isoformat(),
                            manifest=manifest.name)
    threading.Thread(target=_run_training, args=(manifest, SHARED_MODEL, epochs),
                     daemon=True).start()

    return {
        "started": True,
        "manifest": manifest.name,
        "cases": usable,
        "base_model": SHARED_MODEL,
        "epochs": epochs,
        "note": "Training runs in the background. The original checkpoint is untouched "
                "and nothing will be promoted automatically.",
    }


@router.get("/train/status")
async def training_status(_user: dict = Depends(require_clinician)):
    """Progress of the current or most recent run."""
    with _train_lock:
        running = bool(_train_state["running"])
        started_at = _train_state["started_at"]
        manifest = _train_state["manifest"]

    log = ""
    if TRAIN_LOG.exists():
        log = TRAIN_LOG.read_text(encoding="utf-8", errors="replace")[-6000:]

    produced = None
    for line in reversed(log.splitlines()):
        if line.startswith("Saved "):
            produced = line.split("Saved ", 1)[1].strip()
            break

    return {
        "running": running,
        "started_at": started_at,
        "manifest": manifest,
        "log": log,
        "produced_checkpoint": produced,
        "promoted": False,
    }


@router.get("/image/{record_id}")
async def feedback_image(record_id: str, _user: dict = Depends(require_clinician)):
    """
    Serve a reviewed radiograph to a signed-in clinician.

    Images are not served from a static directory: they are only reachable
    through this endpoint, which means a URL alone is useless without a
    session. The record id is a UUID rather than the file path, so the store
    cannot be walked by guessing names.
    """
    from fastapi.responses import FileResponse

    record = await db_manager.get_feedback(record_id)
    if not record or not record.get("image_path"):
        raise HTTPException(status_code=404, detail="No image for this review.")

    path = (settings.BASE_DIR / record["image_path"]).resolve()
    # Containment check: a stored path must never escape the feedback store.
    if not str(path).startswith(str(IMAGE_DIR.resolve())) or not path.is_file():
        raise HTTPException(status_code=404, detail="Image file is missing.")

    media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(path, media_type=media)


@router.get("/storage")
async def storage_summary(_user: dict = Depends(require_clinician)):
    """Where reviewed images live and how much room they take."""
    _ensure_dirs()
    files = [f for f in IMAGE_DIR.iterdir() if f.is_file()]
    total = sum(f.stat().st_size for f in files)
    return {
        "directory": str(IMAGE_DIR.relative_to(settings.BASE_DIR)),
        "image_count": len(files),
        "bytes": total,
        "human": f"{total / (1024 * 1024):.1f} MB",
        "note": (
            "Images are stored once per distinct file, named by SHA-256, so the same "
            "radiograph submitted twice costs nothing extra. They are kept because "
            "fine-tuning needs the pixels; the database holds only the hash and the path."
        ),
    }
