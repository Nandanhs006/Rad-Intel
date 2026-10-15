"""
The clinician review workspace.

A reviewer runs a case against the shared model, marks the prediction right or
wrong, optionally supplies a corrected label and a note. The case is written to
a store separate from the Kaggle dataset, with the model and preprocessing
versions that produced it. Submitting a review changes no checkpoint: review
material becomes training material only through an explicit export, and a
fine-tuning run is a deliberate act with a result somebody has to read.

Who can do what:

* Every signed-in clinician sees every case. That is the point of a shared
  model, and the consent panel says so before anyone logs in.
* A clinician may edit or delete *their own* submissions. Not anyone else's,
  and never the image of a case already recorded -- changing the pixels under
  a verdict somebody else has read would make the record a lie. Re-upload
  instead.
* An administrator may edit or delete any record, and is the only account that
  can activate a different model version.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from rad_intel.api.dependencies import (
    CXRPreprocessor,
    ModelManager,
    get_model_manager,
    get_preprocessor,
)
from rad_intel.api.ratelimit import client_address, limit
from rad_intel.api.schemas import (
    FeedbackExportResponse,
    FeedbackItem,
    FeedbackListResponse,
    FeedbackSubmitResponse,
    FeedbackUpdateRequest,
)
from rad_intel.api.training_job import HEARTBEAT_GRACE_SECONDS, TrainingJob
from rad_intel.auth import accounts, activity
from rad_intel.auth.dependencies import require_admin, require_clinician
from rad_intel.config import settings
from rad_intel.storage.database import db_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/feedback", tags=["Clinician Feedback"])

# Separate from the Kaggle dataset directory by construction.
#
# RAD_INTEL_FEEDBACK_DIR moves the whole store somewhere else, which is what
# hosting needs: most small hosts give a container an ephemeral filesystem
# that is wiped on every redeploy, so reviewed radiographs have to live on a
# mounted volume -- a disk, a network share, or a cloud folder mounted into
# the filesystem -- rather than next to the code. The path is resolved once,
# here, so there is one answer to "where is the data" rather than several.
FEEDBACK_DIR = Path(
    os.environ.get("RAD_INTEL_FEEDBACK_DIR", "").strip()
    or settings.BASE_DIR / "feedback_store"
).expanduser().resolve()
IMAGE_DIR = FEEDBACK_DIR / "images"
EXPORT_DIR = FEEDBACK_DIR / "exports"
WEIGHTS_DIR = settings.BASE_DIR / "weights"

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

TRAIN_LOG = FEEDBACK_DIR / "training.log"
job = TrainingJob(TRAIN_LOG)


def _ensure_dirs() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def _store_path(path: Path) -> str:
    """
    How an image's location is recorded.

    Relative to the project when the store sits inside it, absolute when it
    has been moved to a mounted volume. Resolving one of these back is
    _resolve_image below, which is also where the containment check lives.
    """
    try:
        return str(path.relative_to(settings.BASE_DIR))
    except ValueError:
        return str(path)


def _resolve_image(stored: str | None) -> Path | None:
    """
    Turn a recorded path into a file inside the image store, or nothing.

    Every path is resolved and then checked against the store, so a row whose
    path was tampered with cannot reach a file elsewhere on the disk.
    """
    if not stored:
        return None
    candidate = Path(stored)
    if not candidate.is_absolute():
        candidate = settings.BASE_DIR / candidate
    candidate = candidate.resolve()
    root = IMAGE_DIR.resolve()
    if root not in candidate.parents or not candidate.is_file():
        return None
    return candidate


def _may_edit(record: dict, user: dict) -> bool:
    """The clinician who submitted it, or an administrator."""
    return user.get("role") == "admin" or (
        bool(record.get("owner_id")) and record["owner_id"] == user["id"]
    )


def _as_item(record: dict, user: dict) -> FeedbackItem:
    return FeedbackItem(**record, editable=_may_edit(record, user))


@router.get("/warning")
async def upload_warning(_user: dict = Depends(require_clinician)):
    """The notice the interface shows above the upload control."""
    return {"warning": UPLOAD_WARNING, "preprocessing_version": PREPROCESSING_VERSION}


@router.get("/participants")
async def participants(_user: dict = Depends(require_clinician)):
    """
    Who else is in the study.

    Names and institutions only. A reviewer needs to know whose cases are in
    the queue; they have no need for anybody's email address, phone number or
    Rad-Intel ID, so those are not in this response at all rather than merely
    hidden by the interface.
    """
    accounts_list = await run_in_threadpool(accounts.list_clinicians)
    everyone = [a for a in accounts_list if a["status"] == "active"]
    return {
        "count": len(everyone),
        "participants": [accounts.public_view(a) for a in everyone],
    }


@router.post("/submit", response_model=FeedbackSubmitResponse,
             dependencies=[Depends(limit("feedback-submit", 120, 3600,
                                        "review submissions"))])
async def submit_feedback(
    request: Request,
    file: UploadFile = File(..., description="Chest radiograph reviewed by the clinician"),
    verdict: str = Form(..., description="correct | incorrect"),
    corrected_label: str | None = Form(None, description="NORMAL | PNEUMONIA"),
    note: str | None = Form(None, description="Optional short review note"),
    user: dict = Depends(require_clinician),
    manager: ModelManager = Depends(get_model_manager),
    preprocessor: CXRPreprocessor = Depends(get_preprocessor),
):
    """
    Runs the standard prediction, then records the reviewer's judgement against it.

    The reviewer's name and institution come from their account, not from the
    form. Letting a submission carry its own institution would mean the queue's
    multi-site counts were self-reported per upload, and a record could be
    attributed to a site nobody at it ever saw.
    """
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
        image_path=_store_path(stored),
        institution=user["institution"],
        owner_id=user["id"],
    )
    activity.record("review.submitted", user=user, ip=client_address(request),
                    detail=f"{record_id[:8]} verdict={verdict}")

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
async def list_feedback(status: str | None = None, limit_rows: int = 100, offset: int = 0,
                        scope: str = "all", sort: str = "recent",
                        user: dict = Depends(require_clinician)):
    """
    Lists the review queue.

    `scope=mine` narrows it to the caller's own submissions, which is the only
    practical way to find the one you need to correct once the queue holds
    cases from several sites. `sort=name` orders by reviewer alphabetically;
    the default is newest first.
    """
    owner = user["id"] if scope == "mine" else None
    rows = await db_manager.list_feedback(
        status=status, limit=min(limit_rows, 500), offset=offset,
        owner_id=owner, sort=sort,
    )
    return FeedbackListResponse(
        counts=await db_manager.feedback_counts(),
        institutions=await db_manager.feedback_institutions(),
        items=[_as_item(r, user) for r in rows],
    )


@router.patch("/{record_id}", response_model=FeedbackItem)
async def update_feedback(record_id: str, req: FeedbackUpdateRequest, request: Request,
                          user: dict = Depends(require_clinician)):
    """Edits a queued review. Only its author or an administrator may."""
    existing = await db_manager.get_feedback(record_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Feedback record not found.")
    if not _may_edit(existing, user):
        raise HTTPException(
            status_code=403,
            detail=(f"This case was submitted by {existing['reviewer']}. You can edit "
                    f"your own submissions only — use the “My cases” filter to "
                    f"find them. Ask the study administrator if something else needs "
                    f"correcting."),
        )
    if existing["status"] == "exported" and user.get("role") != "admin":
        raise HTTPException(
            status_code=409,
            detail=("This case has already been exported for training and is frozen, so "
                    "the record still matches what was trained on. Export tag: "
                    f"{existing['exported_in']}. The administrator can still change it."),
        )
    if req.verdict is not None and req.verdict not in ("correct", "incorrect"):
        raise HTTPException(status_code=400, detail="verdict must be 'correct' or 'incorrect'.")
    if req.corrected_label is not None and req.corrected_label not in ("NORMAL", "PNEUMONIA"):
        raise HTTPException(
            status_code=400, detail="corrected_label must be 'NORMAL' or 'PNEUMONIA'."
        )
    if req.verdict == "incorrect" and not (req.corrected_label or existing["corrected_label"]):
        raise HTTPException(
            status_code=400,
            detail="A corrected label is required when the prediction is marked incorrect.",
        )

    await db_manager.update_feedback(
        record_id,
        verdict=req.verdict,
        corrected_label=req.corrected_label,
        note=req.note,
        status=req.status if user.get("role") == "admin" else None,
    )
    activity.record("review.edited", user=user, ip=client_address(request),
                    detail=f"{record_id[:8]} by={'admin' if user['role'] == 'admin' else 'owner'}")
    return _as_item(await db_manager.get_feedback(record_id), user)


@router.delete("/{record_id}")
async def delete_feedback(record_id: str, request: Request,
                          user: dict = Depends(require_clinician)):
    """
    Removes a review the caller submitted, e.g. the wrong image was uploaded.

    The stored image is deleted with it only when no other review still refers
    to the same file, since images are stored once per distinct radiograph and
    shared between records.

    Two things deliberately do not happen. An export manifest already written
    is not rewritten, and a model already fine-tuned on this case is not
    changed: both are historical records of what was actually trained on, and
    editing them afterwards would make the provenance a fiction. Deleting a
    record keeps it out of every *future* export.
    """
    existing = await db_manager.get_feedback(record_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Feedback record not found.")
    if not _may_edit(existing, user):
        raise HTTPException(
            status_code=403,
            detail=(f"This case was submitted by {existing['reviewer']}. You can delete "
                    f"your own submissions only."),
        )
    if job.running:
        raise HTTPException(
            status_code=409,
            detail=("A fine-tuning run is in progress and is reading the exported cases. "
                    "Deleting can resume when it finishes."),
        )

    await db_manager.delete_feedback(record_id)

    removed_image = False
    if existing.get("image_sha256"):
        remaining = await db_manager.count_feedback_for_image(existing["image_sha256"])
        if remaining == 0:
            path = _resolve_image(existing.get("image_path"))
            if path:
                path.unlink()
                removed_image = True

    activity.record("review.deleted", user=user, ip=client_address(request),
                    detail=f"{record_id[:8]} image_removed={removed_image}")
    was_exported = existing["status"] == "exported"
    return {
        "deleted": True,
        "image_removed": removed_image,
        "detail": (
            "Review deleted and it will not appear in any future export."
            + (" The export it was already part of, and any model trained from that "
               "export, are unchanged — those are records of what was actually "
               "trained on." if was_exported else "")
            + ("" if removed_image else
               " The image file is kept because another review still refers to it.")
        ),
    }


@router.post("/export", response_model=FeedbackExportResponse,
             dependencies=[Depends(limit("feedback-export", 20, 3600, "exports"))])
async def create_training_export(request: Request,
                                 user: dict = Depends(require_clinician)):
    """
    Writes a versioned JSONL manifest of every pending review and freezes them.

    This produces review material for a later, separate fine-tuning run. It
    does not start training and does not alter any checkpoint.
    """
    if job.running:
        held_by = job.snapshot()["holder_name"]
        raise HTTPException(
            status_code=409,
            detail=(f"{held_by} is running a fine-tuning job right now. Exporting would "
                    f"freeze records underneath it and change what is being trained on "
                    f"halfway through, so it waits until the run finishes. You can "
                    f"carry on uploading and correcting cases."),
        )

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
    activity.record("export.created", user=user, ip=client_address(request),
                    detail=f"{tag} records={updated}")
    return FeedbackExportResponse(
        export_tag=tag,
        path=_store_path(path),
        record_count=updated,
        note=(
            "Export written. No training has been started and no checkpoint was modified. "
            "Fine-tuning remains a separate, explicit, versioned step."
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
            "path": _store_path(p),
            "records": sum(1 for line in p.open(encoding="utf-8") if line.strip()),
            "bytes": p.stat().st_size,
        })
    return {"exports": items}


# --- training ---------------------------------------------------------------
#
# Fine-tuning takes minutes, so it cannot be a request/response cycle. The
# endpoint starts scripts/finetune_from_feedback.py as a separate process and
# the interface polls for progress. The script keeps every guarantee it has on
# the command line: the Kaggle test partition is not trained on, the existing
# checkpoint is not overwritten, and nothing is promoted automatically.

@router.post("/train", dependencies=[Depends(limit("feedback-train", 8, 3600,
                                                   "training runs"))])
async def start_training(request: Request, epochs: int = Form(3),
                         user: dict = Depends(require_clinician)):
    """Start a fine-tuning run over the most recent export."""
    if not 1 <= epochs <= 20:
        raise HTTPException(status_code=400, detail="Choose between 1 and 20 epochs.")

    if job.running:
        state = job.snapshot()
        raise HTTPException(
            status_code=409,
            detail=(f"{state['holder_name']} started a fine-tuning run at "
                    f"{(state['started_at'] or '')[11:16]} UTC. There is one shared "
                    f"model, so one run at a time. You can keep uploading and "
                    f"correcting cases while it finishes."),
        )

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

    if not job.claim(user, manifest, usable, epochs):
        raise HTTPException(status_code=409, detail="Another run started a moment ago.")

    job.start(
        script=settings.BASE_DIR / "scripts" / "finetune_from_feedback.py",
        base_dir=settings.BASE_DIR,
        manifest=manifest,
        base_model=SHARED_MODEL,
        epochs=epochs,
    )
    activity.record("training.started", user=user, ip=client_address(request),
                    detail=f"{manifest.name} cases={usable} epochs={epochs}")

    return {
        "started": True,
        "manifest": manifest.name,
        "cases": usable,
        "base_model": SHARED_MODEL,
        "epochs": epochs,
        "holder_name": user["full_name"],
        "note": ("Training runs on the server. The original checkpoint is untouched and "
                 "nothing is promoted automatically. Keep this page open: if the browser "
                 "that started the run stops reporting in for "
                 f"{HEARTBEAT_GRACE_SECONDS // 60} minutes the run is cancelled, because "
                 "nobody would be left to read the result."),
    }


@router.post("/train/cancel")
async def cancel_training(request: Request, user: dict = Depends(require_clinician)):
    """Stop the run. Allowed to the clinician who started it, and to an administrator."""
    state = job.snapshot()
    if not state["running"]:
        raise HTTPException(status_code=409, detail="No training run is in progress.")
    if not job.may_cancel(user):
        raise HTTPException(
            status_code=403,
            detail=(f"{state['holder_name']} started this run. Only they or the study "
                    f"administrator can cancel it."),
        )

    by = "administrator" if (user["role"] == "admin"
                             and state["holder_id"] != user["id"]) else "the clinician who started it"
    job.cancel(reason=f"cancelled by {by}", by=user["full_name"])
    activity.record("training.cancelled", user=user, ip=client_address(request),
                    detail=str(state["manifest"]))
    return {
        "cancelled": True,
        "detail": ("Run stopped. No new model version was written and the current model "
                   "is untouched. The export is still there, so it can be retried."),
    }


@router.get("/train/status")
async def training_status(user: dict = Depends(require_clinician)):
    """
    Progress of the current or most recent run, and what the caller may do.

    Polling this is also the holder's heartbeat: a run whose starter has gone
    away is cancelled rather than left holding the queue.
    """
    job.heartbeat(user["id"])
    state = job.snapshot()

    log = ""
    if TRAIN_LOG.exists():
        log = TRAIN_LOG.read_text(encoding="utf-8", errors="replace")[-8000:]

    produced = None
    for line in reversed(log.splitlines()):
        if line.startswith("Saved "):
            produced = line.split("Saved ", 1)[1].strip()
            break

    return {
        "running": state["running"],
        "started_at": state["started_at"],
        "finished_at": state["finished_at"],
        "manifest": state["manifest"],
        "cases": state["cases"],
        "epochs": state["epochs"],
        "holder_name": state["holder_name"],
        "holder_institution": state["holder_institution"],
        "is_holder": state["holder_id"] == user["id"],
        "can_cancel": job.may_cancel(user),
        "cancelled_by": state["cancelled_by"],
        "cancel_reason": state["cancel_reason"],
        "log": log,
        "produced_checkpoint": produced,
        "promoted": False,
        "heartbeat_grace_seconds": HEARTBEAT_GRACE_SECONDS,
    }


# --- model versions ---------------------------------------------------------
#
# The question this answers is the obvious one a reviewer asks after their
# first run: where did the new model go? Fine-tuning writes a new, numbered
# checkpoint and never overwrites the one being served, so until somebody
# activates it, every prediction -- here and on the Analyse tab -- still comes
# from the version the paper reports. That is correct behaviour and it is also
# invisible, so it is reported explicitly.

_meta_cache: dict[tuple[str, float], dict] = {}


def _checkpoint_meta(path: Path) -> dict:
    """
    Metadata from a checkpoint, cached by modification time.

    A checkpoint is ~140 MB and this is called for every version on every
    poll, so reading them all on each request would make the panel cost more
    than the training does.
    """
    key = (str(path), path.stat().st_mtime)
    if key in _meta_cache:
        return _meta_cache[key]

    import torch

    summary: dict = {}
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if isinstance(blob, dict):
            summary = {
                "model_type": blob.get("model_type"),
                "threshold": blob.get("optimal_threshold"),
                "metrics_before": blob.get("metrics_before"),
                "metrics_after": blob.get("metrics_after"),
                "provenance": blob.get("provenance"),
            }
    except Exception as exc:
        logger.warning("Could not read metadata from %s: %s", path.name, exc)
        summary = {"error": "metadata unreadable"}

    # Keyed by (path, mtime), so a rewritten checkpoint is re-read. Bounded
    # rather than cleared on each miss: with several versions on disk,
    # clearing meant nothing was ever cached.
    if len(_meta_cache) > 12:
        _meta_cache.pop(next(iter(_meta_cache)))
    _meta_cache[key] = summary
    return summary


@router.get("/models")
async def model_versions(manager: ModelManager = Depends(get_model_manager),
                         _user: dict = Depends(require_clinician)):
    """
    Every version of the shared model, and which one is actually being served.

    The base checkpoint is the one whose numbers the project reports. A
    fine-tuned version is scored on the same held-out partition so the two are
    comparable, and is served only if an administrator activates it.
    """
    base = WEIGHTS_DIR / f"best_{SHARED_MODEL}_model.pt"
    live = manager.active_checkpoint(SHARED_MODEL)
    versions = []

    for path in [base, *sorted(WEIGHTS_DIR.glob(f"best_{SHARED_MODEL}_v*_model.pt"))]:
        if not path.exists():
            continue
        meta = await run_in_threadpool(_checkpoint_meta, path)
        after = meta.get("metrics_after") or {}
        provenance = meta.get("provenance") or {}
        versions.append({
            "file": path.name,
            "label": "base (as reported in the project's results)" if path == base
                     else f"fine-tuned {path.stem.split('_')[-2]}",
            "is_base": path == base,
            "is_live": live is not None and Path(live).name == path.name,
            "threshold": meta.get("threshold"),
            "accuracy": after.get("accuracy"),
            "auroc": after.get("auroc"),
            "sensitivity": after.get("sensitivity"),
            "specificity": after.get("specificity"),
            "reviewed_cases": provenance.get("reviewed_cases"),
            "created_at": provenance.get("created_at"),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
        })

    return {
        "shared_model": SHARED_MODEL,
        "live_file": Path(live).name if live else base.name,
        "versions": versions,
        "where_to_test": (
            "The Analyse X-ray tab runs the live version shown here, so that is where "
            "to try the shared model on a case without filing a review."
        ),
        "note": (
            "A fine-tuned version is scored on the held-out partition the project's "
            "results use, so the two rows are comparable. Nothing is served until an "
            "administrator activates it, and the base checkpoint is never overwritten, "
            "so activating a version is always reversible."
        ),
    }


@router.post("/models/activate")
async def activate_version(request: Request, file: str = Form(...),
                           manager: ModelManager = Depends(get_model_manager),
                           admin: dict = Depends(require_admin)):
    """
    Choose which version of the shared model the service serves. Administrator only.

    Implemented as a pointer file rather than by copying over the base
    checkpoint, so the version the project's reported results describe stays
    byte-identical on disk and "go back to the base model" cannot fail.
    """
    candidate = (WEIGHTS_DIR / Path(file).name).resolve()
    if not str(candidate).startswith(str(WEIGHTS_DIR.resolve())) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="No such checkpoint.")

    meta_is_base = candidate.name == f"best_{SHARED_MODEL}_model.pt"
    meta = await run_in_threadpool(_checkpoint_meta, candidate)
    if meta.get("model_type") not in (None, SHARED_MODEL):
        raise HTTPException(
            status_code=400,
            detail=(f"{candidate.name} holds a {meta.get('model_type')} model. Only "
                    f"{SHARED_MODEL} versions can be served as the shared model."),
        )

    manager.activate_checkpoint(SHARED_MODEL, None if meta_is_base else candidate)
    activity.record("model.activated", user=admin, ip=client_address(request),
                    detail=candidate.name)
    return {
        "activated": candidate.name,
        "detail": (f"The service now serves {candidate.name}. Every prediction, here and "
                   f"on the Analyse tab, comes from it until it is changed back."),
    }


# --- stored images ----------------------------------------------------------

@router.get("/image/{record_id}",
            dependencies=[Depends(limit("feedback-image", 400, 3600,
                                        "radiograph views"))])
async def feedback_image(record_id: str, _user: dict = Depends(require_clinician)):
    """
    Serve a reviewed radiograph to a signed-in clinician.

    Images are not served from a static directory: they are only reachable
    through this endpoint, which means a URL alone is useless without a
    session. The record id is a UUID rather than the file path, so the store
    cannot be walked by guessing names, and the resolved path is checked
    against the store before anything is read.
    """
    from fastapi.responses import FileResponse

    record = await db_manager.get_feedback(record_id)
    if not record or not record.get("image_path"):
        raise HTTPException(status_code=404, detail="No image for this review.")

    path = _resolve_image(record.get("image_path"))
    if not path:
        raise HTTPException(status_code=404, detail="Image file is missing.")

    media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(path, media_type=media, headers={
        # The image is only valid for whoever holds the session that fetched
        # it, so it must not be left in a shared cache.
        "Cache-Control": "private, max-age=300",
    })


@router.get("/storage")
async def storage_summary(_user: dict = Depends(require_clinician)):
    """Where reviewed images live and how much room they take."""
    _ensure_dirs()
    files = [f for f in IMAGE_DIR.iterdir() if f.is_file()]
    total = sum(f.stat().st_size for f in files)
    return {
        "directory": _store_path(IMAGE_DIR),
        "image_count": len(files),
        "bytes": total,
        "human": f"{total / (1024 * 1024):.1f} MB",
        "note": (
            "Images are stored on this server, one file per distinct radiograph, named "
            "by SHA-256, so the same image submitted twice costs nothing extra. They are "
            "kept because fine-tuning needs the pixels; the database holds only the hash "
            "and the path. They are served to signed-in clinicians through an "
            "authenticated endpoint and are not reachable from a plain URL. A review "
            "deleted by its author takes its image with it, unless another review still "
            "refers to the same file."
        ),
    }
