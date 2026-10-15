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
import re
import shutil
import tempfile
import zipfile
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
    CommunityFetchRequest,
    FeedbackExportResponse,
    FeedbackItem,
    FeedbackListResponse,
    FeedbackSubmitResponse,
    FeedbackUpdateRequest,
)
from rad_intel.api.training_job import HEARTBEAT_GRACE_SECONDS, TrainingJob
from rad_intel.auth import accounts, activity
from rad_intel.auth.dependencies import (
    is_admin_session,
    require_admin,
    require_clinician,
)
from rad_intel.config import settings
from rad_intel.storage import drive
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

# The second lineage: the same architecture, trained from nothing on the
# public dataset together with every reviewed case, every time. It is a
# separate set of files on purpose. best_hybrid_model.pt is the checkpoint the
# project's reported results describe and it is never written to by any of
# this; the community model starts at best_community_hybrid_v1_model.pt and
# counts up, so no run can overwrite an earlier one either.
COMMUNITY_MODEL = f"community_{SHARED_MODEL}"
COMMUNITY_PREFIX = f"best_{COMMUNITY_MODEL}_v"

# A hybrid checkpoint is ~136 MB. The ceiling is generous enough for one that
# grew a little and small enough that a stray upload cannot fill the disk.
MAX_CHECKPOINT_BYTES = 400 * 1024 * 1024

# A fine-tuned version of the shared model, i.e. best_hybrid_v3_model.pt.
# Written out rather than inferred from a filename, because this is what
# decides which files may become the default everybody gets.
_VERSION_IN_LINEAGE = re.compile(rf"best_{SHARED_MODEL}_v\d+_model\.pt")

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


async def _image_for(record: dict) -> Path | None:
    """
    The radiograph for a review, fetching it back from Drive if need be.

    The disk is a cache. A server that came up on an empty volume -- a
    redeploy, a restore, a move to another host -- has the database but not
    the pixels, and this is where they come back, once, the first time
    somebody asks to see that case.
    """
    local = _resolve_image(record.get("image_path"))
    if local:
        return local

    file_id = record.get("drive_file_id")
    if not file_id or not drive.configured():
        return None

    # Where it should have been. Recomputed from the digest rather than taken
    # from the stored path, so a row whose path was tampered with cannot make
    # this write outside the store.
    digest = record.get("image_sha256") or ""
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        return None
    suffix = Path(record.get("image_path") or "").suffix.lower()
    target = IMAGE_DIR / f"{digest}{'.png' if suffix == '.png' else '.jpg'}"

    try:
        got = await run_in_threadpool(drive.store.download, file_id, target)
    except drive.DriveError as exc:
        logger.warning("Could not fetch %s back from Drive: %s", record["id"][:8], exc)
        return None
    if not got:
        logger.warning("Drive no longer has the radiograph for %s.", record["id"][:8])
        return None
    logger.info("Restored the radiograph for %s from Drive.", record["id"][:8])
    return target if target.is_file() else None


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

    # The durable copy. A reviewed radiograph is the one thing in this study
    # that cannot be made again, and a rented disk is not where it should
    # only exist. Failing to reach Drive does not fail the review: the case
    # is saved either way and the administrator can sweep up afterwards.
    drive_file_id = None
    if drive.configured():
        try:
            drive_file_id = await run_in_threadpool(
                drive.store.upload, stored, stored.name,
                "image/png" if suffix == ".png" else "image/jpeg",
            )
        except Exception as exc:
            logger.warning("A reviewed radiograph could not be copied to Drive: %s", exc)

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
        drive_file_id=drive_file_id,
    )
    activity.record("review.submitted", user=user, ip=client_address(request),
                    detail=f"{record_id[:8]} verdict={verdict} "
                           f"drive={'yes' if drive_file_id else 'no'}")

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
            # And the durable copy, or deleting a case would leave the
            # radiograph sitting in a Drive folder for ever -- which is not
            # what anyone means by deleting it.
            file_id = existing.get("drive_file_id")
            if file_id and drive.configured():
                try:
                    await run_in_threadpool(drive.store.delete, file_id)
                    removed_image = True
                except Exception as exc:
                    logger.warning("Could not remove %s from Drive: %s", file_id, exc)

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


def _manifest_row(row: dict) -> dict:
    """One line of an export manifest."""
    return {
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
    }


async def _write_export() -> tuple[str, Path, int]:
    """
    Freeze every pending review into a new manifest. Returns (tag, path, count).

    Shared by the clinicians' export button and by building a training bundle,
    because both mean the same thing: this is the set of records as they stood
    at this moment, and they stop being editable so they keep matching it.
    A count of zero means nothing was pending and no file was written.
    """
    pending = await db_manager.list_feedback(status="pending", limit=500)
    if not pending:
        return "", EXPORT_DIR, 0

    _ensure_dirs()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"reviewed-export-{stamp}"
    path = EXPORT_DIR / f"{tag}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in pending:
            fh.write(json.dumps(_manifest_row(row), ensure_ascii=False) + "\n")

    updated = await db_manager.mark_feedback_exported([r["id"] for r in pending], tag)
    return tag, path, updated


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

    tag, path, updated = await _write_export()
    if not updated:
        raise HTTPException(status_code=400, detail="No pending reviews to export.")
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
#
# Starting and cancelling a run are the administrator's, not a clinician's.
# Training is the one thing here that costs hours of machine time and freezes
# everybody else's exports while it runs, and a clinician's job in this study
# is to review cases, not to wait on a job queue. They keep everything else:
# uploading, correcting and deleting their own cases, writing an export, and
# seeing that a run is going and who is holding it.

@router.post("/train", dependencies=[Depends(limit("feedback-train", 8, 3600,
                                                   "training runs"))])
async def start_training(request: Request, epochs: int = Form(3),
                         user: dict = Depends(require_admin)):
    """Start a fine-tuning run over the most recent export. Administrator only."""
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
                 "nothing is promoted automatically. Leave this page open: if the browser "
                 "that started the run stops reporting in for "
                 f"{HEARTBEAT_GRACE_SECONDS // 60} minutes the run is cancelled, because "
                 "nobody would be left to read the result. The public tool keeps working "
                 "throughout — it serves its own loaded copy of the model and this runs "
                 "in a separate process."),
    }


@router.post("/train/cancel")
async def cancel_training(request: Request, user: dict = Depends(require_admin)):
    """Stop the run. Administrator only, as starting one is."""
    state = job.snapshot()
    if not state["running"]:
        raise HTTPException(status_code=409, detail="No training run is in progress.")
    if not job.may_cancel(user):
        raise HTTPException(
            status_code=403,
            detail=(f"{state['holder_name']} started this run. Only they or the study "
                    f"administrator can cancel it."),
        )

    by = ("the administrator who started it" if state["holder_id"] == user["id"]
          else "the administrator")
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
        # Both halves: the account may cancel, and this session came in
        # through the administrator door. Without the second, an
        # administrator reading the queue as a clinician would be shown a
        # cancel button that the endpoint then refuses.
        "can_cancel": job.may_cancel(user) and is_admin_session(user),
        "can_train": is_admin_session(user),
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
                # A checkpoint trained from scratch has no "before" to
                # compare against, so the trainer records its scores here.
                "metrics": blob.get("metrics"),
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
        # A fine-tuned version records before/after; a checkpoint trained in
        # one go records its scores under "metrics". The base checkpoint is
        # the second kind, so without this fallback the row the clinicians are
        # being asked to compare against showed nothing but dashes.
        after = meta.get("metrics_after") or meta.get("metrics") or {}
        provenance = meta.get("provenance") or {}
        versions.append({
            "file": path.name,
            "label": "base (as reported in the project's results)" if path == base
                     else f"fine-tuned {path.stem.split('_')[-2]}",
            "is_base": path == base,
            "is_live": live is not None and Path(live).name == path.name,
            "threshold": meta.get("threshold"),
            "accuracy": after.get("accuracy"),
            "auroc": after.get("auroc", after.get("auc_roc")),
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


@router.post("/models/activate",
             dependencies=[Depends(limit("admin-activate", 20, 3600,
                                         "model changes"))])
async def activate_version(request: Request, file: str = Form(...),
                           password: str = Form(...),
                           manager: ModelManager = Depends(get_model_manager),
                           admin: dict = Depends(require_admin)):
    """
    Choose which version of the shared model the service serves. Administrator only.

    The password is asked for every time, as it is for a release. This one
    reaches further than a release does: releasing adds a version to the list
    for people who go looking, while this changes what everybody is served
    without choosing anything. The more consequential of the two should not
    be the easier one.

    Implemented as a pointer file rather than by copying over the base
    checkpoint, so the version the project's reported results describe stays
    byte-identical on disk and "go back to the base model" cannot fail.
    """
    if not await run_in_threadpool(accounts.identify_admin, admin["email"], password):
        activity.record("model.activate_refused", user=admin, ip=client_address(request),
                        detail="wrong password", outcome="refused")
        raise HTTPException(status_code=403,
                            detail="Password incorrect. Nothing changed.")

    candidate = (WEIGHTS_DIR / Path(file).name).resolve()
    if not str(candidate).startswith(str(WEIGHTS_DIR.resolve())) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="No such checkpoint.")

    # Only this lineage. A community checkpoint is a hybrid inside, so the
    # model_type check below would wave it through, and it would then become
    # the default model for everyone who never chose it. The community model
    # reaches people one way: released by name, chosen deliberately.
    if not (candidate.name == f"best_{SHARED_MODEL}_model.pt"
            or _VERSION_IN_LINEAGE.fullmatch(candidate.name)):
        raise HTTPException(
            status_code=400,
            detail=(f"{candidate.name} is not a version of the shared model. The "
                    f"community model is not served by default — it is released by "
                    f"name and people choose it on the Analyse tab."),
        )

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


# --- the community model ----------------------------------------------------
#
# The second lineage. Fine-tuning nudges the reported checkpoint with one
# batch of reviews; this one throws the previous community model away and
# trains the same architecture from ImageNet initialisation on the public
# dataset *plus every reviewed case to date*, so each run is a model built
# from all the evidence rather than a model carrying the accumulated drift of
# every run before it. Nothing about it touches best_hybrid_model.pt.
#
# It trains off this server, which is the whole reason these two endpoints
# exist. A full retrain is hours of GPU time; a host that can serve the site
# cannot do it in a sensible wall-clock time, and renting a machine that can,
# to sit idle between runs, is the wrong shape. So: the administrator
# downloads a bundle, trains it on a borrowed GPU, and uploads the checkpoint
# back. The server's job is to hand over exactly what was trained on and to
# refuse anything that comes back that it cannot verify.

BUNDLE_DIR = FEEDBACK_DIR / "bundles"
# How many built bundles to keep. Each one carries a copy of every reviewed
# image, so they are not free, and only the newest is ever trained on.
KEEP_BUNDLES = 3

BUNDLE_README = """Rad-Intel community model training bundle
=========================================

tag:   {tag}
built: {built}
cases: {cases} reviewed radiographs

What is in here
---------------
manifest.jsonl   one line per reviewed case: the label, who reviewed it, which
                 site it came from, and what the model had predicted.
images/          the radiographs themselves, named by SHA-256, exactly the
                 files the manifest points at.

What is NOT in here
-------------------
The public chest-radiograph dataset. It is 2.3 GB and anyone can download it,
so the training script fetches it itself.

What to do with it
------------------
Follow COMMUNITY_MODEL.md in the Rad-Intel repository. In short: open the
notebook it describes on a free GPU, give it this zip, let it run, and upload
the .pt file it produces back into the administrator console.

These are radiographs people uploaded to a study. Keep the file somewhere
private, and delete it from the machine you train on when you are done.
"""


def _release_name(filename: str) -> str:
    """The name this checkpoint would be offered under, e.g. community_hybrid_v1."""
    from rad_intel.models import registry

    described = registry.describe(filename)
    return described[1] if described else filename


def _community_versions() -> list[Path]:
    """Community checkpoints on disk, oldest first."""
    def version_of(path: Path) -> int:
        match = re.fullmatch(rf"{COMMUNITY_PREFIX}(\d+)_model\.pt", path.name)
        return int(match.group(1)) if match else 0

    return sorted(
        (p for p in WEIGHTS_DIR.glob(f"{COMMUNITY_PREFIX}*_model.pt") if version_of(p)),
        key=version_of,
    )


def _next_community_path() -> Path:
    """The filename the next upload takes. Never an existing one."""
    n = 1
    while (WEIGHTS_DIR / f"{COMMUNITY_PREFIX}{n}_model.pt").exists():
        n += 1
    return WEIGHTS_DIR / f"{COMMUNITY_PREFIX}{n}_model.pt"


async def _all_reviewed_rows() -> list[dict]:
    """
    Every review in the store, newest first, deduplicated by radiograph.

    Cumulative on purpose: a community run trains on all the evidence, not on
    the latest batch. Where the same radiograph was reviewed more than once --
    the same image submitted by two sites, or re-reviewed after a correction
    -- the most recent review wins, because that is the label somebody stands
    behind now.
    """
    rows: list[dict] = []
    offset, page = 0, 500
    while True:
        batch = await db_manager.list_feedback(limit=page, offset=offset, sort="recent")
        rows.extend(batch)
        if len(batch) < page:
            break
        offset += page

    seen: set[str] = set()
    unique = []
    for row in rows:
        digest = row.get("image_sha256") or ""
        if digest and digest in seen:
            continue
        seen.add(digest)
        unique.append(row)
    return unique


def _build_bundle(tag: str, entries: list[tuple[dict, Path]]) -> Path:
    """Write the zip. Runs off the event loop; copying images is slow."""
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    target = BUNDLE_DIR / f"{tag}.zip"
    partial = target.with_suffix(".zip.partial")

    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        lines = []
        for row, image in entries:
            inside = f"images/{image.name}"
            bundle.write(image, inside)
            lines.append(json.dumps({**_manifest_row(row), "image_path": inside},
                                    ensure_ascii=False))
        bundle.writestr("manifest.jsonl", "\n".join(lines) + "\n")
        bundle.writestr("README.txt", BUNDLE_README.format(
            tag=tag,
            built=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            cases=len(entries),
        ))

    # Renamed into place only once it is complete, so a download can never
    # pick up a half-written archive.
    partial.replace(target)

    for old in sorted(BUNDLE_DIR.glob("community-bundle-*.zip"), reverse=True)[KEEP_BUNDLES:]:
        old.unlink(missing_ok=True)
    return target


@router.get("/community")
async def community_model(_user: dict = Depends(require_clinician)):
    """
    The community model: what it is, which versions exist, and what is waiting.

    Readable by every clinician. Their cases are what it is made of, so they
    can see what has been built from them without being able to build one.
    """
    from rad_intel.models import registry

    live = registry.released()
    by_file = {entry["file"]: name for name, entry in live.items()}

    versions = []
    for path in reversed(_community_versions()):
        meta = await run_in_threadpool(_checkpoint_meta, path)
        scores = meta.get("metrics") or meta.get("metrics_after") or {}
        provenance = meta.get("provenance") or {}
        versions.append({
            "file": path.name,
            "name": _release_name(path.name),
            "is_released": path.name in by_file,
            "threshold": meta.get("threshold"),
            "metrics": {k: v for k, v in scores.items() if isinstance(v, (int, float))},
            "reviewed_cases": provenance.get("reviewed_cases"),
            "trained_images": provenance.get("trained_images"),
            "bundle": provenance.get("bundle"),
            "created_at": provenance.get("created_at"),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
        })

    reviewed = len(await _all_reviewed_rows())
    bundles = [
        {"file": b.name,
         "bytes": b.stat().st_size,
         "built_at": datetime.fromtimestamp(b.stat().st_mtime, timezone.utc).isoformat()}
        for b in sorted(BUNDLE_DIR.glob("community-bundle-*.zip"), reverse=True)
    ] if BUNDLE_DIR.exists() else []

    return {
        "name": COMMUNITY_MODEL,
        "reviewed_cases": reviewed,
        "versions": versions,
        "bundles": bundles,
        "note": (
            "A separate model from the one the project reports. Each run discards the "
            "previous community model and trains the same architecture from scratch on "
            "the public dataset together with every case reviewed so far, so a version "
            "is built from all the evidence rather than from the latest batch. It is "
            "scored on the same held-out partition as the reported models, and it only "
            "becomes choosable on the Analyse tab when the administrator releases it. "
            "The checkpoint the project's results describe is never written to."
        ),
    }


@router.post("/bundle",
             dependencies=[Depends(limit("feedback-bundle", 10, 3600, "training bundles"))])
async def build_bundle(request: Request, admin: dict = Depends(require_admin)):
    """
    Package every reviewed case into one zip, ready to train elsewhere.

    It freezes anything still pending first, exactly as the clinicians' export
    button does, because a bundle has to name the records it holds and they
    have to stop changing underneath it.
    """
    if job.running:
        raise HTTPException(
            status_code=409,
            detail=("A run is in progress and is reading the exported cases. Building "
                    "a bundle would freeze records underneath it."),
        )

    rows = await _all_reviewed_rows()
    entries = []
    missing = 0
    for row in rows:
        # Through the Drive fallback, so a bundle built on a server whose
        # disk was wiped still carries every radiograph rather than quietly
        # training on the handful that happen to be local.
        image = await _image_for(row)
        label = row.get("corrected_label") or row.get("predicted_label")
        if not image or label not in ("NORMAL", "PNEUMONIA"):
            missing += 1
            continue
        entries.append((row, image))

    if len(entries) < 4:
        raise HTTPException(
            status_code=400,
            detail=(f"Only {len(entries)} usable reviewed radiograph"
                    f"{'' if len(entries) == 1 else 's'} in the store. Training on so "
                    f"few teaches the model nothing and risks skewing it. Wait for more "
                    f"reviews — a mix of normal and pneumonia, including cases the model "
                    f"got wrong."),
        )
    classes = {c: sum(1 for row, _ in entries
                      if (row.get("corrected_label") or row["predicted_label"]) == c)
               for c in ("NORMAL", "PNEUMONIA")}
    if min(classes.values()) == 0:
        raise HTTPException(
            status_code=400,
            detail=("Every reviewed case in the store carries the same label. A model "
                    "trained on one class collapses, so this is refused until there are "
                    "examples of both."),
        )

    # Freeze whatever was still pending, so the bundle's records keep matching
    # it. Nothing to freeze is normal: it means somebody already exported.
    export_tag, _, frozen = await _write_export()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"community-bundle-{stamp}"
    path = await run_in_threadpool(_build_bundle, tag, entries)

    activity.record("community.bundle_built", user=admin, ip=client_address(request),
                    detail=f"{tag} cases={len(entries)} frozen={frozen}")
    return {
        "tag": tag,
        "file": path.name,
        "download": f"/api/v1/feedback/bundle/{path.name}",
        "cases": len(entries),
        "normal": classes["NORMAL"],
        "pneumonia": classes["PNEUMONIA"],
        "skipped": missing,
        "newly_frozen": frozen,
        "export_tag": export_tag or None,
        "bytes": path.stat().st_size,
        "detail": (
            f"Bundle built with {len(entries)} reviewed radiographs "
            f"({classes['NORMAL']} normal, {classes['PNEUMONIA']} pneumonia)"
            + (f" and froze {frozen} case{'' if frozen == 1 else 's'} that were still "
               f"pending." if frozen else ".")
            + " Download it, train with it, then upload the checkpoint back here. "
              "Nothing has changed for anybody using the site."
        ),
    }


@router.get("/bundle/{filename}")
async def download_bundle(filename: str, request: Request,
                          admin: dict = Depends(require_admin)):
    """
    Hand a built bundle over. Administrator only.

    It carries radiographs from the review store, so it is served the same way
    a single one is: through an authenticated endpoint, by name, with the path
    resolved and checked against the bundle directory rather than trusted.
    """
    from fastapi.responses import FileResponse

    safe = (BUNDLE_DIR / Path(filename).name).resolve()
    if (not BUNDLE_DIR.exists() or BUNDLE_DIR.resolve() not in safe.parents
            or safe.suffix != ".zip" or not safe.is_file()):
        raise HTTPException(status_code=404, detail="No such bundle.")

    activity.record("community.bundle_downloaded", user=admin, ip=client_address(request),
                    detail=safe.name)
    return FileResponse(safe, media_type="application/zip", filename=safe.name,
                        headers={"Cache-Control": "no-store"})


def _validate_checkpoint(path: Path) -> dict:
    """
    Read an uploaded checkpoint and decide whether it is one of ours.

    Loaded with weights_only=True, so nothing in the file is executed. A
    PyTorch checkpoint is a pickle, and unpickling one is running whatever it
    contains; the restricted loader takes tensors and plain data and refuses
    everything else, which is all our checkpoints hold anyway.

    Raises ValueError with something a person can act on.
    """
    import torch

    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"That file could not be read as a PyTorch checkpoint ({exc}). Upload the "
            f".pt file the training notebook saved, not a zip or a folder."
        ) from exc

    if not isinstance(blob, dict):
        raise ValueError("That checkpoint is not a dictionary, so it was not written "
                         "by the Rad-Intel training script.")

    state = blob.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("That checkpoint has no model_state_dict, so there are no "
                         "weights in it.")

    model_type = blob.get("model_type")
    if model_type != SHARED_MODEL:
        raise ValueError(
            f"That checkpoint says it holds a '{model_type}' model. The community model "
            f"is the {SHARED_MODEL} architecture, so only a {SHARED_MODEL} checkpoint "
            f"can be uploaded here."
        )

    threshold = blob.get("optimal_threshold")
    if not isinstance(threshold, (int, float)) or not 0.0 < float(threshold) < 1.0:
        raise ValueError(
            "That checkpoint carries no operating threshold. Serving it would fall back "
            "to 0.5 and disagree with the sensitivity and specificity it reports, so it "
            "is refused."
        )

    scores = blob.get("metrics") or blob.get("metrics_after") or {}
    numeric = {k: v for k, v in scores.items() if isinstance(v, (int, float))}
    for needed in ("accuracy", "auroc", "sensitivity", "specificity"):
        if needed not in numeric:
            raise ValueError(
                f"That checkpoint reports no {needed}. A model nobody can judge is worse "
                f"than no model, so it cannot be uploaded. Let the training script finish "
                f"its evaluation on the held-out partition."
            )

    # The weights have to fit the network that will be built to serve them.
    # Compared against the base checkpoint's own keys and shapes, which is
    # exact and costs no model construction.
    reference = WEIGHTS_DIR / f"best_{SHARED_MODEL}_model.pt"
    if reference.is_file():
        try:
            ref = torch.load(reference, map_location="cpu", weights_only=True, mmap=True)
            ref_state = ref.get("model_state_dict", {}) if isinstance(ref, dict) else {}
        except Exception:
            ref_state = {}
        if ref_state:
            theirs = {k: tuple(v.shape) for k, v in state.items()
                      if hasattr(v, "shape")}
            ours = {k: tuple(v.shape) for k, v in ref_state.items()
                    if hasattr(v, "shape")}
            mismatched = [k for k, shape in ours.items()
                          if k in theirs and theirs[k] != shape]
            missing = [k for k in ours if k not in theirs]
            if mismatched or len(missing) > len(ours) * 0.05:
                raise ValueError(
                    f"The weights in that checkpoint do not fit the {SHARED_MODEL} "
                    f"architecture ({len(missing)} layers missing, {len(mismatched)} the "
                    f"wrong shape). It was trained from a different model definition, so "
                    f"loading it would produce an untrained network that still returns "
                    f"confident-looking numbers."
                )

    return {
        "model_type": model_type,
        "threshold": float(threshold),
        "metrics": numeric,
        "provenance": blob.get("provenance") if isinstance(blob.get("provenance"), dict) else {},
    }


# Where a checkpoint may be fetched from.
#
# An allowlist rather than "any https URL", because this endpoint makes the
# server fetch something an address says is there, and a server that will
# fetch any address on request is a way to reach things only it can reach --
# a cloud provider's metadata service, a database on the same private
# network. These are the places somebody actually puts a 136 MB file to pass
# it on, and nothing on them is internal to us.
_FETCH_HOSTS = (
    "drive.google.com", "drive.usercontent.google.com", "docs.google.com",
    "dropbox.com", "dropboxusercontent.com",
    "github.com", "githubusercontent.com",
    "huggingface.co", "hf.co",
)
_FETCH_REDIRECTS = 5
_DRIVE_ID = re.compile(r"/d/(?P<id>[A-Za-z0-9_-]{10,})|[?&]id=(?P<q>[A-Za-z0-9_-]{10,})")


def _allowed_host(url: str) -> str | None:
    """The host, when this address is one we will fetch from. Otherwise None."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return None
    host = parsed.hostname.lower()
    for allowed in _FETCH_HOSTS:
        if host == allowed or host.endswith("." + allowed):
            return host
    return None


def _direct_link(url: str) -> str:
    """
    Turn a share link into one that returns the file itself.

    A Google Drive link opens a viewer, and a file this size also meets the
    "can't scan it for viruses" page rather than the bytes. Dropbox's share
    link renders a preview. Both have a documented form that does not.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if host.endswith("google.com"):
        match = _DRIVE_ID.search(url)
        file_id = match and (match["id"] or match["q"])
        if file_id:
            return ("https://drive.usercontent.google.com/download"
                    f"?id={file_id}&export=download&confirm=t")
    if host.endswith("dropbox.com"):
        base = url.split("?", 1)[0]
        return f"{base}?dl=1"
    return url


def _download_checkpoint(url: str, into: Path) -> Path:
    """
    Fetch a checkpoint to a file. Raises ValueError with something readable.

    Redirects are followed by hand so each hop is checked against the
    allowlist: following them automatically would let an allowed address
    hand the request on to one that is not.
    """
    import httpx

    target = into / "fetched.pt"
    current = _direct_link(url)

    with httpx.Client(timeout=httpx.Timeout(30.0, read=300.0),
                      follow_redirects=False) as client:
        for _ in range(_FETCH_REDIRECTS):
            if not _allowed_host(current):
                raise ValueError(
                    "That link is not one this server will fetch from. Put the file in "
                    "Google Drive, Dropbox, a GitHub release or Hugging Face, share it "
                    "so anyone with the link can read it, and paste that link. Or use "
                    "the file picker above instead."
                )
            with client.stream("GET", current) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    current = str(response.headers.get("location") or "")
                    if not current:
                        raise ValueError("That link redirected to nowhere.")
                    continue
                if response.status_code == 404:
                    raise ValueError(
                        "Nothing is there (404). Check the link, and check the file is "
                        "shared so that anyone with the link can read it."
                    )
                if response.status_code >= 400:
                    raise ValueError(
                        f"The link answered {response.status_code}. If it is a Google "
                        f"Drive file, make sure sharing is set to “Anyone with the "
                        f"link”, not “Restricted”."
                    )

                written = 0
                with target.open("wb") as out:
                    for chunk in response.iter_bytes(4 * 1024 * 1024):
                        written += len(chunk)
                        if written > MAX_CHECKPOINT_BYTES:
                            raise ValueError(
                                f"That file is larger than the "
                                f"{MAX_CHECKPOINT_BYTES // (1024 * 1024)} MB limit. A "
                                f"{SHARED_MODEL} checkpoint is about 136 MB."
                            )
                        out.write(chunk)
                if written < 1024:
                    raise ValueError(
                        "That link gave back a few bytes rather than a checkpoint — "
                        "usually a sharing page instead of the file. Check that anyone "
                        "with the link can read it."
                    )
                return target

    raise ValueError("That link redirected too many times.")


def _keep_community_checkpoint(staged: Path) -> tuple[Path, dict]:
    """Check a staged checkpoint and, if it passes, keep it as the next version."""
    summary = _validate_checkpoint(staged)
    target = _next_community_path()
    shutil.move(str(staged), target)
    return target, summary


def _community_response(target: Path, summary: dict, source: str) -> dict:
    _meta_cache.clear()
    return {
        "file": target.name,
        "name": _release_name(target.name),
        "metrics": summary["metrics"],
        "threshold": summary["threshold"],
        "size_mb": round(target.stat().st_size / (1024 * 1024), 1),
        "source": source,
        "detail": (
            f"Stored as {target.name}. Nothing is being served from it and nobody can "
            f"choose it yet — it is now in “Release a model to the public”, where "
            f"releasing it asks for your password again."
        ),
    }


@router.post("/community/fetch",
             dependencies=[Depends(limit("community-upload", 10, 3600,
                                         "community model uploads"))])
async def fetch_community_model(req: CommunityFetchRequest, request: Request,
                                admin: dict = Depends(require_admin)):
    """
    Pull a community model in from a share link. Administrator only.

    The alternative to pushing 136 MB up a home connection: the notebook that
    trained it saves straight to a drive, and the server fetches it from
    there at the speed those two machines talk to each other.
    """
    if not await run_in_threadpool(accounts.identify_admin, admin["email"], req.password):
        activity.record("community.fetch_refused", user=admin, ip=client_address(request),
                        detail="wrong password", outcome="refused")
        raise HTTPException(status_code=403,
                            detail="Password incorrect. Nothing was fetched.")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    holding = Path(tempfile.mkdtemp(prefix="radintel-fetch-", dir=WEIGHTS_DIR))
    try:
        try:
            staged = await run_in_threadpool(_download_checkpoint, req.url.strip(), holding)
            target, summary = await run_in_threadpool(_keep_community_checkpoint, staged)
        except ValueError as exc:
            activity.record("community.fetch_refused", user=admin,
                            ip=client_address(request), detail=str(exc)[:200],
                            outcome="refused")
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.warning("Could not fetch a community checkpoint: %s", exc)
            raise HTTPException(
                status_code=502,
                detail=("The server could not reach that link. Check it opens in your "
                        "own browser, then try again."),
            )
    finally:
        shutil.rmtree(holding, ignore_errors=True)

    activity.record("community.model_fetched", user=admin, ip=client_address(request),
                    detail=f"{target.name} acc={summary['metrics'].get('accuracy')}")
    return _community_response(target, summary, source="link")


@router.post("/community/upload",
             dependencies=[Depends(limit("community-upload", 10, 3600,
                                         "community model uploads"))])
async def upload_community_model(
    request: Request,
    file: UploadFile = File(..., description="Checkpoint written by the training script"),
    password: str = Form(..., description="The administrator's password"),
    admin: dict = Depends(require_admin),
):
    """
    Take in a community model trained elsewhere. Administrator only.

    The password is asked for because this writes a file that can then be put
    in front of the public. It is checked before the upload is read, so a
    wrong one costs nothing.

    Uploading does not release anything. The checkpoint lands as a new
    numbered version, is scored and listed, and appears in the release panel
    for a separate, deliberate act.
    """
    if not await run_in_threadpool(accounts.identify_admin, admin["email"], password):
        activity.record("community.upload_refused", user=admin, ip=client_address(request),
                        detail="wrong password", outcome="refused")
        raise HTTPException(status_code=403,
                            detail="Password incorrect. Nothing was uploaded.")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    holding = Path(tempfile.mkdtemp(prefix="radintel-upload-", dir=WEIGHTS_DIR))
    staged = holding / "upload.pt"
    try:
        written = 0
        with staged.open("wb") as out:
            while chunk := await file.read(4 * 1024 * 1024):
                written += len(chunk)
                if written > MAX_CHECKPOINT_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(f"That file is larger than the "
                                f"{MAX_CHECKPOINT_BYTES // (1024 * 1024)} MB limit. A "
                                f"{SHARED_MODEL} checkpoint is about 136 MB."),
                    )
                out.write(chunk)
        if not written:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        try:
            target, summary = await run_in_threadpool(_keep_community_checkpoint, staged)
        except ValueError as exc:
            activity.record("community.upload_refused", user=admin,
                            ip=client_address(request), detail=str(exc)[:200],
                            outcome="refused")
            raise HTTPException(status_code=400, detail=str(exc))
    finally:
        shutil.rmtree(holding, ignore_errors=True)

    activity.record("community.model_uploaded", user=admin, ip=client_address(request),
                    detail=f"{target.name} acc={summary['metrics'].get('accuracy')}")
    return _community_response(target, summary, source="upload")


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

    path = await _image_for(record)
    if not path:
        raise HTTPException(status_code=404, detail="Image file is missing.")

    media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(path, media_type=media, headers={
        # The image is only valid for whoever holds the session that fetched
        # it, so it must not be left in a shared cache.
        "Cache-Control": "private, max-age=300",
    })


@router.get("/drive")
async def drive_status(_admin: dict = Depends(require_admin)):
    """
    Whether reviewed radiographs have a copy anywhere but this disk.

    Administrator only: it names a folder in a personal Drive, and a
    clinician has no business knowing where the administrator keeps things.
    """
    status = await run_in_threadpool(drive.store.check)
    waiting = await db_manager.feedback_without_drive_copy(limit=1000)
    return {
        **status,
        "awaiting_backup": len(waiting),
        "scope_note": (
            "Rad-Intel holds one Google permission: access to files it created "
            "itself. It cannot read, list or even see anything else in that "
            "Drive — your own documents, the trained checkpoints, any of it."
        ),
    }


@router.post("/drive/backfill",
             dependencies=[Depends(limit("drive-backfill", 12, 3600, "backups"))])
async def backfill_drive(request: Request, admin: dict = Depends(require_admin)):
    """
    Copy to Drive every radiograph that is not there yet.

    For two cases: radiographs reviewed before Drive was connected, and ones
    whose upload failed at the time because Google was unreachable. Submitting
    a review deliberately does not fail when Drive does, so this is where that
    debt is paid off.
    """
    if not drive.configured():
        raise HTTPException(
            status_code=503,
            detail=("Google Drive is not set up on this server, so there is nowhere "
                    "to copy anything to. Run scripts/connect_drive.py."),
        )

    waiting = await db_manager.feedback_without_drive_copy(limit=500)
    done, failed, gone = 0, 0, 0
    for row in waiting:
        path = _resolve_image(row.get("image_path"))
        if not path:
            gone += 1
            continue
        try:
            file_id = await run_in_threadpool(
                drive.store.upload, path, path.name,
                "image/png" if path.suffix.lower() == ".png" else "image/jpeg",
            )
        except Exception as exc:
            logger.warning("Backfill failed for %s: %s", row["id"][:8], exc)
            failed += 1
            # One failure is usually all of them -- no credentials, no room.
            # Stopping says so in one message instead of five hundred.
            break
        await db_manager.set_drive_file_id(row["id"], file_id)
        done += 1

    activity.record("drive.backfill", user=admin, ip=client_address(request),
                    detail=f"copied={done} failed={failed} missing_locally={gone}",
                    outcome="error" if failed else "ok")
    remaining = len(await db_manager.feedback_without_drive_copy(limit=1000))
    return {
        "copied": done,
        "failed": failed,
        "missing_locally": gone,
        "remaining": remaining,
        "detail": (
            (f"Copied {done} radiograph{'' if done == 1 else 's'} to Drive. "
             if done else "Nothing was copied. ")
            + (f"{gone} could not be found on this disk and are not in Drive either — "
               f"those are gone. " if gone else "")
            + (f"Stopped after a failure with {remaining} still to do; check the "
               f"connection and the account's free space."
               if failed else
               ("Everything reviewed so far is now in Drive."
                if not remaining else f"{remaining} still to do."))
        ),
    }


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
