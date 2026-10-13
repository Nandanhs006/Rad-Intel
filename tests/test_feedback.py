"""
Clinician feedback store: validation, persistence, editing and export.

The property worth protecting here is that review material stays separate
from the evaluated dataset and never reaches a checkpoint on its own. These
tests use a temporary database so the real queue is untouched.
"""

import pathlib
import tempfile

import pytest

from rad_intel.storage.database import DatabaseManager

# pytest-asyncio runs in auto mode (see pyproject.toml), so async tests need no marker.


@pytest.fixture
def db():
    return DatabaseManager(pathlib.Path(tempfile.mkdtemp()) / "feedback_test.db")


def _case(**overrides):
    base = dict(
        reviewer="Dr Test",
        model_version="hybrid",
        preprocessing_version="clahe2.0-8x8_pil-bilinear-224_imagenet-norm",
        image_sha256="a" * 64,
        predicted_label="PNEUMONIA",
        predicted_probability=0.87,
        threshold=0.5574,
        verdict="correct",
    )
    base.update(overrides)
    return base


async def test_submission_records_the_provenance_fields(db):
    """Model and preprocessing versions must travel with the review."""
    rec_id = await db.save_feedback(**_case())
    row = await db.get_feedback(rec_id)
    assert row["model_version"] == "hybrid"
    assert row["preprocessing_version"].startswith("clahe2.0")
    assert row["image_sha256"] == "a" * 64
    assert row["threshold"] == pytest.approx(0.5574)
    assert row["status"] == "pending"


async def test_correction_is_stored_separately_from_the_prediction(db):
    rec_id = await db.save_feedback(
        **_case(verdict="incorrect", corrected_label="NORMAL", note="framing artefact")
    )
    row = await db.get_feedback(rec_id)
    assert row["predicted_label"] == "PNEUMONIA"
    assert row["corrected_label"] == "NORMAL"
    assert row["note"] == "framing artefact"


async def test_a_queued_review_can_be_edited(db):
    rec_id = await db.save_feedback(**_case())
    assert await db.update_feedback(rec_id, note="revised", verdict="incorrect",
                                    corrected_label="NORMAL")
    row = await db.get_feedback(rec_id)
    assert (row["note"], row["verdict"], row["corrected_label"]) == ("revised", "incorrect", "NORMAL")


async def test_editing_an_unknown_record_reports_failure(db):
    assert await db.update_feedback("does-not-exist", note="x") is False


async def test_export_marks_records_and_tags_them(db):
    ids = [await db.save_feedback(**_case(image_sha256=str(i) * 64)) for i in range(3)]
    assert await db.mark_feedback_exported(ids, "reviewed-export-TEST") == 3
    for rec_id in ids:
        row = await db.get_feedback(rec_id)
        assert row["status"] == "exported"
        assert row["exported_in"] == "reviewed-export-TEST"


async def test_exported_records_leave_the_pending_queue(db):
    pending_id = await db.save_feedback(**_case())
    exported_id = await db.save_feedback(**_case(image_sha256="b" * 64))
    await db.mark_feedback_exported([exported_id], "reviewed-export-TEST")

    pending = await db.list_feedback(status="pending")
    assert [r["id"] for r in pending] == [pending_id]
    assert (await db.feedback_counts()) == {"pending": 1, "exported": 1, "total": 2}


async def test_exporting_nothing_is_a_no_op(db):
    assert await db.mark_feedback_exported([], "reviewed-export-TEST") == 0


async def test_feedback_table_is_independent_of_the_analyses_table(db):
    """Writing a review must not touch the analysis history."""
    await db.save_feedback(**_case())
    assert await db.list_analyses(limit=10) == []
