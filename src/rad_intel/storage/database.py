"""
Asynchronous SQLite storage service for Rad-Intel inference and report history.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import aiosqlite

from rad_intel.config import BASE_DIR

DB_PATH = BASE_DIR / "rad_intel.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    model_name TEXT NOT NULL,
    prediction_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    probabilities_json TEXT NOT NULL,
    dominant_zone TEXT,
    is_bilateral INTEGER NOT NULL DEFAULT 0,
    report_markdown TEXT NOT NULL,
    engine TEXT NOT NULL,
    patient_id TEXT,
    execution_time_ms REAL NOT NULL
);
"""


CREATE_FEEDBACK_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS clinician_feedback (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    model_version TEXT NOT NULL,
    preprocessing_version TEXT NOT NULL,
    image_sha256 TEXT NOT NULL,
    image_path TEXT,
    predicted_label TEXT NOT NULL,
    predicted_probability REAL NOT NULL,
    threshold REAL NOT NULL,
    verdict TEXT NOT NULL,
    corrected_label TEXT,
    note TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    exported_in TEXT
);
"""


class DatabaseManager:
    """Manages asynchronous SQLite connection and query execution."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._initialized: bool = False

    async def init_db(self):
        """Initializes database schema."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(CREATE_TABLE_SQL)
            await db.execute(CREATE_FEEDBACK_TABLE_SQL)
            await db.commit()
        self._initialized = True

    async def _ensure_db(self):
        if not self._initialized:
            await self.init_db()

    async def save_analysis(
        self,
        model_name: str,
        prediction_class: str,
        confidence: float,
        probabilities: dict[str, float],
        report_markdown: str,
        engine: str,
        dominant_zone: str | None = None,
        is_bilateral: bool = False,
        patient_id: str | None = None,
        execution_time_ms: float = 0.0,
        record_id: str | None = None,
    ) -> str:
        """Inserts an analysis record and returns its UUID."""
        await self._ensure_db()
        rec_id = record_id or str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        probs_json = json.dumps(probabilities)

        query = """
        INSERT INTO analyses (
            id, created_at, model_name, prediction_class, confidence,
            probabilities_json, dominant_zone, is_bilateral, report_markdown,
            engine, patient_id, execution_time_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            rec_id,
            created_at,
            model_name,
            prediction_class,
            float(confidence),
            probs_json,
            dominant_zone,
            1 if is_bilateral else 0,
            report_markdown,
            engine,
            patient_id,
            float(execution_time_ms),
        )

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(query, params)
            await db.commit()

        return rec_id

    async def get_analysis(self, record_id: str) -> dict[str, Any] | None:
        """Retrieves a single analysis record by ID."""
        await self._ensure_db()
        query = "SELECT * FROM analyses WHERE id = ?;"
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, (record_id,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    item = dict(row)
                    item["probabilities"] = json.loads(item["probabilities_json"])
                    item["is_bilateral"] = bool(item["is_bilateral"])
                    return item
        return None

    async def list_analyses(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """Lists historical analyses ordered by creation time descending."""
        await self._ensure_db()
        query = "SELECT * FROM analyses ORDER BY created_at DESC LIMIT ? OFFSET ?;"
        results = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, (limit, offset)) as cursor:
                rows = await cursor.fetchall()
                for row in rows:
                    item = dict(row)
                    item["probabilities"] = json.loads(item["probabilities_json"])
                    item["is_bilateral"] = bool(item["is_bilateral"])
                    results.append(item)
        return results

    # --- Clinician feedback -------------------------------------------------
    #
    # A review queue, deliberately separate from `analyses`. Nothing in here
    # touches the Kaggle dataset or its split: reviewed cases accumulate in
    # their own table and their own directory, and become training material
    # only through an explicit, versioned export.

    async def save_feedback(
        self,
        reviewer: str,
        model_version: str,
        preprocessing_version: str,
        image_sha256: str,
        predicted_label: str,
        predicted_probability: float,
        threshold: float,
        verdict: str,
        corrected_label: str | None = None,
        note: str | None = None,
        image_path: str | None = None,
    ) -> str:
        """Inserts a feedback record and returns its UUID."""
        await self._ensure_db()
        rec_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        query = """
        INSERT INTO clinician_feedback (
            id, created_at, updated_at, reviewer, model_version, preprocessing_version,
            image_sha256, image_path, predicted_label, predicted_probability, threshold,
            verdict, corrected_label, note, status, exported_in
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', NULL);
        """
        params = (
            rec_id, now, now, reviewer, model_version, preprocessing_version,
            image_sha256, image_path, predicted_label, float(predicted_probability),
            float(threshold), verdict, corrected_label, note,
        )
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(query, params)
            await db.commit()
        return rec_id

    async def list_feedback(
        self, status: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Returns feedback records, newest first, optionally filtered by status."""
        await self._ensure_db()
        where, params = "", []
        if status:
            where = "WHERE status = ?"
            params.append(status)
        params.extend([limit, offset])
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                f"SELECT * FROM clinician_feedback {where} "
                "ORDER BY datetime(created_at) DESC LIMIT ? OFFSET ?;",
                params,
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def get_feedback(self, record_id: str) -> dict[str, Any] | None:
        await self._ensure_db()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM clinician_feedback WHERE id = ?;", (record_id,)
            ) as cur:
                row = await cur.fetchone()
                return dict(row) if row else None

    async def update_feedback(
        self,
        record_id: str,
        verdict: str | None = None,
        corrected_label: str | None = None,
        note: str | None = None,
        status: str | None = None,
    ) -> bool:
        """Edits a queued review. Returns False when the id is unknown."""
        await self._ensure_db()
        sets, params = ["updated_at = ?"], [datetime.now(timezone.utc).isoformat()]
        for column, value in (
            ("verdict", verdict),
            ("corrected_label", corrected_label),
            ("note", note),
            ("status", status),
        ):
            if value is not None:
                sets.append(f"{column} = ?")
                params.append(value)
        params.append(record_id)
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                f"UPDATE clinician_feedback SET {', '.join(sets)} WHERE id = ?;", params
            )
            await db.commit()
            return cur.rowcount > 0

    async def mark_feedback_exported(self, record_ids: list[str], export_tag: str) -> int:
        """Marks records as exported under a tag. Returns the number updated."""
        await self._ensure_db()
        if not record_ids:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        placeholders = ",".join("?" for _ in record_ids)
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                f"UPDATE clinician_feedback SET status = 'exported', exported_in = ?, "
                f"updated_at = ? WHERE id IN ({placeholders});",
                [export_tag, now, *record_ids],
            )
            await db.commit()
            return cur.rowcount

    async def feedback_counts(self) -> dict[str, int]:
        """Counts by status, for the review queue header."""
        await self._ensure_db()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT status, COUNT(*) FROM clinician_feedback GROUP BY status;"
            ) as cur:
                rows = await cur.fetchall()
        counts = {status: count for status, count in rows}
        counts["total"] = sum(counts.values())
        return counts


db_manager = DatabaseManager()
