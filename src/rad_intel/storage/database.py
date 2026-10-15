"""
Asynchronous SQLite storage service for Rad-Intel inference and report history.
"""

import asyncio
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
    institution TEXT,
    predicted_label TEXT NOT NULL,
    predicted_probability REAL NOT NULL,
    threshold REAL NOT NULL,
    verdict TEXT NOT NULL,
    corrected_label TEXT,
    note TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    exported_in TEXT,
    drive_file_id TEXT
);
"""


class DatabaseManager:
    """Manages asynchronous SQLite connection and query execution."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._initialized: bool = False
        # Several requests arrive together, and every one of them calls
        # _ensure_db. Without this each would run the schema setup -- which
        # takes an exclusive lock to switch on write-ahead logging -- and they
        # would block each other on the very statement meant to stop that.
        self._init_lock = asyncio.Lock()

    def _open(self):
        """A connection that waits for a lock rather than failing at once."""
        return aiosqlite.connect(self.db_path, timeout=15.0)

    async def init_db(self):
        """Initializes database schema."""
        async with self._open() as db:
            # Write-ahead logging and a busy timeout, matching auth/store.py.
            # This file is shared with the account tables and with whatever
            # the administrator is running on the command line, and on the
            # defaults a page load's worth of parallel requests was enough to
            # produce "database is locked".
            await db.execute("PRAGMA journal_mode = WAL;")
            await db.execute("PRAGMA synchronous = NORMAL;")
            await db.execute("PRAGMA busy_timeout = 15000;")
            await db.execute(CREATE_TABLE_SQL)
            await db.execute(CREATE_FEEDBACK_TABLE_SQL)
            # Additive migrations for databases created before these columns.
            # `owner_id` is what makes a review editable by the clinician who
            # submitted it and nobody else; rows predating it have no owner and
            # are therefore administrator-only, which is the safe default.
            async with db.execute("PRAGMA table_info(clinician_feedback);") as cur:
                columns = {row[1] for row in await cur.fetchall()}
            # `drive_file_id` is where the durable copy of the radiograph is.
            # A row without one has never reached Drive; that is what the
            # administrator's "back up what is missing" sweep looks for.
            for column, decl in (("institution", "TEXT"), ("owner_id", "TEXT"),
                                 ("drive_file_id", "TEXT")):
                if column not in columns:
                    await db.execute(
                        f"ALTER TABLE clinician_feedback ADD COLUMN {column} {decl};"
                    )
            await db.commit()
        self._initialized = True

    async def _ensure_db(self):
        if self._initialized:
            return
        async with self._init_lock:
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

        async with self._open() as db:
            await db.execute(query, params)
            await db.commit()

        return rec_id

    async def get_analysis(self, record_id: str) -> dict[str, Any] | None:
        """Retrieves a single analysis record by ID."""
        await self._ensure_db()
        query = "SELECT * FROM analyses WHERE id = ?;"
        async with self._open() as db:
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
        async with self._open() as db:
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
        institution: str | None = None,
        owner_id: str | None = None,
        drive_file_id: str | None = None,
    ) -> str:
        """Inserts a feedback record and returns its UUID."""
        await self._ensure_db()
        rec_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        query = """
        INSERT INTO clinician_feedback (
            id, created_at, updated_at, reviewer, model_version, preprocessing_version,
            image_sha256, image_path, institution, predicted_label, predicted_probability,
            threshold, verdict, corrected_label, note, status, exported_in, owner_id,
            drive_file_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', NULL, ?, ?);
        """
        params = (
            rec_id, now, now, reviewer, model_version, preprocessing_version,
            image_sha256, image_path, institution, predicted_label,
            float(predicted_probability), float(threshold), verdict, corrected_label, note,
            owner_id, drive_file_id,
        )
        async with self._open() as db:
            await db.execute(query, params)
            await db.commit()
        return rec_id

    # Sort orders the queue offers. Kept as a fixed map rather than
    # interpolating a column name from the request, so the sort parameter can
    # never become a way to inject SQL.
    _FEEDBACK_ORDERS = {
        "recent": "created_at DESC",
        "oldest": "created_at ASC",
        "name": "reviewer COLLATE NOCASE ASC, created_at DESC",
        "institution": "institution COLLATE NOCASE ASC, created_at DESC",
    }

    async def list_feedback(
        self, status: str | None = None, limit: int = 100, offset: int = 0,
        owner_id: str | None = None, sort: str = "recent",
    ) -> list[dict[str, Any]]:
        """
        Returns feedback records, newest first by default.

        `owner_id` narrows the queue to one clinician's own submissions, which
        is what the "mine" filter uses: with every case visible to everybody,
        finding your own to correct is otherwise a scroll.
        """
        await self._ensure_db()
        clauses, params = [], []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if owner_id:
            clauses.append("owner_id = ?")
            params.append(owner_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order = self._FEEDBACK_ORDERS.get(sort, self._FEEDBACK_ORDERS["recent"])
        params.extend([limit, offset])
        async with self._open() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                f"SELECT * FROM clinician_feedback {where} "
                f"ORDER BY {order} LIMIT ? OFFSET ?;",
                params,
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def get_feedback(self, record_id: str) -> dict[str, Any] | None:
        await self._ensure_db()
        async with self._open() as db:
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
        async with self._open() as db:
            cur = await db.execute(
                f"UPDATE clinician_feedback SET {', '.join(sets)} WHERE id = ?;", params
            )
            await db.commit()
            return cur.rowcount > 0

    async def set_drive_file_id(self, record_id: str, drive_file_id: str) -> bool:
        """
        Record where a radiograph's durable copy is.

        Separate from update_feedback, which is the clinician editing their
        own review and touches updated_at. This changes nothing a person
        wrote, so it must not make a record look freshly edited.
        """
        await self._ensure_db()
        async with self._open() as db:
            cur = await db.execute(
                "UPDATE clinician_feedback SET drive_file_id = ? WHERE id = ?;",
                (drive_file_id, record_id),
            )
            await db.commit()
            return cur.rowcount > 0

    async def feedback_without_drive_copy(self, limit: int = 500) -> list[dict[str, Any]]:
        """Reviews whose radiograph has never reached Drive. Oldest first."""
        await self._ensure_db()
        async with self._open() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM clinician_feedback "
                "WHERE (drive_file_id IS NULL OR drive_file_id = '') "
                "AND image_path IS NOT NULL ORDER BY created_at ASC LIMIT ?;",
                (limit,),
            ) as cur:
                return [dict(row) for row in await cur.fetchall()]

    async def delete_feedback(self, record_id: str) -> bool:
        """Removes one review. Returns False when the id is unknown."""
        await self._ensure_db()
        async with self._open() as db:
            cur = await db.execute(
                "DELETE FROM clinician_feedback WHERE id = ?;", (record_id,)
            )
            await db.commit()
            return cur.rowcount > 0

    async def count_feedback_for_image(self, image_sha256: str) -> int:
        """
        How many reviews still reference an image.

        Images are stored once per distinct file and shared between records, so
        the file may only be removed when the last review of it is gone.
        """
        await self._ensure_db()
        async with self._open() as db:
            async with db.execute(
                "SELECT COUNT(*) FROM clinician_feedback WHERE image_sha256 = ?;",
                (image_sha256,),
            ) as cur:
                (count,) = await cur.fetchone()
        return int(count)

    async def mark_feedback_exported(self, record_ids: list[str], export_tag: str) -> int:
        """Marks records as exported under a tag. Returns the number updated."""
        await self._ensure_db()
        if not record_ids:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        placeholders = ",".join("?" for _ in record_ids)
        async with self._open() as db:
            cur = await db.execute(
                f"UPDATE clinician_feedback SET status = 'exported', exported_in = ?, "
                f"updated_at = ? WHERE id IN ({placeholders});",
                [export_tag, now, *record_ids],
            )
            await db.commit()
            return cur.rowcount

    async def feedback_institutions(self) -> dict[str, int]:
        """Review counts per institution, so multi-site coverage is visible."""
        await self._ensure_db()
        async with self._open() as db:
            async with db.execute(
                "SELECT COALESCE(NULLIF(TRIM(institution), ''), 'unspecified'), COUNT(*) "
                "FROM clinician_feedback GROUP BY 1 ORDER BY 2 DESC;"
            ) as cur:
                return {name: count for name, count in await cur.fetchall()}

    async def feedback_counts(self) -> dict[str, int]:
        """Counts by status, for the review queue header."""
        await self._ensure_db()
        async with self._open() as db:
            async with db.execute(
                "SELECT status, COUNT(*) FROM clinician_feedback GROUP BY status;"
            ) as cur:
                rows = await cur.fetchall()
        counts = {status: count for status, count in rows}
        counts["total"] = sum(counts.values())
        return counts


db_manager = DatabaseManager()
