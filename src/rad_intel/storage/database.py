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


class DatabaseManager:
    """Manages asynchronous SQLite connection and query execution."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._initialized: bool = False

    async def init_db(self):
        """Initializes database schema."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(CREATE_TABLE_SQL)
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


db_manager = DatabaseManager()
