"""
Released models: the versions anybody may choose on the Analyse tab.

Fine-tuning writes a new checkpoint and changes nothing. That is deliberate:
a model trained on reviewed cases has to be looked at before it is used. This
is the step that puts one in front of people -- the administrator releases a
version, and it appears in the model list on the public tool alongside the
ones the project reports.

A release is a file, not a flag in memory, so it survives a restart. It
records which architecture the weights belong to, the scores the version was
measured at, and who released it and when. The base checkpoints are never
listed here: they are built in, always available, and cannot be withdrawn.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from rad_intel.config import settings

logger = logging.getLogger(__name__)

WEIGHTS_DIR = settings.BASE_DIR / "weights"
REGISTRY_PATH = WEIGHTS_DIR / "released_models.json"

# best_hybrid_v2_model.pt           -> lineage "hybrid",           name "hybrid_v2"
# best_community_hybrid_v1_model.pt -> lineage "community_hybrid", name "community_hybrid_v1"
_VERSIONED = re.compile(r"^best_(?P<lineage>.+)_v(?P<version>\d+)_model\.pt$")


def describe(filename: str) -> tuple[str, str] | None:
    """
    (lineage, public name) for a versioned checkpoint, or None.

    The lineage is where the weights came from, not necessarily which network
    they are: "community_hybrid" is a lineage trained on the Kaggle data plus
    the reviewed cases, and the network it holds is a plain hybrid. The
    network comes from the checkpoint's own model_type, which is why
    `release` takes it separately.
    """
    match = _VERSIONED.match(filename)
    if not match:
        return None
    return match["lineage"], f"{match['lineage']}_v{match['version']}"


def _load() -> dict[str, dict]:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Ignoring unreadable %s: %s", REGISTRY_PATH.name, exc)
        return {}


def _save(data: dict[str, dict]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def released() -> dict[str, dict]:
    """Released versions whose checkpoint is still on disk."""
    return {
        name: entry for name, entry in _load().items()
        if (WEIGHTS_DIR / str(entry.get("file", ""))).is_file()
    }


def resolve(name: str) -> tuple[str, Path] | None:
    """(architecture, checkpoint path) for a released name, or None."""
    entry = released().get(name)
    if not entry:
        return None
    return entry["architecture"], WEIGHTS_DIR / entry["file"]


def release(filename: str, released_by: str, metrics: dict | None = None,
            note: str | None = None, architecture: str | None = None) -> dict:
    """
    Publish a trained checkpoint. Returns its registry entry.

    `architecture` is the network the weights actually belong to, read from
    the checkpoint's own model_type by the caller. It matters because a
    lineage name need not be an architecture name: the community model is
    released as community_hybrid_v1 but *is* a hybrid, and serving has to
    build a hybrid to load it into. Without this the filename would be the
    only source of that answer, and an architecture called
    "community_hybrid" does not exist.

    Raises ValueError when the file is not a versioned checkpoint sitting in
    the weights directory, which is also what stops a path from somewhere
    else being published by name.
    """
    safe = Path(filename).name
    path = (WEIGHTS_DIR / safe).resolve()
    if WEIGHTS_DIR.resolve() not in path.parents or not path.is_file():
        raise ValueError("No such checkpoint in the weights directory.")

    described = describe(safe)
    if not described:
        raise ValueError(
            "Only a versioned checkpoint can be released, e.g. best_hybrid_v2_model.pt "
            "or best_community_hybrid_v1_model.pt. The base checkpoints are always "
            "available and are not released or withdrawn."
        )
    lineage, name = described

    data = _load()
    data[name] = {
        "file": safe,
        "architecture": architecture or lineage,
        "lineage": lineage,
        "metrics": metrics or {},
        "note": (note or "").strip()[:300] or None,
        "released_by": released_by,
        "released_at": datetime.now(timezone.utc).isoformat(),
    }
    _save(data)
    logger.info("Released %s as %s.", safe, name)
    return {"name": name, **data[name]}


def withdraw(name: str) -> bool:
    """Take a version off the public list. The checkpoint itself is untouched."""
    data = _load()
    if name not in data:
        return False
    data.pop(name)
    _save(data)
    logger.info("Withdrew %s.", name)
    return True


def candidates() -> list[dict]:
    """
    Fine-tuned checkpoints on disk, and whether each is released.

    What the administrator picks from.
    """
    live = released()
    by_file = {entry["file"]: name for name, entry in live.items()}
    out = []
    for path in sorted(WEIGHTS_DIR.glob("best_*_v*_model.pt")):
        described = describe(path.name)
        if not described:
            continue
        lineage, name = described
        out.append({
            "file": path.name,
            "name": name,
            "lineage": lineage,
            "is_released": path.name in by_file,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
        })
    return out
