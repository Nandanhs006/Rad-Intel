r"""
Train the community model: the fusion architecture, from scratch, on the
public dataset together with every clinician-reviewed case.

This is the second lineage. It is not a fine-tune of the checkpoint the paper
reports -- it discards whatever the previous community version was and builds
a new model from ImageNet-initialised backbones on all the evidence collected
so far. Each run is therefore a model made from everything, not a model
carrying the accumulated drift of every run before it.

What it guarantees
------------------
* ``weights/best_hybrid_model.pt`` is never read as a starting point and never
  written to. That file is the checkpoint the paper's numbers describe.
* The 624-image Kaggle test partition is never trained on. It is the only
  yardstick that makes this model's scores readable beside the reported ones,
  so it is used solely for scoring -- the same split, the same seed, the same
  function the paper's own training used.
* Reviewed cases join the *training* split only, never the validation or test
  split.
* The output is a new numbered file. Nothing is promoted, released or served
  by running this.
* Preprocessing and the training recipe are the shared ones, imported rather
  than reimplemented, so the only difference from the reported model is the
  data it saw.

Where to run it
---------------
On a GPU, not on the web server. A full retrain is hours of compute; the host
that serves the site should serve the site. The intended route is the free
GPU in Google Colab -- see COMMUNITY_MODEL.md, which has the cells to paste.

Usage
-----
    # with the zip the administrator console produced
    python scripts/train_community_model.py --bundle community-bundle-....zip

    # or with a manifest already unpacked somewhere
    python scripts/train_community_model.py \
        --manifest reviewed/manifest.jsonl --images reviewed/images

    # point at the public dataset explicitly if it is not auto-detected
    python scripts/train_community_model.py --bundle b.zip --data-dir /data/chest_xray

The file it writes, ``best_community_hybrid_v<n>_model.pt``, is what you
upload in the administrator console under "The community model".
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import torch

from rad_intel.config import settings
from rad_intel.training.trainer import HybridTrainer, TrainingConfig

LABELS = {"NORMAL": 0, "PNEUMONIA": 1}
LINEAGE = "community_hybrid"
ARCHITECTURE = "hybrid"


# ------------------------------------------------------------------ the bundle

def unpack_bundle(bundle: Path, into: Path) -> tuple[Path, Path]:
    """
    Extract a bundle from the console. Returns (manifest, images directory).

    Entries are checked rather than trusted: a zip can name a path that
    escapes the directory it is extracted into, and this one arrives as a
    file somebody downloaded.
    """
    if not bundle.is_file():
        sys.exit(f"Bundle not found: {bundle}")

    into.mkdir(parents=True, exist_ok=True)
    root = into.resolve()
    with zipfile.ZipFile(bundle) as archive:
        for member in archive.namelist():
            target = (root / member).resolve()
            if root not in target.parents and target != root:
                sys.exit(f"Refusing to extract {member!r}: it points outside the "
                         f"extraction directory.")
        archive.extractall(into)

    manifest = into / "manifest.jsonl"
    if not manifest.is_file():
        sys.exit(f"{bundle.name} has no manifest.jsonl in it. Download the bundle "
                 f"again from the administrator console.")
    return manifest, into / "images"


def load_reviewed(manifest: Path, images: Path) -> list[tuple[Path, int]]:
    """Read a manifest into (image path, label) pairs, skipping what is unusable."""
    if not manifest.is_file():
        sys.exit(f"Manifest not found: {manifest}")

    samples: list[tuple[Path, int]] = []
    skipped = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        label = LABELS.get((row.get("label") or "").upper())
        recorded = str(row.get("image_path") or "")
        # A bundle's manifest says "images/<sha>.jpg". A manifest written by
        # the server says where the file sits on the server. Either way only
        # the filename matters here, and it is resolved inside the images
        # directory we were given.
        candidate = (images / Path(recorded).name).resolve()
        if label is None or images.resolve() not in candidate.parents \
                or not candidate.is_file():
            skipped += 1
            continue
        samples.append((candidate, label))

    if skipped:
        print(f"  skipped {skipped} entries with a missing image or unusable label")
    return samples


# ------------------------------------------------------------------- the output

def find_dataset() -> str | None:
    """
    Locate the public dataset without requiring --data-dir.

    The shared loader probes its own conventional paths, and this repository
    keeps the dataset beside the paper, so that location is checked here
    rather than changing a module the evaluation also uses.
    """
    for candidate in (
        settings.BASE_DIR / "Rad-Intel paper" / "chest_xray",
        settings.BASE_DIR / "chest_xray",
        settings.BASE_DIR / "data" / "chest_xray",
    ):
        if (candidate / "test").is_dir():
            return str(candidate)
    return None


def next_version(weights_dir: Path) -> Path:
    n = 1
    while (weights_dir / f"best_{LINEAGE}_v{n}_model.pt").exists():
        n += 1
    return weights_dir / f"best_{LINEAGE}_v{n}_model.pt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path,
                    help="Zip from the administrator console")
    ap.add_argument("--manifest", type=Path,
                    help="Manifest, if the bundle is already unpacked")
    ap.add_argument("--images", type=Path,
                    help="Directory holding the reviewed radiographs")
    ap.add_argument("--data-dir", default=None,
                    help="Public chest_xray dataset root (auto-detected if omitted)")
    ap.add_argument("--out", type=Path, default=settings.BASE_DIR / "weights",
                    help="Where to write the checkpoint")
    ap.add_argument("--epochs", type=int, default=20,
                    help="Phase 1: head and attention, on cached features")
    ap.add_argument("--finetune-epochs", type=int, default=10,
                    help="Phase 2: end to end through both backbones")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--finetune-batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    if not args.bundle and not args.manifest:
        ap.error("pass --bundle, or --manifest together with --images")
    if args.manifest and not args.images:
        ap.error("--manifest needs --images")

    scratch = Path(tempfile.mkdtemp(prefix="radintel-community-"))
    if args.bundle:
        manifest, images = unpack_bundle(args.bundle, scratch / "bundle")
        print(f"Bundle: {args.bundle.name}")
    else:
        manifest, images = args.manifest, args.images

    print(f"Manifest: {manifest}")
    reviewed = load_reviewed(manifest, images)
    counts = {c: sum(1 for _, label in reviewed if label == c) for c in (0, 1)}
    print(f"  reviewed cases: {len(reviewed)}  "
          f"(normal {counts[0]}, pneumonia {counts[1]})")

    if len(reviewed) < 4:
        sys.exit(f"Only {len(reviewed)} usable reviewed cases. Collect more before "
                 f"training a community model.")
    if min(counts.values()) == 0:
        sys.exit("Reviewed cases contain only one class. Training on this would "
                 "collapse the classifier; collect examples of both.")

    args.out.mkdir(parents=True, exist_ok=True)
    target = next_version(args.out)
    print(f"\nWill write {target.name}")
    print("The checkpoint the paper reports, best_hybrid_model.pt, is not read as a "
          "starting point and is not written to.\n")

    config = TrainingConfig(
        model_type=ARCHITECTURE,
        epochs=args.epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        finetune_batch_size=args.finetune_batch_size,
        learning_rate=args.lr,
        data_dir=args.data_dir or find_dataset(),
        checkpoint_dir=args.out,
        checkpoint_name=target.name,
        extra_train_samples=reviewed,
        # The two flags that keep this run in its own lane.
        sync_default_checkpoint=False,
        cache_tag=f"community_{len(reviewed)}",
    )

    trainer = HybridTrainer(config=config)
    started = datetime.now(timezone.utc)
    model, metrics = trainer.train_cached()
    minutes = (datetime.now(timezone.utc) - started).total_seconds() / 60

    # The trainer writes the checkpoint; this adds where the data came from, so
    # the console can say what a version was built out of and the release panel
    # can show it beside the reported models.
    blob = torch.load(target, map_location="cpu", weights_only=False)

    # The trainer's metrics dataclass calls them auc_roc and f1_score; the
    # release panel, the public model list and the paper's own tables call
    # them auroc and f1. Both names are written, so one checkpoint reads
    # correctly everywhere instead of showing a dash in half the places.
    scores = dict(blob.get("metrics") or {})
    for mine, theirs in (("auc_roc", "auroc"), ("f1_score", "f1"),
                         ("sensitivity", "recall")):
        if isinstance(scores.get(mine), (int, float)):
            scores.setdefault(theirs, scores[mine])
    blob["metrics"] = scores

    blob["provenance"] = {
        "lineage": LINEAGE,
        "architecture": ARCHITECTURE,
        "trained_from": "imagenet-initialised backbones (not a fine-tune)",
        "manifest": manifest.name,
        "bundle": args.bundle.name if args.bundle else None,
        "reviewed_cases": len(reviewed),
        "reviewed_normal": counts[0],
        "reviewed_pneumonia": counts[1],
        "epochs": args.epochs,
        "finetune_epochs": args.finetune_epochs,
        "minutes": round(minutes, 1),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(blob, target)

    print(f"\nSaved {target}")
    print(f"  accuracy {scores.get('accuracy', float('nan')):.4f}   "
          f"AUROC {scores.get('auc_roc', scores.get('auroc', float('nan'))):.4f}")
    print(f"  trained on {len(reviewed)} reviewed cases plus the public dataset, "
          f"in {minutes:.0f} minutes")
    print("\nNothing has been released. Upload this file in the administrator "
          "console under “The community model”, read its scores beside the "
          "reported ones, and release it only if it earns it.")


if __name__ == "__main__":
    main()
