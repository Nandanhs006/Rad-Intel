"""
Fine-tune a model on clinician-reviewed cases, offline and versioned.

This is the step that turns the review queue into learning. It is deliberately
a script and not a button: training from data a reviewer supplied is a
deliberate act with a result that has to be inspected before anyone relies
on it.

What it guarantees:

* The original Kaggle test partition is never trained on. It is the only
  yardstick that makes a new model comparable to the paper's numbers, so it
  stays untouched and is used solely to score the result.
* The original checkpoint is never overwritten. Output goes to a new
  versioned file, and the previous version stays exactly where it was.
* Reviewed cases are split into their own train and validation sets. They are
  never merged into the dataset split the paper reports.
* Nothing is promoted automatically. The script prints a before-and-after
  comparison on the held-out partition and stops. Selecting the new version
  for serving is a separate, explicit act.
* Preprocessing is the same single path the service and the evaluation use,
  imported rather than reimplemented.

Usage
-----
    python scripts/finetune_from_feedback.py \\
        --manifest feedback_store/exports/reviewed-export-20261013T235813Z.jsonl \\
        --base-model hybrid

    # inspect the printed comparison, then if it is an improvement:
    python scripts/finetune_from_feedback.py --promote weights/best_hybrid_v2_model.pt
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader

from rad_intel.config import settings
from rad_intel.models.factory import create_model
from rad_intel.training.dataset import CXRDataset, get_transforms, load_dataset_splits
from rad_intel.training.losses import get_loss_function

LABELS = {"NORMAL": 0, "PNEUMONIA": 1}


# --------------------------------------------------------------------------- data

def load_manifest(path: Path) -> list[tuple[Path, int]]:
    """Read an export manifest into (image path, label) pairs."""
    if not path.exists():
        sys.exit(f"Manifest not found: {path}")

    samples, skipped = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        img = settings.BASE_DIR / row["image_path"]
        label = LABELS.get((row.get("label") or "").upper())
        if label is None or not img.exists():
            skipped += 1
            continue
        samples.append((img, label))

    if skipped:
        print(f"  skipped {skipped} entries with a missing image or unusable label")
    return samples


def split_reviewed(samples, val_fraction=0.25, seed=42):
    """
    Hold out a slice of the reviewed cases for validation.

    Stratified so that a tiny review batch does not end up with a validation
    set containing only one class.
    """
    rng = np.random.default_rng(seed)
    train, val = [], []
    for cls in (0, 1):
        members = [s for s in samples if s[1] == cls]
        rng.shuffle(members)
        n_val = max(1, int(round(len(members) * val_fraction))) if len(members) > 1 else 0
        val.extend(members[:n_val])
        train.extend(members[n_val:])
    return train, val


# ---------------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate(model, loader, device, threshold):
    """Score a model on a loader. Returns accuracy, AUROC, sensitivity, specificity."""
    model.eval()
    probs, labels = [], []
    for images, targets in loader:
        p = F.softmax(model(images.to(device)), dim=-1)[:, 1]
        probs.extend(p.cpu().tolist())
        labels.extend(targets.tolist())

    probs, labels = np.asarray(probs), np.asarray(labels)
    pred = (probs >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    # The same measures the project reports for its own models, so a version
    # released from reviewed cases can be read beside them rather than being
    # a number nobody can place.
    precision = tp / max(tp + fp, 1)
    sensitivity = tp / max(tp + fn, 1)
    return {
        "accuracy": (tp + tn) / max(len(labels), 1),
        "auroc": float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else float("nan"),
        "precision": precision,
        "sensitivity": sensitivity,
        "recall": sensitivity,
        "specificity": tn / max(tn + fp, 1),
        "f1": 2 * precision * sensitivity / max(precision + sensitivity, 1e-9),
        "confusion": f"{tn}/{fp}/{fn}/{tp}",
    }


def show(tag: str, m: dict) -> None:
    print(f"  {tag:<22s} acc {m['accuracy']:.4f}  auroc {m['auroc']:.4f}  "
          f"prec {m['precision']:.4f}  sens {m['sensitivity']:.4f}  "
          f"spec {m['specificity']:.4f}  F1 {m['f1']:.4f}  "
          f"TN/FP/FN/TP {m['confusion']}")


def _find_dataset() -> str | None:
    """
    Locate the Kaggle dataset without requiring --data-dir.

    The shared loader only probes its own conventional paths, and this
    repository keeps the dataset beside the paper, so the usual locations are
    checked here rather than changing a module the evaluation also uses.
    """
    for candidate in (
        settings.BASE_DIR / "Rad-Intel paper" / "chest_xray",
        settings.BASE_DIR / "chest_xray",
        settings.BASE_DIR / "data" / "chest_xray",
    ):
        if (candidate / "test").is_dir():
            return str(candidate)
    return None


# ----------------------------------------------------------------------- promotion

def promote(checkpoint: Path) -> None:
    """Make a reviewed version the served model, keeping the previous one."""
    if not checkpoint.exists():
        sys.exit(f"Checkpoint not found: {checkpoint}")
    meta = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_type = meta.get("model_type")
    if not model_type:
        sys.exit("Checkpoint has no model_type; refusing to promote.")

    live = settings.BASE_DIR / "weights" / f"best_{model_type}_model.pt"
    if live.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = live.with_name(f"{live.stem}_superseded_{stamp}.pt")
        shutil.copy2(live, backup)
        print(f"Previous model preserved at {backup.relative_to(settings.BASE_DIR)}")

    shutil.copy2(checkpoint, live)
    print(f"Promoted {checkpoint.name} to {live.relative_to(settings.BASE_DIR)}")
    print("Restart the service for the change to take effect.")


# ---------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, help="Export manifest produced by the review queue")
    ap.add_argument("--base-model", default="hybrid", help="Model version to fine-tune from")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr-backbone", type=float, default=5e-6)
    ap.add_argument("--lr-head", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--data-dir", default=None, help="Kaggle dataset root, for evaluation only")
    ap.add_argument("--promote", type=Path, help="Promote this checkpoint and exit")
    args = ap.parse_args()

    if args.promote:
        return promote(args.promote)
    if not args.manifest:
        ap.error("--manifest is required unless --promote is used")

    device = settings.torch_device
    print(f"Device: {device}")

    # --- reviewed cases ----------------------------------------------------
    print(f"\nManifest: {args.manifest}")
    reviewed = load_manifest(args.manifest)
    if len(reviewed) < 4:
        sys.exit(f"Only {len(reviewed)} usable reviewed cases. Too few to fine-tune "
                 f"on; collect more reviews first.")
    counts = {c: sum(1 for _, l in reviewed if l == c) for c in (0, 1)}
    print(f"  reviewed cases: {len(reviewed)}  (normal {counts[0]}, pneumonia {counts[1]})")
    if min(counts.values()) == 0:
        sys.exit("Reviewed cases contain only one class. Fine-tuning on this would "
                 "collapse the classifier; collect examples of both.")

    train_s, val_s = split_reviewed(reviewed)
    print(f"  fine-tune train {len(train_s)} / validation {len(val_s)}")

    # --- the untouched yardstick -------------------------------------------
    # The Kaggle test partition is loaded for scoring only. It is never put in
    # a training loader, which is what keeps a new version comparable to the
    # numbers the paper reports.
    data_dir = args.data_dir or _find_dataset()
    _, _, test_samples, class_weights = load_dataset_splits(data_dir=data_dir)
    print(f"  held-out Kaggle test partition: {len(test_samples)} images (never trained on)")

    eval_tf = get_transforms(is_train=False)
    train_tf = get_transforms(is_train=True)
    test_loader = DataLoader(CXRDataset(test_samples, transform=eval_tf),
                             batch_size=16, shuffle=False)
    train_loader = DataLoader(CXRDataset(train_s, transform=train_tf),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(CXRDataset(val_s, transform=eval_tf),
                            batch_size=args.batch_size, shuffle=False) if val_s else None

    # --- baseline -----------------------------------------------------------
    base_ckpt = settings.BASE_DIR / "weights" / f"best_{args.base_model}_model.pt"
    if not base_ckpt.exists():
        sys.exit(f"Base checkpoint not found: {base_ckpt}")
    meta = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    threshold = float(meta.get("optimal_threshold", 0.5))

    model = create_model(model_name=args.base_model, num_classes=settings.NUM_CLASSES,
                         pretrained=True, weights_path=str(base_ckpt), device=device)

    print(f"\nBaseline on the held-out partition (threshold {threshold:.4f}):")
    before = evaluate(model, test_loader, device, threshold)
    show("before fine-tuning", before)

    # --- fine-tune ----------------------------------------------------------
    # Low learning rates and few epochs: the aim is to absorb a handful of
    # corrected cases, not to retrain the network.
    backbone_markers = ("densenet", "swin", "extractor", "model.conv1", "model.bn1", "model.layer")
    backbone, head = [], []
    for name, param in model.named_parameters():
        param.requires_grad = True
        (backbone if any(m in name for m in backbone_markers) else head).append(param)

    optimizer = AdamW([
        {"params": backbone, "lr": args.lr_backbone},
        {"params": head, "lr": args.lr_head},
    ], weight_decay=1e-4)
    criterion = get_loss_function(loss_type="focal",
                                  class_weights=class_weights.to(device), gamma=2.0)

    print(f"\nFine-tuning {args.epochs} epochs "
          f"(backbone lr {args.lr_backbone}, head lr {args.lr_head})")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            total += loss.item() * images.size(0)
            seen += images.size(0)
        msg = f"  epoch {epoch}/{args.epochs}  train loss {total / max(seen, 1):.4f}"
        if val_loader:
            v = evaluate(model, val_loader, device, threshold)
            msg += f"  reviewed-val acc {v['accuracy']:.3f}"
        print(msg, flush=True)

    # --- score and save -----------------------------------------------------
    print(f"\nAfter fine-tuning, same held-out partition:")
    after = evaluate(model, test_loader, device, threshold)
    show("after fine-tuning", after)

    d_acc = after["accuracy"] - before["accuracy"]
    d_auc = after["auroc"] - before["auroc"]
    print(f"\n  change: accuracy {d_acc:+.4f}   AUROC {d_auc:+.4f}")

    # A tolerance band, because a few thousandths on 624 images is sampling
    # noise rather than a real difference, and calling it either way would
    # overstate what this comparison can resolve.
    TOL = 0.002
    if d_acc < -TOL or d_auc < -TOL:
        print("  Worse on the held-out partition. Do not promote.")
    elif d_acc > TOL or d_auc > TOL:
        print("  Improved on the held-out partition. Inspect the cases, then promote if sound.")
    else:
        print("  Indistinguishable on the held-out partition at this sample size; "
              "the change is within noise. Promoting buys nothing measurable.")

    weights_dir = settings.BASE_DIR / "weights"
    n = 2
    while (weights_dir / f"best_{args.base_model}_v{n}_model.pt").exists():
        n += 1
    out = weights_dir / f"best_{args.base_model}_v{n}_model.pt"

    torch.save({
        "model_type": args.base_model,
        "optimal_threshold": threshold,
        "class_weights": class_weights.tolist(),
        "model_state_dict": model.state_dict(),
        "provenance": {
            "base_checkpoint": base_ckpt.name,
            "manifest": str(args.manifest),
            "reviewed_cases": len(reviewed),
            "epochs": args.epochs,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "metrics_before": before,
        "metrics_after": after,
    }, out)

    print(f"\nSaved {out.relative_to(settings.BASE_DIR)}")
    print(f"The original {base_ckpt.name} is unchanged.")
    print("Nothing has been promoted. To serve this version:")
    print(f"  python scripts/finetune_from_feedback.py --promote {out.relative_to(settings.BASE_DIR)}")


if __name__ == "__main__":
    main()
