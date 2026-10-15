"""
Evaluate every trained checkpoint on the held-out test partition and emit the
numbers Table IV, Table V and Figure 2 need.

Runs each model once over the official 624-image test folder, caches the
per-image probabilities, then derives:

  * the full metric row at the checkpoint's own operating point
  * paired McNemar tests between configurations on identical images
  * DeLong-style paired bootstrap confidence intervals on AUROC differences
  * ROC and calibration curves, written as CSV for plotting

Paired tests need per-image predictions rather than summary statistics, which
is why they live here and not in the training script. They are valid from a
single run: McNemar compares two models on the same images, so it answers
"do these two disagree more than chance?" without needing repeated seeds.

Usage:
    python scripts/evaluate_all.py --data-dir path/to/chest_xray
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rad_intel.config import settings
from rad_intel.models.factory import create_model
from rad_intel.training.dataset import CXRDataset, get_transforms, load_dataset_splits
from rad_intel.training.metrics import compute_metrics

MODELS = ["densenet121", "swin_t", "resnet50", "hybrid_no_cbam", "hybrid"]
LABELS = {
    "densenet121": "DenseNet121",
    "swin_t": "Swin-T",
    "resnet50": "ResNet50",
    "hybrid_no_cbam": "Fusion without CBAM",
    "hybrid": "Complete classifier",
}


def predict_test_set(model_name, weights_path, test_ds, device, batch_size=16):
    """Per-image P(pneumonia) for one checkpoint, plus its stored threshold."""
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    stored_type = ckpt.get("model_type") if isinstance(ckpt, dict) else None
    if stored_type and stored_type != model_name:
        raise SystemExit(
            f"{weights_path.name} holds a '{stored_type}' model but '{model_name}' "
            f"was requested. Refusing to evaluate mismatched weights."
        )
    threshold = float(ckpt.get("optimal_threshold", 0.5)) if isinstance(ckpt, dict) else 0.5

    model = create_model(model_name, pretrained=False, weights_path=str(weights_path), device=device)
    model.eval()

    probs, labels = [], []
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for images, lbls in loader:
            out = F.softmax(model(images.to(device)), dim=1)[:, 1]
            probs.extend(out.cpu().tolist())
            labels.extend(lbls.tolist())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.array(labels), np.array(probs), threshold


def expected_calibration_error(y, p, bins=15):
    """ECE with equal-width bins, as reported in Table IV."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def reliability_curve(y, p, bins=15):
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum() == 0:
            continue
        rows.append(((lo + hi) / 2, float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows


def mcnemar(y, pa, ta, pb, tb):
    """
    Exact McNemar on the discordant pairs of two thresholded classifiers.

    b = A right / B wrong, c = A wrong / B right. Under the null these split
    evenly, so p is a two-sided binomial tail on min(b, c).
    """
    from scipy.stats import binomtest

    a_ok = (pa >= ta).astype(int) == y
    b_ok = (pb >= tb).astype(int) == y
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(min(b, c), b + c, 0.5).pvalue * 1.0 if b + c > 0 else 1.0)


def auroc(y, p):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def paired_bootstrap_auroc(y, pa, pb, n=2000, seed=42):
    """Bootstrap CI on AUROC(A) - AUROC(B), resampling images in pairs."""
    rng = np.random.default_rng(seed)
    diffs = []
    idx = np.arange(len(y))
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        diffs.append(auroc(y[s], pa[s]) - auroc(y[s], pb[s]))
    d = np.array(diffs)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def holm(pvals):
    """Holm step-down adjustment, order preserved."""
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser(description="Evaluate all checkpoints on the test partition")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--weights-dir", default="weights")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = settings.torch_device
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    _, _, test_samples, _ = load_dataset_splits(data_dir=args.data_dir)
    test_ds = CXRDataset(test_samples, transform=get_transforms(is_train=False))
    print(f"Test images: {len(test_ds)} | device: {device}\n")

    preds, rows = {}, []
    for name in MODELS:
        wp = Path(args.weights_dir) / f"best_{name}_model.pt"
        if not wp.exists():
            print(f"[skip] {name}: no checkpoint at {wp}")
            continue
        print(f"[run ] {name} ...", flush=True)
        y, p, thr = predict_test_set(name, wp, test_ds, device, args.batch_size)
        preds[name] = (y, p, thr)

        m = compute_metrics(y, p, threshold=thr)
        from sklearn.metrics import average_precision_score, brier_score_loss

        rows.append({
            "model": LABELS[name],
            "key": name,
            "threshold": round(thr, 4),
            "accuracy": round(float(m.accuracy), 4),
            "auroc": round(auroc(y, p), 4),
            "auprc": round(float(average_precision_score(y, p)), 4),
            "sensitivity": round(float(m.sensitivity), 4),
            "specificity": round(float(m.specificity), 4),
            "precision": round(float(m.precision), 4),
            "f1": round(float(m.f1_score), 4),
            "balanced_accuracy": round((float(m.sensitivity) + float(m.specificity)) / 2, 4),
            "brier": round(float(brier_score_loss(y, p)), 4),
            "ece": round(expected_calibration_error(y, p), 4),
        })
        np.savetxt(out / f"probs_{name}.csv",
                   np.column_stack([y, p]), delimiter=",",
                   header="label,p_pneumonia", comments="")
        r = rows[-1]
        print(f"       acc {r['accuracy']:.4f}  auroc {r['auroc']:.4f}  "
              f"auprc {r['auprc']:.4f}  ece {r['ece']:.4f}  brier {r['brier']:.4f}")

    if not rows:
        raise SystemExit("No checkpoints found.")

    # ---- Table IV ----
    (out / "table4_metrics.json").write_text(json.dumps(rows, indent=2))
    print("\n=== TABLE IV ===")
    hdr = f"{'Configuration':<22}{'AUROC':>8}{'AUPRC':>8}{'Sens':>8}{'Spec':>8}{'F1':>8}{'BalAcc':>8}{'ECE':>8}"
    print(hdr)
    for r in rows:
        print(f"{r['model']:<22}{r['auroc']:>8.3f}{r['auprc']:>8.3f}{r['sensitivity']:>8.3f}"
              f"{r['specificity']:>8.3f}{r['f1']:>8.3f}{r['balanced_accuracy']:>8.3f}{r['ece']:>8.3f}")

    # ---- Table V: paired comparisons against the complete classifier ----
    if "hybrid" in preds:
        y, ph, th = preds["hybrid"]
        comps, raw_p = [], []
        for name in MODELS:
            if name == "hybrid" or name not in preds:
                continue
            _, pb, tb = preds[name]
            b, c, pval = mcnemar(y, ph, th, pb, tb)
            d, lo, hi = paired_bootstrap_auroc(y, ph, pb)
            comps.append({"vs": LABELS[name], "mcnemar_b": b, "mcnemar_c": c,
                          "mcnemar_p": round(pval, 4),
                          "delta_auroc": round(d, 4),
                          "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
            raw_p.append(pval)
        for comp, adj in zip(comps, holm(np.array(raw_p))):
            comp["mcnemar_p_holm"] = round(float(adj), 4)
        (out / "table5_comparisons.json").write_text(json.dumps(comps, indent=2))
        print("\n=== TABLE V (complete classifier vs each) ===")
        print(f"{'Comparison':<24}{'dAUROC [95% CI]':>26}{'McNemar p':>12}{'Holm p':>10}")
        for c in comps:
            ci = f"{c['delta_auroc']:+.3f} [{c['ci_low']:+.3f},{c['ci_high']:+.3f}]"
            print(f"vs {c['vs']:<21}{ci:>26}{c['mcnemar_p']:>12.4f}{c['mcnemar_p_holm']:>10.4f}")
        print("\n(b/c = images the complete classifier gets right and the other wrong, "
              "and vice versa)")
        for c in comps:
            print(f"  vs {c['vs']}: b={c['mcnemar_b']}, c={c['mcnemar_c']}")

    # ---- Figure 2 data ----
    from sklearn.metrics import roc_curve

    for name, (y, p, _) in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        np.savetxt(out / f"roc_{name}.csv", np.column_stack([fpr, tpr]),
                   delimiter=",", header="fpr,tpr", comments="")
        with open(out / f"reliability_{name}.csv", "w") as f:
            f.write("bin_center,mean_predicted,observed_frequency,count\n")
            for row in reliability_curve(y, p):
                f.write(",".join(str(x) for x in row) + "\n")

    print(f"\nWrote metrics, ROC and reliability data to {out}/")


if __name__ == "__main__":
    main()
