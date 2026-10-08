r"""
Evaluation script for Rad-Intel model checkpoints on the Kaggle Chest X-Ray test set.
Evaluates all 624 test images and outputs comprehensive clinical metrics.

Usage:
  .\.venv\Scripts\python.exe scripts/evaluate.py
  .\.venv\Scripts\python.exe scripts/evaluate.py --checkpoint weights/best_hybrid_model.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rad_intel.models.factory import create_model
from rad_intel.training.dataset import CXRDataset, get_transforms, load_dataset_splits
from rad_intel.training.metrics import EvaluationMetrics, compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Rad-Intel Model on Chest X-Ray Test Set"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weights/best_hybrid_model.pt",
        help="Path to trained PyTorch checkpoint (.pt)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="hybrid",
        help="Model architecture type (hybrid, densenet121, swin_t, resnet50)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to Kaggle chest_xray directory (auto-detected if omitted)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for test evaluation (default: 32)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="evaluation_report.json",
        help="Path to save evaluation metrics JSON report",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Running evaluation on device: {device}")

    # 1. Load test samples
    _, _, test_samples, _ = load_dataset_splits(data_dir=args.data_dir)
    print(f"[EVAL] Loaded {len(test_samples)} test images (234 Normal, 390 Pneumonia).")

    # 2. Build model and load weights
    ckpt_path = Path(args.checkpoint)
    saved_threshold = None

    if ckpt_path.exists():
        print(f"[EVAL] Loading trained weights from: {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_type = checkpoint.get("model_type", args.model_type)
        model = create_model(model_type, pretrained=False).to(device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            saved_threshold = checkpoint.get("optimal_threshold")
        else:
            model.load_state_dict(checkpoint)
        print(f"[EVAL] Successfully loaded weights (trained model type: {model_type}).")
    else:
        print(
            f"[WARNING] Checkpoint {ckpt_path} not found! Evaluating model with pretrained ImageNet weights."
        )
        model = create_model(args.model_type, pretrained=True).to(device)

    model.eval()

    # 3. Create DataLoader
    test_ds = CXRDataset(test_samples, transform=get_transforms(is_train=False))
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 4. Inference loop
    all_probs: list[float] = []
    all_y: list[int] = []
    t0 = time.time()

    print("[EVAL] Running inference across all test radiographs...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_y.extend(labels.tolist())

            if (batch_idx + 1) % 5 == 0 or len(all_y) == len(test_samples):
                print(f"  Processed {len(all_y)}/{len(test_samples)} radiographs...")

    inference_time = time.time() - t0
    print(f"[EVAL] Completed test inference in {inference_time:.2f}s ({inference_time/len(test_samples)*1000:.1f}ms/image).")

    # 5. Compute clinical metrics
    metrics = compute_metrics(all_y, all_probs, threshold=saved_threshold)
    print("\n" + metrics.summary_table("Rad-Intel Test Set Clinical Evaluation"))

    # 6. Save JSON report
    report_path = Path(args.output_json)
    with open(report_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"\n[SUCCESS] Detailed metrics exported to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
