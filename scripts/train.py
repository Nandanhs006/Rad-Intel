r"""
Rad-Intel Deep Learning Model Training Script.
Trains DenseNet121 or Hybrid DenseNet-Swin-CBAM on the Kaggle Chest X-Ray dataset
targeting 90%+ test accuracy.

Usage:
  .\.venv\Scripts\python.exe scripts/train.py --model densenet121
  .\.venv\Scripts\python.exe scripts/train.py --model hybrid --epochs 20 --loss focal
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rad_intel.training.trainer import HybridTrainer, TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Rad-Intel Deep Learning Model for Automated Pneumonia Detection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="densenet121",
        choices=["densenet121", "swin_t", "resnet50", "hybrid_no_cbam", "hybrid"],
        help="Model architecture to train (default: densenet121 for fast CPU transfer learning, hybrid for dual-branch attention)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs for training attention & classifier head (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and feature caching (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="focal",
        choices=["focal", "weighted_ce"],
        help="Loss function type (default: focal)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.0,
        help="Gamma focusing parameter for Focal Loss (default: 2.0)",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional limit on training samples",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional limit on validation samples",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional limit on test samples",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=10,
        help="Phase 2 end-to-end fine-tuning epochs; 0 keeps backbones frozen (default: 10)",
    )
    parser.add_argument(
        "--finetune-batch-size",
        type=int,
        default=12,
        help="Batch size for phase-2 end-to-end fine-tuning (default: 12; lower if CUDA OOM)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to Kaggle chest_xray directory (auto-detected if omitted)",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="Filename for the saved best model checkpoint (defaults to best_<model>_model.pt)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = TrainingConfig(
        model_type=args.model,
        epochs=args.epochs,
        finetune_epochs=args.finetune_epochs,
        finetune_batch_size=args.finetune_batch_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        loss_type=args.loss,
        focal_gamma=args.gamma,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        data_dir=args.data_dir,
        checkpoint_name=args.checkpoint_name,
    )

    trainer = HybridTrainer(config=config)
    t0 = time.time()
    model, test_metrics = trainer.train_cached()
    total_time = time.time() - t0

    print(f"\n[DONE] Training and test evaluation completed in {total_time/60:.1f} minutes.")
    print(f"Final Test Accuracy: {test_metrics.accuracy * 100:.2f}%")
    print(f"Final AUC-ROC:       {test_metrics.auc_roc:.4f}")
    print(f"Final Sensitivity:   {test_metrics.sensitivity * 100:.2f}%")
    print(f"Final Specificity:   {test_metrics.specificity * 100:.2f}%")


if __name__ == "__main__":
    main()
