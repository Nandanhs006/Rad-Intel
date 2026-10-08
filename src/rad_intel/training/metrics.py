"""
Comprehensive clinical evaluation metrics for pneumonia classification.
Computes Accuracy, AUC-ROC, Sensitivity (Recall), Specificity, F1-Score,
Confusion Matrix, and optimal decision threshold via Youden's Index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvaluationMetrics:
    accuracy: float
    auc_roc: float
    sensitivity: float  # True Positive Rate (Recall for Pneumonia)
    specificity: float  # True Negative Rate (Recall for Normal)
    precision: float
    f1_score: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    optimal_threshold: float
    total_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "accuracy_percent": f"{self.accuracy * 100:.2f}%",
            "auc_roc": round(self.auc_roc, 4),
            "sensitivity": round(self.sensitivity, 4),
            "sensitivity_percent": f"{self.sensitivity * 100:.2f}%",
            "specificity": round(self.specificity, 4),
            "specificity_percent": f"{self.specificity * 100:.2f}%",
            "precision": round(self.precision, 4),
            "f1_score": round(self.f1_score, 4),
            "confusion_matrix": {
                "TP": self.true_positives,
                "FP": self.false_positives,
                "TN": self.true_negatives,
                "FN": self.false_negatives,
            },
            "optimal_threshold": round(self.optimal_threshold, 4),
            "total_samples": self.total_samples,
        }

    def summary_table(self, title: str = "Clinical Evaluation Summary") -> str:
        lines = [
            f"=== {title} ===",
            f" Total Evaluated: {self.total_samples} radiographs",
            f" Test Accuracy  : {self.accuracy * 100:.2f}%",
            f" AUC-ROC        : {self.auc_roc:.4f}",
            f" Sensitivity    : {self.sensitivity * 100:.2f}% (Pneumonia Detection Rate)",
            f" Specificity    : {self.specificity * 100:.2f}% (Normal Specificity)",
            f" F1-Score       : {self.f1_score:.4f}",
            f" Precision      : {self.precision * 100:.2f}%",
            f" Decision Thresh: {self.optimal_threshold:.4f}",
            "--- Confusion Matrix ---",
            f" [TN: {self.true_negatives:>4} | FP: {self.false_positives:>4}]",
            f" [FN: {self.false_negatives:>4} | TP: {self.true_positives:>4}]",
            "===============================",
        ]
        return "\n".join(lines)


def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """
    Computes optimal decision threshold using Youden's J statistic:
    J = Sensitivity + Specificity - 1 = TPR - FPR
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_thresh = float(thresholds[best_idx])
    # Clamp within sensible probability bounds
    return max(0.1, min(0.9, best_thresh))


def compute_metrics(
    y_true: np.ndarray | torch.Tensor | list[int],
    y_probs: np.ndarray | torch.Tensor | list[float],
    threshold: float | None = None,
) -> EvaluationMetrics:
    """
    Evaluates binary classification predictions.

    Args:
        y_true: Ground truth binary labels (0=Normal, 1=Pneumonia).
        y_probs: Predicted probability of positive class (Pneumonia).
        threshold: Decision threshold for positive classification. If None,
                   calculates optimal threshold via Youden's index.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_probs, torch.Tensor):
        y_probs = y_probs.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)

    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_probs)

    y_pred = (y_probs >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))

    try:
        auc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        auc = 0.5

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    sens = float(recall_score(y_true, y_pred, zero_division=0))  # TPR / Recall

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return EvaluationMetrics(
        accuracy=acc,
        auc_roc=auc,
        sensitivity=sens,
        specificity=spec,
        precision=prec,
        f1_score=f1,
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        optimal_threshold=float(threshold),
        total_samples=len(y_true),
    )
