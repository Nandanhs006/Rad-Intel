"""
Loss functions for Rad-Intel deep learning training.
Includes Focal Loss and class-weighted Cross-Entropy to address pneumonia class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme foreground-background / class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: torch.Tensor | list[float] | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C) raw unnormalized logits.
            targets: (B,) ground truth class indices.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # probability of true class
        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def get_loss_function(
    loss_type: str = "focal",
    class_weights: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> nn.Module:
    """Factory for loss functions."""
    if loss_type.lower() == "focal":
        return FocalLoss(alpha=class_weights, gamma=gamma)
    if loss_type.lower() == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    return nn.CrossEntropyLoss()
