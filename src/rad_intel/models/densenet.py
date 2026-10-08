"""
DenseNet121 local feature extractor and baseline model.
Reference:
Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017).
Densely Connected Convolutional Networks. CVPR, pp. 4700-4708.
"""

import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


class DenseNetExtractor(nn.Module):
    """
    DenseNet121 feature extractor for extracting local fine-grained representations.
    Exposes feature maps of shape (B, 1024, 7, 7) for 224x224 input.
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        base = densenet121(weights=weights)
        self.features = base.features
        self.relu = nn.ReLU(inplace=True)
        self.out_channels = 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = self.relu(features)
        return out


class DenseNet121Baseline(nn.Module):
    """
    DenseNet121 standalone classifier for ablation and benchmark comparison.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = False, dropout: float = 0.2):
        super().__init__()
        self.extractor = DenseNetExtractor(pretrained=pretrained)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.extractor.out_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extractor(x)
        pooled = self.pool(feat).flatten(1)
        logits = self.classifier(pooled)
        return logits
