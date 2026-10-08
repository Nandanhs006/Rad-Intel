"""
Swin Transformer global feature extractor and baseline model.
Reference:
Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021).
Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. ICCV, pp. 9992-10002.
"""

import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights


class SwinExtractor(nn.Module):
    """
    Swin Transformer (Tiny) global feature extractor.
    Extracts high-level contextual spatial tokens and formats them into
    (B, 768, 7, 7) spatial feature maps for 224x224 input.
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = Swin_T_Weights.DEFAULT if pretrained else None
        base = swin_t(weights=weights)
        self.features = base.features
        self.norm = base.norm
        self.out_channels = 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # features(x) outputs (B, H, W, C) = (B, 7, 7, 768)
        feat = self.features(x)
        feat = self.norm(feat)
        # Permute to (B, C, H, W) = (B, 768, 7, 7) for convolutional fusion
        feat = feat.permute(0, 3, 1, 2).contiguous()
        return feat


class SwinTransformerBaseline(nn.Module):
    """
    Swin Transformer standalone classifier for ablation and benchmark comparison.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = False, dropout: float = 0.2):
        super().__init__()
        self.extractor = SwinExtractor(pretrained=pretrained)
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
