"""
Hybrid DenseNet-Swin Transformer Architecture with CBAM Attention.
Core model of Rad-Intel for automated pneumonia detection.
"""

import torch
import torch.nn as nn
from rad_intel.models.cbam import CBAM
from rad_intel.models.densenet import DenseNetExtractor
from rad_intel.models.swin import SwinExtractor


class HybridDenseNetSwinCBAM(nn.Module):
    """
    Hybrid Architecture fusing DenseNet121 local feature maps with Swin Transformer
    global context tokens, refined via CBAM (Convolutional Block Attention Module).

    Workflow:
      1. DenseNet branch: (B, 3, 224, 224) -> (B, 1024, 7, 7)
      2. Swin branch:     (B, 3, 224, 224) -> (B, 768, 7, 7)
      3. Feature Fusion:  Concat -> (B, 1792, 7, 7) -> Conv1x1 -> (B, 512, 7, 7)
      4. Attention:       CBAM(512) -> (B, 512, 7, 7) (Optional for ablation)
      5. Classification:  AdaptiveAvgPool2d(1) -> Dropout -> Linear(512, num_classes)
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = False,
        use_cbam: bool = True,
        fusion_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_cbam = use_cbam
        self.fusion_dim = fusion_dim

        # Branch extractors
        self.densenet_branch = DenseNetExtractor(pretrained=pretrained)
        self.swin_branch = SwinExtractor(pretrained=pretrained)

        total_in_channels = self.densenet_branch.out_channels + self.swin_branch.out_channels

        # Fusion Projection Layer (1x1 Conv + BN + ReLU)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_in_channels, fusion_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
        )

        # Attention refinement
        if self.use_cbam:
            self.cbam = CBAM(in_planes=fusion_dim, ratio=16, kernel_size=7)
        else:
            self.cbam = nn.Identity()

        # Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract fused and attention-refined feature map."""
        local_feat = self.densenet_branch(x)    # (B, 1024, 7, 7)
        global_feat = self.swin_branch(x)       # (B, 768, 7, 7)

        # Concatenate along channel dimension
        fused = torch.cat([local_feat, global_feat], dim=1)  # (B, 1792, 7, 7)
        projected = self.fusion_conv(fused)                  # (B, 512, 7, 7)

        if self.use_cbam:
            refined = self.cbam(projected)                  # (B, 512, 7, 7)
        else:
            refined = projected

        return refined

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        pooled = self.global_pool(feat).flatten(1)          # (B, 512)
        logits = self.classifier(pooled)                     # (B, num_classes)
        return logits

    def get_target_gradcam_layer(self) -> nn.Module:
        """Returns the optimal target layer for Grad-CAM gradient backpropagation."""
        if self.use_cbam:
            return self.cbam.spatial_att.conv
        return self.fusion_conv[0]
