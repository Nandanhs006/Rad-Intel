"""
Model factory and registry for Rad-Intel.
Provides unified model instantiation, checkpoint loading, and target layer resolution.
"""

from typing import Literal
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from rad_intel.models.hybrid import HybridDenseNetSwinCBAM
from rad_intel.models.densenet import DenseNet121Baseline
from rad_intel.models.swin import SwinTransformerBaseline

ModelType = Literal[
    "hybrid", "hybrid_no_cbam", "densenet121", "swin_t", "resnet50"
]


class ResNet50Baseline(nn.Module):
    """ResNet50 baseline model for comparison benchmarks."""

    def __init__(self, num_classes: int = 2, pretrained: bool = False, dropout: float = 0.2):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = resnet50(weights=weights)
        in_features = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_target_gradcam_layer(self) -> nn.Module:
        return self.model.layer4[-1].conv3


def create_model(
    model_name: ModelType = "hybrid",
    num_classes: int = 2,
    pretrained: bool = False,
    weights_path: str | Path | None = None,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """
    Factory function to instantiate models and optionally load custom weights.
    """
    name = model_name.lower().strip()

    if name == "hybrid":
        model = HybridDenseNetSwinCBAM(num_classes=num_classes, pretrained=pretrained, use_cbam=True)
    elif name == "hybrid_no_cbam":
        model = HybridDenseNetSwinCBAM(num_classes=num_classes, pretrained=pretrained, use_cbam=False)
    elif name == "densenet121":
        model = DenseNet121Baseline(num_classes=num_classes, pretrained=pretrained)
    elif name == "swin_t":
        model = SwinTransformerBaseline(num_classes=num_classes, pretrained=pretrained)
    elif name == "resnet50":
        model = ResNet50Baseline(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Available: hybrid, hybrid_no_cbam, densenet121, swin_t, resnet50")

    if weights_path:
        w_path = Path(weights_path)
        if w_path.exists() and w_path.is_file():
            try:
                checkpoint = torch.load(w_path, map_location=device, weights_only=False)
            except Exception:
                checkpoint = torch.load(w_path, map_location=device)

            if isinstance(checkpoint, dict):
                ckpt_model_type = checkpoint.get("model_type")
                if ckpt_model_type and ckpt_model_type != name:
                    # Model type mismatch, avoid corrupted partial loading
                    return model
                state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            else:
                state_dict = checkpoint

            # Remove possible 'module.' prefix from DDP
            clean_state_dict = {
                (k[7:] if k.startswith("module.") else k): v
                for k, v in state_dict.items()
            }
            model.load_state_dict(clean_state_dict, strict=False)

    model.to(device)
    model.eval()
    return model


def get_model_target_layer(model: nn.Module) -> nn.Module:
    """Resolve the appropriate convolutional/attention target layer for Grad-CAM."""
    if hasattr(model, "get_target_gradcam_layer"):
        return model.get_target_gradcam_layer()
    if isinstance(model, HybridDenseNetSwinCBAM):
        return model.get_target_gradcam_layer()
    if isinstance(model, DenseNet121Baseline):
        # The last conv layer in DenseNet121 features
        convs = [m for m in model.extractor.features.modules() if isinstance(m, nn.Conv2d)]
        if convs:
            return convs[-1]
    if isinstance(model, SwinTransformerBaseline):
        # The last norm or layer in Swin features
        return model.extractor.norm
    if isinstance(model, ResNet50Baseline):
        return model.get_target_gradcam_layer()

    # Fallback to the last convolutional layer or parameter module
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if conv_layers:
        return conv_layers[-1]
    norm_layers = [m for m in model.modules() if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d))]
    if norm_layers:
        return norm_layers[-1]
    raise ValueError("Unable to determine target layer for Grad-CAM.")
