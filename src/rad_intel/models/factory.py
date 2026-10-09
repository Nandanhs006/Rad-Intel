"""
Model factory and registry for Rad-Intel.
Provides unified model instantiation, checkpoint loading, and target layer resolution.
"""

from typing import Literal
from pathlib import Path
import warnings
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
        # Whole layer4 block, not layer4[-1].conv3. conv3's output still has to
        # pass through bn3, the residual addition and a ReLU before the pooled
        # feature the classifier sees, so attributing to it alone discards the
        # skip connection. Measured agreement with occlusion sensitivity over
        # the sample set: r=+0.31 for the block against r=+0.23 for conv3.
        return self.model.layer4


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
                    # Model type mismatch. Refusing the weights is correct, but
                    # returning silently hands back an UNTRAINED head that still
                    # emits confident-looking probabilities at chance level.
                    # Say so loudly instead.
                    warnings.warn(
                        f"Checkpoint '{w_path.name}' holds a '{ckpt_model_type}' model but "
                        f"'{name}' was requested. The weights were NOT loaded and this model "
                        f"is UNTRAINED - its predictions are meaningless. Set DEFAULT_MODEL="
                        f"'{ckpt_model_type}' or train a '{name}' checkpoint.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
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


def get_gradcam_reshape_transform(model: nn.Module):
    """
    Layout adapter for Grad-CAM, or None when the target is already NCHW.

    torchvision's Swin emits (B, H, W, C). pytorch-grad-cam assumes
    (B, C, H, W) and will otherwise treat the 768 channels as image width,
    producing a map anti-correlated with the model's actual sensitivity
    (measured r=-0.12 against an occlusion reference before this fix).
    """
    if isinstance(model, SwinTransformerBaseline):
        return lambda t: t.permute(0, 3, 1, 2).contiguous() if t.ndim == 4 else t
    return None


def get_model_target_layer(model: nn.Module) -> nn.Module:
    """Resolve the appropriate convolutional/attention target layer for Grad-CAM."""
    if hasattr(model, "get_target_gradcam_layer"):
        return model.get_target_gradcam_layer()
    if isinstance(model, HybridDenseNetSwinCBAM):
        return model.get_target_gradcam_layer()
    if isinstance(model, DenseNet121Baseline):
        # Use the output of the whole feature trunk (norm5, 1024 channels at
        # 7x7) -- this is exactly what the classifier pools over. The last
        # Conv2d in `features` is denseblock4.denselayer16.conv2, which emits
        # only the 32 new growth-rate channels of the final dense layer, i.e.
        # ~3% of the signal the decision uses. Measured deletion AUC over the
        # sample set: 0.683 for norm5 vs 0.723 for that conv (lower is more
        # faithful).
        return model.extractor.features.norm5
    if isinstance(model, SwinTransformerBaseline):
        # Final feature stage. Its output is NHWC, so this target is only
        # correct in combination with get_gradcam_reshape_transform.
        return model.extractor.features[-1]
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
