"""
Grad-CAM & Grad-CAM++ Explainability Module for Rad-Intel.
Generates class activation maps and extracts anatomical quadrant localization metrics.
"""

from typing import Literal
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from rad_intel.models.factory import get_model_target_layer
from rad_intel.xai.visualizer import overlay_heatmap_on_image, image_to_base64


class GradCAMExplainer:
    """
    Grad-CAM / Grad-CAM++ engine for chest radiograph explainability.
    Computes visual saliency and computes anatomical localization
    (Left/Right Lung, Upper/Lower Zones) for downstream clinical report generation.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
        method: Literal["gradcam", "gradcam++"] = "gradcam",
    ):
        self.model = model
        self.target_layer = target_layer or get_model_target_layer(model)
        self.method = method

        cam_class = GradCAMPlusPlus if method == "gradcam++" else GradCAM
        self.cam = cam_class(
            model=self.model,
            target_layers=[self.target_layer],
        )

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_category: int | None = None,
    ) -> np.ndarray:
        """
        Generates normalized 2D Grad-CAM heatmap for the input tensor.

        Args:
            input_tensor: (1, 3, H, W) normalized float tensor.
            target_category: Category index to explain (None = model argmax).

        Returns:
            heatmap: (H, W) float32 numpy array with values in [0.0, 1.0].
        """
        targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        # grayscale_cam shape is (1, H, W)
        return grayscale_cam[0]

    @staticmethod
    def analyze_anatomical_localization(heatmap: np.ndarray) -> dict:
        """
        Maps the 2D heatmap into radiological anatomical zones.
        Note standard CXR radiological orientation:
          - Image Left (col 0 .. W/2) = Patient Right Lung
          - Image Right (col W/2 .. W) = Patient Left Lung
          - Image Top (row 0 .. H/2) = Upper Zone / Apical
          - Image Bottom (row H/2 .. H) = Lower Zone / Basal
        """
        h, w = heatmap.shape
        mid_h, mid_w = h // 2, w // 2

        # 4 Quadrants
        right_upper = float(np.mean(heatmap[0:mid_h, 0:mid_w]))
        right_lower = float(np.mean(heatmap[mid_h:h, 0:mid_w]))
        left_upper = float(np.mean(heatmap[0:mid_h, mid_w:w]))
        left_lower = float(np.mean(heatmap[mid_h:h, mid_w:w]))

        zones = {
            "right_upper_zone": round(right_upper, 4),
            "right_lower_zone": round(right_lower, 4),
            "left_upper_zone": round(left_upper, 4),
            "left_lower_zone": round(left_lower, 4),
        }

        # Find primary focus
        dominant_zone = max(zones, key=zones.get)
        dominant_score = zones[dominant_zone]

        right_lung_intensity = (right_upper + right_lower) / 2.0
        left_lung_intensity = (left_upper + left_lower) / 2.0
        total_intensity = right_lung_intensity + left_lung_intensity + 1e-6

        is_bilateral = (
            abs(right_lung_intensity - left_lung_intensity) / total_intensity < 0.35
            and dominant_score > 0.15
        )

        zone_descriptions = {
            "right_upper_zone": "Right upper lung field / apical region",
            "right_lower_zone": "Right lower lung field / retrocardiac & basal zone",
            "left_upper_zone": "Left upper lung field / apical region",
            "left_lower_zone": "Left lower lung field / retrocardiac & basal zone",
        }

        return {
            "zone_scores": zones,
            "dominant_zone": dominant_zone,
            "dominant_zone_description": zone_descriptions.get(dominant_zone, dominant_zone),
            "dominant_intensity": dominant_score,
            "right_lung_intensity": round(right_lung_intensity, 4),
            "left_lung_intensity": round(left_lung_intensity, 4),
            "is_bilateral": is_bilateral,
            "distribution_summary": (
                "Bilateral multifocal opacities"
                if is_bilateral
                else f"Focal opacity predominantly in {zone_descriptions.get(dominant_zone)}"
            ),
        }

    def explain(
        self,
        input_tensor: torch.Tensor,
        original_rgb: np.ndarray,
        target_category: int | None = None,
        alpha: float = 0.5,
    ) -> dict:
        """
        Complete explanation pipeline: produces heatmap, overlay, base64 data URL,
        and radiological anatomical quadrant localization.
        """
        heatmap = self.generate_heatmap(input_tensor, target_category)
        overlay_rgb = overlay_heatmap_on_image(original_rgb, heatmap, alpha=alpha)
        base64_overlay = image_to_base64(overlay_rgb)
        localization = self.analyze_anatomical_localization(heatmap)

        return {
            "method": self.method,
            "target_category": target_category,
            "heatmap": heatmap,
            "overlay_rgb": overlay_rgb,
            "overlay_base64": base64_overlay,
            "localization": localization,
        }
