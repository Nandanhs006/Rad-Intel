"""
LIME (Local Interpretable Model-agnostic Explanations) Module for Rad-Intel.
Generates superpixel-based explanations for chest X-ray predictions.
"""

from typing import Callable
import numpy as np
import torch
import torch.nn.functional as F
from lime import lime_image
from skimage.segmentation import mark_boundaries

from rad_intel.xai.visualizer import image_to_base64


class LIMECXRExplainer:
    """
    LIME image explanation engine for Chest X-Ray models.
    Identifies interpretable superpixels contributing positively or negatively
    to the pneumonia prediction.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device | str = "cpu"):
        self.model = model
        self.device = device
        self.explainer = lime_image.LimeImageExplainer(random_state=42)

    def _get_predict_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        """Creates a prediction function compatible with LIME (NumPy -> NumPy probabilities)."""
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        def batch_predict(images: np.ndarray) -> np.ndarray:
            # images shape: (B, H, W, 3), values in [0, 255] or [0, 1]
            if images.max() > 1.0:
                imgs_norm = images.astype(np.float32) / 255.0
            else:
                imgs_norm = images.astype(np.float32)

            standardized = (imgs_norm - mean) / std
            tensor = torch.from_numpy(standardized).permute(0, 3, 1, 2).float().to(self.device)

            self.model.eval()
            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
            return probs

        return batch_predict

    def explain(
        self,
        image_rgb: np.ndarray,
        target_category: int = 1,
        num_samples: int = 150,
        num_features: int = 5,
    ) -> dict:
        """
        Generates LIME superpixel explanation.

        Args:
            image_rgb: (H, W, 3) uint8 RGB image.
            target_category: Category index to explain (default: 1 for Pneumonia).
            num_samples: Number of perturbation samples (100-200 for fast inference).
            num_features: Top superpixels to highlight.

        Returns:
            dict containing overlay image, base64 string, and superpixel weights.
        """
        predict_fn = self._get_predict_fn()

        explanation = self.explainer.explain_instance(
            image_rgb,
            predict_fn,
            top_labels=2,
            hide_color=0,
            num_samples=num_samples,
            random_seed=42,
        )

        temp, mask = explanation.get_image_and_mask(
            target_category,
            positive_only=True,
            num_features=num_features,
            hide_rest=False,
        )

        # Mark superpixel boundaries on original image
        marked = mark_boundaries(temp / 255.0 if temp.max() > 1.0 else temp, mask)
        overlay_rgb = (marked * 255).astype(np.uint8)
        base64_overlay = image_to_base64(overlay_rgb)

        return {
            "method": "lime",
            "target_category": target_category,
            "num_samples": num_samples,
            "overlay_rgb": overlay_rgb,
            "overlay_base64": base64_overlay,
            "top_features_count": num_features,
        }
