"""
Unit tests for XAI explainability and visualization.
"""

import numpy as np
import pytest
from rad_intel.models.factory import create_model
from rad_intel.xai.gradcam import GradCAMExplainer
from rad_intel.xai.visualizer import normalize_heatmap, overlay_heatmap_on_image, image_to_base64


def test_visualizer_functions(dummy_cxr_image):
    h, w = dummy_cxr_image.shape[:2]
    mock_heatmap = np.random.rand(h, w).astype(np.float32)

    norm = normalize_heatmap(mock_heatmap)
    assert 0.0 <= norm.min() <= norm.max() <= 1.0

    overlay = overlay_heatmap_on_image(dummy_cxr_image, norm)
    assert overlay.shape == dummy_cxr_image.shape
    assert overlay.dtype == np.uint8

    b64 = image_to_base64(overlay)
    assert b64.startswith("data:image/png;base64,")


def test_gradcam_explanation(dummy_tensor, dummy_cxr_image):
    model = create_model("hybrid", num_classes=2, pretrained=False)
    explainer = GradCAMExplainer(model=model, method="gradcam")

    res = explainer.explain(dummy_tensor, dummy_cxr_image, target_category=1)
    assert "heatmap" in res
    assert "overlay_base64" in res
    assert "localization" in res
    assert res["overlay_base64"].startswith("data:image/png;base64,")

    loc = res["localization"]
    assert "zone_scores" in loc
    assert "dominant_zone" in loc
