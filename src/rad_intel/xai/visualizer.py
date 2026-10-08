"""
Visualization utilities for XAI saliency heatmaps and overlays.
"""

import base64
import io
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Normalizes heatmap array to [0.0, 1.0]."""
    denom = np.max(heatmap) - np.min(heatmap)
    if denom == 0:
        return np.zeros_like(heatmap, dtype=np.float32)
    return (heatmap - np.min(heatmap)) / denom


def overlay_heatmap_on_image(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlays a 2D saliency heatmap onto an RGB background image.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image.
        heatmap: (H, W) or (h, w) float heatmap.
        alpha: Blend ratio for heatmap (0.0 to 1.0).
        colormap: OpenCV colormap ID (default: cv2.COLORMAP_JET).

    Returns:
        overlay_rgb: (H, W, 3) uint8 RGB image.
    """
    h, w = image_rgb.shape[:2]
    if heatmap.shape != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    norm_map = normalize_heatmap(heatmap)
    uint8_map = np.uint8(255 * norm_map)

    # OpenCV colormap expects BGR, then convert back to RGB
    colored_bgr = cv2.applyColorMap(uint8_map, colormap)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    # Alpha blending: image * (1 - alpha) + colored_rgb * alpha
    blended = np.float32(image_rgb) * (1.0 - alpha) + np.float32(colored_rgb) * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended


def image_to_base64(image_rgb: np.ndarray, format: str = "PNG") -> str:
    """Encodes an RGB NumPy image array to a data URL base64 string."""
    pil_img = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


def save_overlay(image_rgb: np.ndarray, output_path: str | Path) -> Path:
    """Saves RGB image to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_bgr)
    return path
