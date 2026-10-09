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
    alpha: float = 0.6,
    colormap: int = cv2.COLORMAP_JET,
    threshold: float = 0.6,
    gamma: float = 1.5,
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

    # Thresholded, saliency-weighted alpha blending.
    #
    # A constant alpha paints EVERY pixel, and JET is vivid at every value
    # (zero maps to saturated blue), so the radiograph used to disappear under
    # a full-frame colour wash. Grad-CAM on this model is also diffuse -- after
    # min-max normalisation roughly half the frame sits above 0.5 -- so merely
    # scaling opacity by saliency is not enough on its own.
    #
    # `threshold` sets where colour starts (below it the pixel is left as the
    # original radiograph), and `gamma` fades the lower part of what remains,
    # leaving colour only on the region the model actually responded to.
    # `alpha` is the peak opacity reached at maximum saliency.
    # `threshold` is interpreted as a quantile of this map, not an absolute
    # value, so the painted area stays comparable between a sharply peaked CAM
    # and a diffuse one. A fixed absolute cut shows half the frame on a flat
    # map and a single speck on a peaked one.
    cut = float(np.quantile(norm_map, np.clip(threshold, 0.0, 0.99)))
    ramp = np.clip((norm_map - cut) / max(1.0 - cut, 1e-6), 0.0, 1.0) ** gamma
    pixel_alpha = (alpha * ramp).astype(np.float32)[..., None]
    blended = np.float32(image_rgb) * (1.0 - pixel_alpha) + np.float32(colored_rgb) * pixel_alpha
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


def compute_body_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    Boolean mask of the thorax/patient region in a preprocessed radiograph.

    Chest radiographs in this dataset arrive letterboxed, with black padding
    and background surrounding the patient. Saliency falling there explains
    nothing anatomical, and quadrant statistics computed over it are diluted by
    pixels that contain no tissue. Otsu thresholding plus a morphological close
    and largest-connected-component selection isolates the body reliably enough
    for both purposes.

    Returns an all-True mask if segmentation fails, so callers degrade to the
    previous unmasked behaviour rather than losing the explanation entirely.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb
    try:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        if n_labels <= 1:
            return np.ones(gray.shape, dtype=bool)
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = labels == largest
    except cv2.error:
        return np.ones(gray.shape, dtype=bool)

    # A mask covering almost nothing means segmentation failed; prefer the
    # unmasked map over returning an empty explanation.
    if mask.mean() < 0.15:
        return np.ones(gray.shape, dtype=bool)
    return mask


def compute_lung_field_mask(
    image_rgb: np.ndarray,
    top: float = 0.18,
    bottom: float = 0.82,
    side: float = 0.10,
) -> np.ndarray:
    """
    Approximate lung-field region: the body mask narrowed to the band where
    lung parenchyma actually lies.

    compute_body_mask keeps the whole patient silhouette, roughly 70% of a
    224x224 frame, so shoulders, neck and upper arms count as "inside the
    thorax" and saliency landing on a clavicle survives masking. On a frontal
    chest radiograph the lungs occupy a predictable band of the body bounding
    box, so the fractions above trim the apical/cervical region above the lung
    apices, the sub-diaphragmatic region below the costophrenic angles, and
    the lateral chest wall.

    This is a geometric prior, not a segmentation. It is deliberately crude:
    it cannot follow the diaphragm or the mediastinal border, and on an
    unusually rotated or cropped film it will clip real lung. It changes only
    what is displayed and scored by zone -- the classifier is untouched, and
    the share of saliency falling outside this region is still reported so the
    underlying behaviour stays visible.
    """
    body = compute_body_mask(image_rgb)
    rows = np.where(body.any(axis=1))[0]
    cols = np.where(body.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return body

    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    h, w = r1 - r0 + 1, c1 - c0 + 1

    band = np.zeros_like(body)
    band[r0 + int(h * top) : r0 + int(h * bottom),
         c0 + int(w * side) : c0 + int(w * (1.0 - side))] = True

    mask = body & band
    # Never hand back an empty explanation.
    return mask if mask.mean() >= 0.08 else body
