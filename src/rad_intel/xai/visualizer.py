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


_LUNG_SEGMENTER = None
_LUNG_SEG_FAILED = False


def _get_lung_segmenter():
    """Lazily load the ChestX-Det PSPNet, once per process."""
    global _LUNG_SEGMENTER, _LUNG_SEG_FAILED
    if _LUNG_SEGMENTER is not None or _LUNG_SEG_FAILED:
        return _LUNG_SEGMENTER
    try:
        from torchxrayvision.baseline_models.chestx_det import PSPNet

        model = PSPNet()
        model.eval()
        _LUNG_SEGMENTER = model
    except Exception:
        # No weights, no network, or the package is absent. Callers fall back
        # to the body mask rather than losing the explanation.
        _LUNG_SEG_FAILED = True
    return _LUNG_SEGMENTER


def compute_lung_field_mask(image_rgb: np.ndarray, prob_threshold: float = 0.5) -> np.ndarray:
    """
    Segment the lung fields with a pretrained ChestX-Det PSPNet.

    Earlier revisions masked with a body silhouette and then with an intensity
    heuristic. Both failed on the cases that matter. A body mask keeps
    shoulders, neck and clavicles, so apical saliency still counted as "inside
    the thorax". An intensity rule assumes lung is darker than its
    surroundings, which is true of aerated lung and false of consolidated
    lung, so it carved out exactly the pathology an explanation should show.

    The segmentation network has no such failure mode: it was trained on chest
    radiographs with anatomical labels and returns Left Lung and Right Lung as
    classes distinct from Heart, Mediastinum and Facies Diaphragmatica, so the
    diaphragm and sub-diaphragmatic structures are excluded by construction
    while the lung bases and costophrenic angles are retained -- lower-lobe
    pneumonia must stay inside the mask.

    Masking changes only what is displayed and which zone is scored. The
    classifier is untouched, and the fraction of saliency falling outside the
    lungs is reported separately so the model's actual behaviour stays
    visible.
    """
    seg = _get_lung_segmenter()
    if seg is None:
        return compute_body_mask(image_rgb)

    try:
        import torch

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb
        h, w = gray.shape[:2]
        # The network expects the [-1024, 1024] range used by its training data.
        x = torch.from_numpy((gray.astype(np.float32) / 255.0 * 2048.0) - 1024.0)[None, None]
        x = torch.nn.functional.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        with torch.no_grad():
            probs = torch.sigmoid(seg(x))[0]

        idx = [seg.targets.index(name) for name in ("Left Lung", "Right Lung")]
        lung = (probs[idx].max(dim=0).values > prob_threshold).float()[None, None]
        lung = torch.nn.functional.interpolate(lung, size=(h, w), mode="nearest")
        mask = lung[0, 0].numpy().astype(bool)
    except Exception:
        return compute_body_mask(image_rgb)

    # A collapsed segmentation would hide the explanation entirely.
    if mask.mean() < 0.05:
        return compute_body_mask(image_rgb)

    # Small dilation: peripheral and subpleural consolidation sits at the edge
    # of the segmented field.
    return cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
