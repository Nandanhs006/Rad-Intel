"""
Medical image preprocessing pipeline for Chest Radiographs (CXRs).
Includes CLAHE contrast enhancement, resizing, and PyTorch normalization.
"""

import io
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

from rad_intel.config import settings


class CXRPreprocessor:
    """
    Standardizes Chest X-Ray images for deep learning inference and explainability.
    Handles RGB/Grayscale inputs, applies CLAHE enhancement, resizes to target resolution,
    and returns both normalized tensor and visualizable NumPy image.
    """

    def __init__(
        self,
        image_size: int = settings.IMAGE_SIZE,
        apply_clahe: bool = settings.APPLY_CLAHE,
        mean: list[float] = settings.NORM_MEAN,
        std: list[float] = settings.NORM_STD,
    ):
        self.image_size = image_size
        self.apply_clahe = apply_clahe
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def load_image(self, source: str | Path | bytes | Image.Image | np.ndarray) -> np.ndarray:
        """Loads and decodes an image into a uint8 RGB NumPy array."""
        if isinstance(source, (str, Path)):
            img = cv2.imread(str(source))
            if img is None:
                raise ValueError(f"Failed to read image from path: {source}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img_rgb

        if isinstance(source, bytes):
            nparr = np.frombuffer(source, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image from byte buffer.")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img_rgb

        if isinstance(source, Image.Image):
            return np.array(source.convert("RGB"))

        if isinstance(source, np.ndarray):
            if source.ndim == 2:
                return cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
            if source.ndim == 3 and source.shape[2] == 1:
                return cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
            return source

        raise TypeError(f"Unsupported image source type: {type(source)}")

    def enhance_clahe(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
        to the luminance channel to enhance lung field opacities.
        """
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)

        limg = cv2.merge((cl, a_channel, b_channel))
        enhanced_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return enhanced_rgb

    def preprocess(
        self,
        source: str | Path | bytes | Image.Image | np.ndarray,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, np.ndarray]:
        """
        Full preprocessing pipeline.
        Returns:
            tensor: (1, 3, image_size, image_size) normalized PyTorch float tensor on target device.
            rgb_resized: (image_size, image_size, 3) uint8 RGB array (for overlays & LIME).
        """
        img_rgb = self.load_image(source)

        if self.apply_clahe:
            img_rgb = self.enhance_clahe(img_rgb)

        # Resize to standardized dimensions
        resized_rgb = cv2.resize(
            img_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
        )

        # Normalize to [0, 1] float
        norm_img = resized_rgb.astype(np.float32) / 255.0
        standardized = (norm_img - self.mean) / self.std

        # Convert to PyTorch format (C, H, W) and add batch dimension (1, C, H, W)
        tensor = torch.from_numpy(standardized).permute(2, 0, 1).unsqueeze(0).float()
        tensor = tensor.to(device)

        return tensor, resized_rgb


# Default preprocessor instance
default_preprocessor = CXRPreprocessor()
