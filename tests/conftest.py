"""
Shared pytest fixtures for Rad-Intel test suite.
"""

import io
import cv2
import numpy as np
import pytest
from PIL import Image
import torch


@pytest.fixture
def dummy_cxr_image():
    """Generates a dummy 224x224 RGB image for testing."""
    img = np.ones((224, 224, 3), dtype=np.uint8) * 60
    cv2.circle(img, (80, 110), 30, (200, 200, 200), -1)
    cv2.circle(img, (144, 110), 30, (200, 200, 200), -1)
    return img


@pytest.fixture
def dummy_cxr_bytes(dummy_cxr_image):
    """Generates dummy PNG bytes."""
    pil_img = Image.fromarray(dummy_cxr_image)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def dummy_tensor():
    """Generates a (1, 3, 224, 224) dummy PyTorch float tensor."""
    return torch.randn(1, 3, 224, 224)
