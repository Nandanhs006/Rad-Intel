"""
Unit tests for image preprocessing and normalization.
"""

import numpy as np
import torch
from rad_intel.preprocessing.transforms import default_preprocessor


def test_preprocessor_with_numpy(dummy_cxr_image):
    tensor, rgb = default_preprocessor.preprocess(dummy_cxr_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert rgb.shape == (224, 224, 3)
    assert rgb.dtype == np.uint8


def test_preprocessor_with_bytes(dummy_cxr_bytes):
    tensor, rgb = default_preprocessor.preprocess(dummy_cxr_bytes)
    assert tensor.shape == (1, 3, 224, 224)
    assert rgb.shape == (224, 224, 3)
