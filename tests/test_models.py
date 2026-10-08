"""
Unit tests for Rad-Intel deep learning model architectures.
"""

import pytest
import torch
from rad_intel.models.cbam import CBAM, ChannelAttention, SpatialAttention
from rad_intel.models.hybrid import HybridDenseNetSwinCBAM
from rad_intel.models.factory import create_model, get_model_target_layer


def test_channel_attention():
    ca = ChannelAttention(in_planes=64, ratio=16)
    x = torch.randn(2, 64, 14, 14)
    out = ca(x)
    assert out.shape == (2, 64, 1, 1)
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_spatial_attention():
    sa = SpatialAttention(kernel_size=7)
    x = torch.randn(2, 64, 14, 14)
    out = sa(x)
    assert out.shape == (2, 1, 14, 14)
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_cbam_block():
    cbam = CBAM(in_planes=128, ratio=16, kernel_size=7)
    x = torch.randn(2, 128, 14, 14)
    out = cbam(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("model_name", ["hybrid", "hybrid_no_cbam", "densenet121", "swin_t", "resnet50"])
def test_all_model_forward_passes(model_name, dummy_tensor):
    model = create_model(model_name, num_classes=2, pretrained=False)
    model.eval()
    with torch.no_grad():
        logits = model(dummy_tensor)
    assert logits.shape == (1, 2)
    assert not torch.isnan(logits).any()

    target_layer = get_model_target_layer(model)
    assert target_layer is not None
