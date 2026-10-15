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


@pytest.fixture(autouse=True)
def no_live_delivery(monkeypatch):
    """
    Keep every test away from the real delivery configuration.

    config.py loads .env into the process environment, so a developer with
    working SMTP credentials had them visible to the whole suite -- and the
    tests that exercise the "nothing is configured" paths went ahead and sent
    live email instead. Delivery is something tests opt into by setting these
    themselves, never something they inherit.
    """
    for name in (
        "SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM",
        "RAD_INTEL_OTP_CHANNEL", "RAD_INTEL_OTP_ECHO", "RAD_INTEL_ADMIN_EMAIL",
        "RAD_INTEL_CORS_ORIGINS", "RAD_INTEL_FEEDBACK_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def no_live_language_model(monkeypatch):
    """
    Keep the suite off the language API.

    Running it was making real calls on whoever's key was in .env: it spent
    their quota, needed a network, and made the reporting test flaky because
    it asserted things about wording a model chooses afresh each time. The
    offline path is the one worth testing anyway -- it is what runs when the
    key is missing or the service is down, which is exactly when a report
    still has to come out. A test that wants the live client sets the key
    itself.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("rad_intel.config.settings.GEMINI_API_KEY", None)
    try:
        from rad_intel.reporting import report_generator as rg

        monkeypatch.setattr(rg.default_report_generator.gemini_client, "_client", None,
                            raising=False)
    except Exception:
        pass
