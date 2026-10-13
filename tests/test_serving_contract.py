"""
Regression tests for the serving path.

These cover the gap that let the API disagree with scripts/evaluate.py: the
evaluation script read the checkpoint's calibrated threshold while the served
predict() silently used argmax, so reported metrics and deployed behaviour
diverged without any test noticing.
"""

import warnings
import torch
import pytest

from rad_intel.config import settings
from rad_intel.api.dependencies import model_manager
from rad_intel.models.factory import create_model


CKPT = settings.BASE_DIR / "weights" / f"best_{settings.DEFAULT_MODEL}_model.pt"


@pytest.mark.skipif(not CKPT.exists(), reason="no checkpoint for the default model")
def test_served_threshold_matches_checkpoint():
    """predict() must use the threshold training selected, not argmax."""
    meta = torch.load(CKPT, map_location="cpu", weights_only=False)
    expected = meta.get("optimal_threshold")
    if expected is None:
        pytest.skip("checkpoint records no optimal_threshold")
    assert model_manager.get_threshold(settings.DEFAULT_MODEL) == pytest.approx(float(expected))


@pytest.mark.skipif(not CKPT.exists(), reason="no checkpoint for the default model")
def test_threshold_decides_the_label():
    """A probability below the threshold must be reported as NORMAL."""
    thr = model_manager.get_threshold(settings.DEFAULT_MODEL)
    assert 0.0 < thr < 1.0

    class _Stub(torch.nn.Module):
        """Emits a fixed P(PNEUMONIA) just under the operating point."""

        def forward(self, x):
            p = max(thr - 0.05, 0.01)
            return torch.log(torch.tensor([[1.0 - p, p]]))

    name = settings.DEFAULT_MODEL
    original = model_manager._models.get(name)
    model_manager._models[name] = _Stub()
    try:
        label, _, probs, _ = model_manager.predict(torch.zeros(1, 3, 224, 224), name)
    finally:
        if original is not None:
            model_manager._models[name] = original
        else:
            model_manager._models.pop(name, None)

    assert probs["PNEUMONIA"] < thr
    assert label == "NORMAL", "probability under the threshold must not be called PNEUMONIA"


def test_mismatched_checkpoint_warns_instead_of_failing_silently():
    """An architecture/checkpoint mismatch must not quietly return an untrained model."""
    if not CKPT.exists():
        pytest.skip("no checkpoint available")
    other = "resnet50" if settings.DEFAULT_MODEL != "resnet50" else "densenet121"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        create_model(model_name=other, pretrained=False, weights_path=str(CKPT))
    assert any(issubclass(w.category, RuntimeWarning) and "UNTRAINED" in str(w.message) for w in caught)


# --- API error contract -------------------------------------------------
# These guard behaviours that previously failed silently or misreported:
# a deliberate 400 was being swallowed by a bare `except Exception` and
# re-raised as 500, and a model with no checkpoint returned a confident
# prediction from an untrained network.

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from rad_intel.api.main import app


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_empty_upload_is_client_error(client):
    r = await client.post("/api/v1/predict", files={"file": ("x.jpeg", b"", "image/jpeg")})
    assert r.status_code == 400, f"expected 400, got {r.status_code}"


async def test_undecodable_upload_is_client_error(client):
    r = await client.post(
        "/api/v1/predict", files={"file": ("x.jpeg", b"this is not an image", "image/jpeg")}
    )
    assert r.status_code == 400, f"expected 400, got {r.status_code}"


async def test_untrained_model_is_refused(client, dummy_cxr_bytes):
    """A model with no checkpoint must not return a prediction."""
    if (settings.BASE_DIR / "weights" / "best_hybrid_model.pt").exists():
        pytest.skip("hybrid checkpoint present; nothing untrained to test")
    r = await client.post(
        "/api/v1/predict",
        files={"file": ("cxr.jpeg", dummy_cxr_bytes, "image/jpeg")},
        data={"model_name": "hybrid"},
    )
    assert r.status_code == 503, f"expected 503, got {r.status_code}"
    assert "no trained checkpoint" in r.json()["detail"]


def test_serving_preprocessing_matches_training():
    """
    The service must build the same tensor the models were trained on.

    These two paths drifted apart once: training resized through torchvision
    (PIL bilinear) while the service used cv2.INTER_AREA. The same radiograph
    produced a different tensor, shifting P(pneumonia) by 0.024 on average and
    flipping 16 of the 624 test decisions, so the reported metrics and the
    deployed classifier were not quite the same model.
    """
    import glob
    import numpy as np
    import torch
    from PIL import Image
    from rad_intel.preprocessing.transforms import default_preprocessor
    from rad_intel.training.dataset import get_transforms

    files = sorted(glob.glob(str(settings.BASE_DIR / "Rad-Intel paper" / "chest_xray"
                                 / "test" / "*" / "*.jpeg")))[:8]
    if not files:
        pytest.skip("test radiographs not available")

    eval_tf = get_transforms(is_train=False)
    for f in files:
        served, rgb = default_preprocessor.preprocess(f)
        trained = eval_tf(Image.open(f).convert("RGB"))[None]
        assert torch.allclose(served, trained, atol=1e-6), (
            f"serving and training preprocessing disagree on {f}: "
            f"max |diff| = {float((served - trained).abs().max()):.3e}"
        )
        # The overlay base must stay a plain uint8 RGB image of the same size.
        assert rgb.dtype == np.uint8 and rgb.shape == (settings.IMAGE_SIZE,
                                                       settings.IMAGE_SIZE, 3)
