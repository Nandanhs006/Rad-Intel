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
