"""
FastAPI dependency injection and singleton model management.
"""

import json
import logging
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from rad_intel.config import settings
from rad_intel.models.factory import create_model, ModelType, get_model_target_layer
from rad_intel.preprocessing.transforms import CXRPreprocessor, default_preprocessor
from rad_intel.reporting.report_generator import ClinicalReportGenerator, default_report_generator
from rad_intel.xai.gradcam import GradCAMExplainer

logger = logging.getLogger(__name__)

# Which checkpoint to serve for a given architecture, when it is not the base
# one. Written by an administrator activating a fine-tuned version.
#
# A pointer file rather than copying the new weights over the base checkpoint:
# the base file is the one whose numbers the project reports, so it stays
# byte-identical on disk and reverting cannot fail halfway through a copy.
ACTIVE_POINTER = settings.BASE_DIR / "weights" / "active_models.json"


def _read_pointer() -> dict[str, str]:
    if not ACTIVE_POINTER.exists():
        return {}
    try:
        data = json.loads(ACTIVE_POINTER.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Ignoring unreadable %s: %s", ACTIVE_POINTER.name, exc)
        return {}


class ModelManager:
    """Manages active neural network models, caching, and inference execution."""

    def __init__(self):
        self.device = settings.torch_device
        self._models: dict[str, torch.nn.Module] = {}
        # Per-model operating point recovered from the checkpoint. Training
        # selects this on validation data and stores it as "optimal_threshold";
        # serving must apply the same value or the deployed decision boundary
        # silently reverts to argmax (0.5) and disagrees with the reported
        # sensitivity/specificity.
        self._thresholds: dict[str, float] = {}
        # Whether a trained checkpoint was actually loaded. Without this the
        # API happily serves an ImageNet-initialised network with a random
        # classifier head and returns confident-looking probabilities for it.
        self._trained: dict[str, bool] = {}
        self.active_model_name: str = settings.DEFAULT_MODEL
        self._load_lock = threading.Lock()

    # --- which checkpoint is served ---------------------------------------

    def base_checkpoint(self, model_name: str) -> Path:
        return settings.BASE_DIR / "weights" / f"best_{model_name}_model.pt"

    def active_checkpoint(self, model_name: str | None = None) -> str | None:
        """
        The checkpoint path this model will actually load, or None if there is none.

        An activated version wins; otherwise the architecture's own base file;
        otherwise the hybrid base, which is the historical fallback.
        """
        target = self._clean_model_name(model_name)
        if settings.MODEL_WEIGHTS_PATH:
            return settings.MODEL_WEIGHTS_PATH

        chosen = _read_pointer().get(target)
        if chosen:
            candidate = settings.BASE_DIR / "weights" / Path(chosen).name
            if candidate.exists():
                return str(candidate)
            logger.warning("Activated checkpoint %s is missing; falling back to the base.",
                           chosen)

        own = self.base_checkpoint(target)
        if own.exists():
            return str(own)
        fallback = self.base_checkpoint("hybrid")
        return str(fallback) if fallback.exists() else None

    def activate_checkpoint(self, model_name: str, checkpoint: Path | None) -> None:
        """
        Serve `checkpoint` for this architecture, or the base one when None.

        The loaded model is dropped so the next request picks the new weights
        up along with their own calibrated threshold. Loading happens on that
        request rather than here, so a bad file surfaces as a failed request
        instead of a half-swapped manager.
        """
        pointer = _read_pointer()
        if checkpoint is None:
            pointer.pop(model_name, None)
        else:
            pointer[model_name] = checkpoint.name
        ACTIVE_POINTER.parent.mkdir(parents=True, exist_ok=True)
        ACTIVE_POINTER.write_text(json.dumps(pointer, indent=2) + "\n", encoding="utf-8")

        with self._load_lock:
            self._models.pop(model_name, None)
            self._thresholds.pop(model_name, None)
            self._trained.pop(model_name, None)
        logger.info("Serving %s for %s.",
                    checkpoint.name if checkpoint else "the base checkpoint", model_name)

    def is_trained(self, model_name: str | None = None) -> bool:
        """True when a matching trained checkpoint was loaded for this model."""
        target_name = self._clean_model_name(model_name)
        self.get_model(target_name)
        return self._trained.get(target_name, False)

    def require_trained(self, model_name: str | None = None) -> str:
        """
        Reject a request for a model with no trained weights.

        Serving an untrained network is worse than serving an error: it returns
        a plausible probability and a Grad-CAM map that look like results.
        """
        from fastapi import HTTPException

        target_name = self._clean_model_name(model_name)
        if not self.is_trained(target_name):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model '{target_name}' has no trained checkpoint "
                    f"(expected weights/best_{target_name}_model.pt). Its predictions "
                    f"would be meaningless, so the request was refused. Train it first "
                    f"or select a model that has weights."
                ),
            )
        return target_name

    def get_threshold(self, model_name: str | None = None) -> float:
        """Operating threshold on P(PNEUMONIA); 0.5 when the checkpoint has none."""
        target_name = self._clean_model_name(model_name)
        self.get_model(target_name)
        return self._thresholds.get(target_name, 0.5)

    def _clean_model_name(self, model_name: str | None) -> str:
        if not model_name or str(model_name).strip().lower() in ("", "string", "null", "none"):
            return self.active_model_name
        return str(model_name).lower().strip()

    def get_model(self, model_name: str | None = None) -> torch.nn.Module:
        """Retrieves or lazily instantiates the requested model architecture."""
        target_name = self._clean_model_name(model_name)
        if target_name not in self._models:
            weights_to_load = self.active_checkpoint(target_name)

            model = create_model(
                model_name=target_name,  # type: ignore
                num_classes=settings.NUM_CLASSES,
                pretrained=True,
                weights_path=weights_to_load,
                device=self.device,
            )
            self._models[target_name] = model

            # Recover the validation-selected operating point from the same
            # checkpoint, but only when the checkpoint actually belongs to this
            # architecture -- create_model discards mismatched weights, and
            # applying a threshold calibrated for another model would be worse
            # than falling back to 0.5.
            self._trained[target_name] = False
            if weights_to_load:
                try:
                    meta = torch.load(weights_to_load, map_location="cpu", weights_only=False)
                    if isinstance(meta, dict) and meta.get("model_type") in (None, target_name):
                        self._trained[target_name] = True
                        thr = meta.get("optimal_threshold")
                        if isinstance(thr, (int, float)) and 0.0 < float(thr) < 1.0:
                            self._thresholds[target_name] = float(thr)
                except Exception:
                    pass
        return self._models[target_name]

    def predict(
        self, tensor: torch.Tensor, model_name: str | None = None
    ) -> tuple[str, float, dict[str, float], float]:
        """
        Runs model inference.
        Returns:
            predicted_class: "NORMAL" or "PNEUMONIA"
            confidence: float in [0.0, 1.0]
            probabilities: dict of class -> probability
            latency_ms: elapsed time in milliseconds
        """
        target_name = self._clean_model_name(model_name)
        model = self.get_model(target_name)
        start_t = time.perf_counter()

        with torch.no_grad():
            logits = model(tensor.to(self.device))
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        latency_ms = (time.perf_counter() - start_t) * 1000.0

        class_names = settings.CLASS_NAMES
        prob_dict = {
            class_names[i]: float(probs[i]) for i in range(len(class_names))
        }

        # Apply the checkpoint's calibrated operating point rather than argmax.
        # With PNEUMONIA as the positive class, argmax is the special case
        # threshold == 0.5; this model's validation-selected value is 0.70, so
        # argmax was over-calling pneumonia on every borderline radiograph.
        threshold = self._thresholds.get(target_name, 0.5)
        p_pneumonia = prob_dict.get("PNEUMONIA", float(probs[-1]))
        is_pneumonia = p_pneumonia >= threshold
        pred_class = "PNEUMONIA" if is_pneumonia else "NORMAL"
        confidence = p_pneumonia if is_pneumonia else 1.0 - p_pneumonia

        return pred_class, confidence, prob_dict, latency_ms


model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    return model_manager


def get_preprocessor() -> CXRPreprocessor:
    return default_preprocessor


def get_report_generator() -> ClinicalReportGenerator:
    return default_report_generator
