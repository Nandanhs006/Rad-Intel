"""
FastAPI dependency injection and singleton model management.
"""

import json
import logging
import os
import threading
import time
from collections import OrderedDict
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

# How many models may be held in memory at once, besides the default.
#
# Each one is between 30 MB and 600 MB of weights, and a loaded model was
# never dropped: anybody could click down the model list on the Analyse tab
# and load every architecture plus every released version. Measured, that is
# 1.2 GB for the five built-in ones, and the lung-segmentation model the
# explanation path loads lazily adds 380 MB on top -- enough to be killed on
# the 2 GB box this is meant to run on, by a visitor doing nothing wrong.
#
# Three is room for the default plus two people comparing, which is what the
# chooser is actually for. Raise it with RAD_INTEL_MODEL_CACHE on a bigger
# machine.
def _cache_limit() -> int:
    try:
        return max(1, int(os.environ.get("RAD_INTEL_MODEL_CACHE", "").strip() or 3))
    except ValueError:
        return 3


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
        # Ordered by when each was last used, so the one evicted is the one
        # nobody has asked for in longest.
        self._models: OrderedDict[str, torch.nn.Module] = OrderedDict()
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

    def known_models(self) -> list[str]:
        """Built-in architectures plus whatever has been released."""
        from rad_intel.models import registry

        return list(settings.CLASS_NAMES and [
            "hybrid", "densenet121", "swin_t", "hybrid_no_cbam", "resnet50",
        ]) + sorted(registry.released())

    def active_checkpoint(self, model_name: str | None = None) -> str | None:
        """
        The checkpoint path this model will actually load, or None if there is none.

        An activated version wins; otherwise the architecture's own base file;
        otherwise the hybrid base, which is the historical fallback.
        """
        target = self._clean_model_name(model_name)

        # A released version names its own checkpoint, and wins over
        # everything else: it was published deliberately.
        from rad_intel.models import registry

        resolved = registry.resolve(target)
        if resolved:
            return str(resolved[1])

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

    def forget(self, model_name: str) -> None:
        """Drop a cached model so the next request loads it afresh."""
        with self._load_lock:
            self._models.pop(model_name, None)
            self._thresholds.pop(model_name, None)
            self._trained.pop(model_name, None)

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

        # The form field is a plain string now, because a released version's
        # name is not known until it is released. This is where a name that
        # is not on offer gets refused.
        known = self.known_models()
        if target_name not in known:
            raise HTTPException(
                status_code=400,
                detail=(f"There is no model called '{target_name}'. The ones available "
                        f"are: {', '.join(known)}."),
            )

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
            from rad_intel.models import registry

            weights_to_load = self.active_checkpoint(target_name)
            resolved = registry.resolve(target_name)
            # A released version is one of the known architectures carrying
            # different weights, so the network is built from the
            # architecture and the name is only a label.
            architecture = resolved[0] if resolved else target_name

            model = create_model(
                model_name=architecture,  # type: ignore
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
                    if isinstance(meta, dict) and meta.get("model_type") in (None, architecture):
                        self._trained[target_name] = True
                        thr = meta.get("optimal_threshold")
                        if isinstance(thr, (int, float)) and 0.0 < float(thr) < 1.0:
                            self._thresholds[target_name] = float(thr)
                except Exception:
                    pass
            self._evict_down_to_limit(keep=target_name)

        # Asked for again, so it goes to the back of the queue to be dropped.
        self._models.move_to_end(target_name)
        return self._models[target_name]

    def _evict_down_to_limit(self, keep: str) -> None:
        """
        Drop the least recently used models until the cache is small enough.

        The default model is never dropped: it is what the great majority of
        requests want, and reloading it costs a second and a half on every
        one of them. Neither is the one being returned right now.
        """
        limit = _cache_limit()
        with self._load_lock:
            protected = {keep, self.active_model_name}
            while len(self._models) > limit:
                for name in list(self._models):
                    if name not in protected:
                        self._models.pop(name, None)
                        self._thresholds.pop(name, None)
                        self._trained.pop(name, None)
                        logger.info("Unloaded %s to stay inside the model cache "
                                    "limit of %d.", name, limit)
                        break
                else:
                    # Everything left is protected; nothing more can go.
                    return

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
