"""
FastAPI dependency injection and singleton model management.
"""

import time
import torch
import torch.nn.functional as F
from rad_intel.config import settings
from rad_intel.models.factory import create_model, ModelType, get_model_target_layer
from rad_intel.preprocessing.transforms import CXRPreprocessor, default_preprocessor
from rad_intel.reporting.report_generator import ClinicalReportGenerator, default_report_generator
from rad_intel.xai.gradcam import GradCAMExplainer


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
        self.active_model_name: str = settings.DEFAULT_MODEL

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
            # Check for model-specific checkpoint, then configured path
            weights_to_load = settings.MODEL_WEIGHTS_PATH
            if not weights_to_load:
                model_ckpt = settings.BASE_DIR / "weights" / f"best_{target_name}_model.pt"
                if model_ckpt.exists():
                    weights_to_load = str(model_ckpt)
                elif (settings.BASE_DIR / "weights" / "best_hybrid_model.pt").exists():
                    weights_to_load = str(settings.BASE_DIR / "weights" / "best_hybrid_model.pt")

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
            if weights_to_load:
                try:
                    meta = torch.load(weights_to_load, map_location="cpu", weights_only=False)
                    if isinstance(meta, dict) and meta.get("model_type") in (None, target_name):
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
