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
        self.active_model_name: str = settings.DEFAULT_MODEL

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

        pred_idx = int(probs.argmax())
        pred_class = class_names[pred_idx]
        confidence = float(probs[pred_idx])

        return pred_class, confidence, prob_dict, latency_ms


model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    return model_manager


def get_preprocessor() -> CXRPreprocessor:
    return default_preprocessor


def get_report_generator() -> ClinicalReportGenerator:
    return default_report_generator
