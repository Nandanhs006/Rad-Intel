"""
Health check and system inspection routes.
"""

from fastapi import APIRouter, Depends
import torch

from rad_intel.config import settings
from rad_intel.api.schemas import HealthResponse, ModelsListResponse, ModelInfo
from rad_intel.api.dependencies import ModelManager, get_model_manager
from rad_intel.models.registry import released

router = APIRouter(tags=["System & Health"])

# The scores the project reports, shown beside each built-in model so the
# chooser is not carrying numbers of its own that can drift out of date.
_REPORTED = {
    "hybrid": {"accuracy": 0.920, "auroc": 0.974},
    "hybrid_no_cbam": {"accuracy": 0.934, "auroc": 0.977},
    "densenet121": {"accuracy": 0.915, "auroc": 0.962},
    "swin_t": {"accuracy": 0.901, "auroc": 0.975},
    "resnet50": {"accuracy": 0.897, "auroc": 0.954},
}


def _score_label(metrics: dict | None) -> str | None:
    """"92.0% acc, AUC 0.974", or nothing when a version was never scored."""
    if not metrics:
        return None
    parts = []
    if metrics.get("accuracy") is not None:
        parts.append(f"{metrics['accuracy'] * 100:.1f}% acc")
    if metrics.get("auroc") is not None:
        parts.append(f"AUC {metrics['auroc']:.3f}")
    return ", ".join(parts) or None


@router.get("/health", response_model=HealthResponse)
@router.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Returns system status, active hardware device, PyTorch version, and service configuration."""
    return HealthResponse(
        status="ok",
        project_name=settings.PROJECT_NAME,
        version=settings.VERSION,
        torch_version=torch.__version__,
        active_device=str(settings.torch_device),
        default_model=settings.DEFAULT_MODEL,
        gemini_reporting_live=bool(settings.GEMINI_API_KEY),
    )


@router.get("/api/v1/models", response_model=ModelsListResponse)
async def list_models(manager: ModelManager = Depends(get_model_manager)):
    """Lists supported architectures, ablation models, and trainable parameter counts."""
    models_meta = [
        ModelInfo(
            name="hybrid",
            description="Proposed Rad-Intel Core Architecture: DenseNet121 + Swin Transformer with CBAM Attention",
            is_hybrid=True,
            is_ablation=False,
            parameters_count=sum(p.numel() for p in manager.get_model("hybrid").parameters()),
            label=_score_label(_REPORTED["hybrid"]),
            metrics=_REPORTED["hybrid"],
        ),
        ModelInfo(
            name="hybrid_no_cbam",
            description="Ablation Variant: DenseNet121 + Swin Transformer without CBAM Attention",
            is_hybrid=True,
            is_ablation=True,
            parameters_count=sum(p.numel() for p in manager.get_model("hybrid_no_cbam").parameters()),
            label=_score_label(_REPORTED["hybrid_no_cbam"]),
            metrics=_REPORTED["hybrid_no_cbam"],
        ),
        ModelInfo(
            name="densenet121",
            description="Ablation Baseline: Standalone DenseNet121 Convolutional Network",
            is_hybrid=False,
            is_ablation=True,
            parameters_count=sum(p.numel() for p in manager.get_model("densenet121").parameters()),
            label=_score_label(_REPORTED["densenet121"]),
            metrics=_REPORTED["densenet121"],
        ),
        ModelInfo(
            name="swin_t",
            description="Ablation Baseline: Standalone Swin Transformer (Tiny)",
            is_hybrid=False,
            is_ablation=True,
            parameters_count=sum(p.numel() for p in manager.get_model("swin_t").parameters()),
            label=_score_label(_REPORTED["swin_t"]),
            metrics=_REPORTED["swin_t"],
        ),
        ModelInfo(
            name="resnet50",
            description="Benchmark Baseline: ResNet50 Residual Network",
            is_hybrid=False,
            is_ablation=True,
            parameters_count=sum(p.numel() for p in manager.get_model("resnet50").parameters()),
            label=_score_label(_REPORTED["resnet50"]),
            metrics=_REPORTED["resnet50"],
        ),
    ]

    # Versions an administrator has released from reviewed cases. They appear
    # in the chooser beside the built-in ones, marked as released and
    # carrying the scores they were measured at, so nobody has to take on
    # trust that a newer model is a better one.
    for name, entry in sorted(released().items()):
        metrics = {k: float(v) for k, v in (entry.get("metrics") or {}).items()
                   if isinstance(v, (int, float))}
        models_meta.append(ModelInfo(
            name=name,
            description=(entry.get("note")
                         or f"Released version of the {entry['architecture']} model, "
                            f"fine-tuned on clinician-reviewed cases."),
            is_hybrid=entry["architecture"].startswith("hybrid"),
            is_ablation=False,
            parameters_count=sum(p.numel() for p in manager.get_model(name).parameters()),
            label=_score_label(metrics),
            is_released=True,
            released_at=entry.get("released_at"),
            metrics=metrics or None,
        ))

    return ModelsListResponse(
        active_model=manager.active_model_name,
        available_models=models_meta,
    )
