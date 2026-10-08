"""
Health check and system inspection routes.
"""

from fastapi import APIRouter, Depends
import torch

from rad_intel.config import settings
from rad_intel.api.schemas import HealthResponse, ModelsListResponse, ModelInfo
from rad_intel.api.dependencies import ModelManager, get_model_manager

router = APIRouter(tags=["System & Health"])


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
        ),
        ModelInfo(
            name="hybrid_no_cbam",
            description="Ablation Variant: DenseNet121 + Swin Transformer without CBAM Attention",
            is_hybrid=True,
            is_ablation=True,
            parameters_count=sum(p.numel() for p in manager.get_model("hybrid_no_cbam").parameters()),
        ),
        ModelInfo(
            name="densenet121",
            description="Ablation Baseline: Standalone DenseNet121 Convolutional Network",
            is_hybrid=False,
            is_ablation=True,
            parameters_count=sum(p.numel() for p in manager.get_model("densenet121").parameters()),
        ),
        ModelInfo(
            name="swin_t",
            description="Ablation Baseline: Standalone Swin Transformer (Tiny)",
            is_hybrid=False,
            is_ablation=True,
            parameters_count=sum(p.numel() for p in manager.get_model("swin_t").parameters()),
        ),
        ModelInfo(
            name="resnet50",
            description="Benchmark Baseline: ResNet50 Residual Network",
            is_hybrid=False,
            is_ablation=True,
            parameters_count=sum(p.numel() for p in manager.get_model("resnet50").parameters()),
        ),
    ]

    return ModelsListResponse(
        active_model=manager.active_model_name,
        available_models=models_meta,
    )
