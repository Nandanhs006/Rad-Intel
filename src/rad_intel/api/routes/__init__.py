from rad_intel.api.routes.health import router as health_router
from rad_intel.api.routes.predict import router as predict_router
from rad_intel.api.routes.explain import router as explain_router
from rad_intel.api.routes.report import router as report_router
from rad_intel.api.routes.analyze import router as analyze_router
from rad_intel.api.routes.history import router as history_router

__all__ = [
    "health_router",
    "predict_router",
    "explain_router",
    "report_router",
    "analyze_router",
    "history_router",
]
