"""
FastAPI application entry point for Rad-Intel Clinical Decision Support Framework.
"""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rad_intel.config import settings
from rad_intel.storage.database import db_manager
from rad_intel.api.dependencies import model_manager
from rad_intel.api.routes import (
    health_router,
    predict_router,
    explain_router,
    report_router,
    analyze_router,
    history_router,
)

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rad_intel")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes audit database and pre-loads default deep learning model on startup."""
    logger.info("Initializing Rad-Intel Database...")
    await db_manager.init_db()

    logger.info("Pre-warming default model: %s on device: %s...", settings.DEFAULT_MODEL, settings.torch_device)
    try:
        model_manager.get_model(settings.DEFAULT_MODEL)
        logger.info("Default model loaded successfully.")
    except Exception as e:
        logger.warning("Could not pre-warm model on startup: %s. Will load lazily on first request.", e)

    yield
    logger.info("Rad-Intel server shutting down.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Data-Driven Clinical Decision Support Framework for Automated Pneumonia Detection from Chest Radiographs",
    lifespan=lifespan,
)

from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# CORS Configuration for frontend / client applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Route Handlers
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(explain_router)
app.include_router(report_router)
app.include_router(analyze_router)
app.include_router(history_router)

# Mount Static Assets and Sample Images
static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

sample_images_dir = settings.BASE_DIR / "sample_images"
if sample_images_dir.exists():
    app.mount("/sample_images", StaticFiles(directory=str(sample_images_dir)), name="sample_images")


@app.get("/", response_class=FileResponse)
async def serve_ui():
    """Serves the minimalistic, light-themed Rad-Intel web UI."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return FileResponse(settings.BASE_DIR / "README.md")


@app.get("/api")
async def api_info():
    """Returns API metadata and documentation endpoints."""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "health_check": "/api/v1/health",
        "description": "Rad-Intel Clinical Decision Support API.",
    }
