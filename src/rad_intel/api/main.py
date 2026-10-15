"""
FastAPI application entry point for the Rad-Intel research and education framework.
"""

from contextlib import asynccontextmanager
import logging
import os
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
    assistant_router,
    feedback_router,
    auth_router,
    admin_router,
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


# The interactive API documentation is a complete map of the service,
# administrator routes included. It is useful while building and it is of no
# use to a member of the public, so it is off unless asked for. Every route
# it described is still guarded; this removes the directory, not the locks.
_docs_enabled = os.environ.get("RAD_INTEL_ENABLE_DOCS", "").strip() in ("1", "true", "yes")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Research and education framework for automated pneumonia classification, explanation and evidence-constrained reporting from chest radiographs. Not for clinical use.",
    lifespan=lifespan,
    docs_url="/docs" if _docs_enabled else None,
    redoc_url="/redoc" if _docs_enabled else None,
    openapi_url="/openapi.json" if _docs_enabled else None,
)


@app.middleware("http")
async def security_headers(request, call_next):
    """
    Headers every response carries.

    None of these replace a check on the server; they close off the ways a
    browser can be talked into doing something with this page on somebody
    else's behalf.
    """
    response = await call_next(request)
    # The login form must not be loadable inside another site's frame, or a
    # clinician can be persuaded to click something they cannot see.
    response.headers.setdefault("X-Frame-Options", "DENY")
    # No guessing a type: an uploaded file served as an image stays an image.
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    # Do not hand the full URL, which can carry a record id, to another site.
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    # Nothing here needs a camera, a microphone or a location.
    response.headers.setdefault(
        "Permissions-Policy", "camera=(), microphone=(), geolocation=(), interest-cohort=()")
    # Scripts and styles come from this origin and the few CDNs the page
    # uses; frames come from nowhere. 'unsafe-inline' is still needed for the
    # inline handlers the original page was written with.
    response.headers.setdefault("Content-Security-Policy", "; ".join([
        "default-src 'self'",
        "img-src 'self' data: blob:",
        "script-src 'self' 'unsafe-inline'",
        "style-src 'self' 'unsafe-inline'",
        "connect-src 'self'",
        "frame-ancestors 'none'",
        "base-uri 'self'",
        "form-action 'self'",
    ]))
    return response

from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc: RequestValidationError):
    """
    Return 422 for a malformed request, without echoing the body.

    FastAPI's default handler puts the offending input in the response, and
    its JSON encoder decodes bytes as UTF-8. Posting a radiograph to a route
    that expects JSON therefore raised UnicodeDecodeError inside the error
    path itself and surfaced as a 500, reporting a server fault for what is
    a client mistake.
    """
    safe = []
    for err in exc.errors():
        trimmed = {k: v for k, v in err.items() if k != "input"}
        if "ctx" in trimmed:
            trimmed["ctx"] = {k: str(v) for k, v in trimmed["ctx"].items()}
        safe.append(trimmed)
    return JSONResponse(status_code=422, content={"detail": safe})


# Cross-origin access.
#
# This was allow_origins=["*"] together with allow_credentials=True. Starlette
# handles that pair by echoing back whichever Origin asked, which means any
# website could make credentialed requests to this API. SameSite=Lax keeps the
# session cookie off cross-site fetches, so it was not exploitable as it
# stood -- but it was one cookie attribute away from being so, and the
# interface is served from this same origin and needs no CORS at all.
#
# RAD_INTEL_CORS_ORIGINS adds specific origins, comma separated, for the case
# where a separate front end really is hosted elsewhere. There is no wildcard
# option: a credentialed wildcard is the bug this replaces.
_cors_origins = [
    origin.strip()
    for origin in os.environ.get("RAD_INTEL_CORS_ORIGINS", "").split(",")
    if origin.strip() and origin.strip() != "*"
]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    logger.info("Cross-origin requests allowed from: %s", ", ".join(_cors_origins))

# Register Route Handlers
app.include_router(health_router)
app.include_router(predict_router)
app.include_router(explain_router)
app.include_router(report_router)
app.include_router(analyze_router)
app.include_router(history_router)
app.include_router(assistant_router)
app.include_router(feedback_router)
app.include_router(auth_router)
app.include_router(admin_router)

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
        "description": "Rad-Intel research and education API. Not for clinical use.",
    }
