#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face Recognition System - Main Application
Fixed version with better import handling
"""

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
# Consolidate typing imports here, removing unused ones
from typing import Any, AsyncGenerator, Dict, List

import asyncio

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Third-party imports
import uvicorn
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.cors import CORSMiddleware

# Local imports
try:
    from src.core.config import get_settings
    from src.ai_services.common.vram_manager import VRAMManager
    from src.ai_services.face_detection.face_detection_service import (
        FaceDetectionService
    )
    from src.ai_services.face_recognition.face_recognition_service import (
        FaceRecognitionService
    )
    from src.ai_services.face_analysis.face_analysis_service import (
        FaceAnalysisService
    )
    from src.api.complete_endpoints import (
        face_detection_router,
        face_recognition_router,
        face_analysis_router,
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

# Setup logging
def setup_logging() -> None: # Added -> None
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/app.log")
        ]
    ) # Closed parenthesis here

setup_logging() # Moved call to after function definition
logger = logging.getLogger(__name__)

async def validation_exception_handler_custom(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle validation errors without trying to decode binary data"""
    try:
        # Create a safe version of error details
        safe_errors = []
        for error in exc.errors():
            safe_error = {}
            for key, value in error.items():
                if isinstance(value, bytes):
                    safe_error[key] = f"<bytes: {len(value)} bytes>"
                elif hasattr(value, '__iter__') and not isinstance(value, str):
                    # Handle lists/tuples that might contain bytes
                    safe_items = []
                    for item in value:
                        if isinstance(item, bytes):
                            safe_items.append(f"<bytes: {len(item)} bytes>")
                        else:
                            safe_items.append(str(item)) # Ensure all items are strings
                    safe_error[key] = ", ".join(safe_items) # Join list into a string
                else:
                    safe_error[key] = str(value) # Ensure value is a string
            safe_errors.append(safe_error)
        
        return JSONResponse(
            status_code=422,
            content={"detail": safe_errors}
        )
    except Exception as e:
        logger.error(f"Error in validation exception handler: {e}")
        return JSONResponse(
            status_code=422,
            content={"detail": "Validation error occurred"}
        )

# Helper function to sanitize data for jsonable_encoder
def sanitize_for_jsonable_encoder(data: Any) -> Any:
    if isinstance(data, bytes):
        return data.decode('utf-8', errors='replace')
    elif isinstance(data, str):
        # Encode to bytes and then decode back to string with error replacement
        # This helps clean up strings that might contain invalid byte sequences
        # represented as characters (e.g., from incorrect prior decoding steps)
        return data.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    elif isinstance(data, list):
        return [sanitize_for_jsonable_encoder(item) for item in data]
    elif isinstance(data, dict):
        return {
            # Sanitize keys as well, in case they are problematic (though less common)
            sanitize_for_jsonable_encoder(k): sanitize_for_jsonable_encoder(v)
            for k, v in data.items()
        }
    return data

async def _fastapi_native_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Custom exception handler for request validation errors."""
    
    original_errors = exc.errors()
    original_body = exc.body

    sanitized_errors = sanitize_for_jsonable_encoder(original_errors)
    sanitized_body = sanitize_for_jsonable_encoder(original_body)

    logger.warning(
        f"Validation error for request {request.url}. "
        f"Sanitized errors: {sanitized_errors}. "
        f"Body type after sanitization: {type(sanitized_body)}"
    )
    # Avoid logging full body if it could be large or sensitive.
    # If further debugging of body content is needed,
    # consider logging a snippet or specific parts.

    return JSONResponse(
        status_code=422, # HTTP 422 Unprocessable Entity
        content=jsonable_encoder({
            "detail": sanitized_errors,
            "body": sanitized_body # Ensure this is serializable
        })
    )

# Service names constant
VRAM_MANAGER_SERVICE = "vram_manager"
FACE_DETECTION_SERVICE = "face_detection_service"
FACE_RECOGNITION_SERVICE = "face_recognition_service"
FACE_ANALYSIS_SERVICE = "face_analysis_service"

ALL_SERVICE_KEYS: List[str] = [
    VRAM_MANAGER_SERVICE,
    FACE_DETECTION_SERVICE,
    FACE_RECOGNITION_SERVICE,
    FACE_ANALYSIS_SERVICE,
]
USER_FACING_SERVICE_KEYS: List[str] = [
    FACE_DETECTION_SERVICE,
    FACE_RECOGNITION_SERVICE,
    FACE_ANALYSIS_SERVICE,
]

async def _initialize_single_service(
    app: FastAPI,
    service_name: str,
    service_class: Any,
    vram_manager: VRAMManager,
    config: Any
) -> bool:
    logger.info(f"🔧 Initializing {service_class.__name__}...")
    service_instance = service_class(vram_manager=vram_manager, config=config)
    if await service_instance.initialize():
        setattr(app.state, service_name, service_instance)
        logger.info(f"✅ {service_class.__name__} initialized")
        return True
    logger.error(f"❌ {service_class.__name__} initialization failed")
    return False

async def initialize_services(app: FastAPI, settings: Any) -> bool:
    """Initialize all services and attach to app.state"""
    try:
        os.makedirs("logs", exist_ok=True)
        os.makedirs("output/detection", exist_ok=True)
        os.makedirs("output/recognition", exist_ok=True)
        os.makedirs("output/analysis", exist_ok=True)
        os.makedirs("temp", exist_ok=True)

        logger.info("🔧 Initializing VRAM Manager...")
        vram_manager = VRAMManager(settings.vram_config)
        setattr(app.state, VRAM_MANAGER_SERVICE, vram_manager)
        logger.info("✅ VRAM Manager initialized")

        # Initialize core services
        core_services_to_init = [
            (
                FACE_DETECTION_SERVICE,
                FaceDetectionService,
                settings.detection_config
            ),
            (
                FACE_RECOGNITION_SERVICE,
                FaceRecognitionService,
                settings.recognition_config
            ),
        ]
        for name, cls, cfg in core_services_to_init:
            if not await _initialize_single_service(app, name, cls, vram_manager, cfg):
                return False # Critical failure if a core service fails

        # Initialize Face Analysis Service (dependent on others)
        if all(hasattr(app.state, key) for key in [
            VRAM_MANAGER_SERVICE, FACE_DETECTION_SERVICE, FACE_RECOGNITION_SERVICE
        ]): # Shortened line
            if not await _initialize_single_service(
                app, FACE_ANALYSIS_SERVICE, FaceAnalysisService,
                vram_manager, settings.analysis_config
            ):
                logger.warning(
                    "Face Analysis Service failed to initialize, proceeding without it."
                )
                # Not returning False, as it might be non-critical
        else:
            logger.warning(
                "⚠️ Face Analysis Service not initialized due to missing dependencies."
            )

        initialized_count = sum(
            1 for key in ALL_SERVICE_KEYS if hasattr(app.state, key)
        )
        logger.info(f"🎉 {initialized_count} services initialized successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Service initialization error: {e}", exc_info=True)
        return False

async def _cleanup_single_service(app: FastAPI, service_name: str) -> None:
    """Helper function to cleanup a single service."""
    service = getattr(app.state, service_name, None)
    if service and hasattr(service, 'cleanup') and callable(service.cleanup):
        try:
            coro = service.cleanup()
            if asyncio.iscoroutine(coro):
                # Await coro or create task if needed for gather elsewhere
                await coro
                logger.info(f"Async cleanup for {service_name} completed.")
            else:
                logger.info(f"Synchronous cleanup for {service_name} completed.")
            logger.info(f"✅ {service_name} cleanup initiated/completed.")
        except Exception as e:
            logger.error(f"❌ Error during {service_name} cleanup: {e}")

    if hasattr(app.state, service_name):
        delattr(app.state, service_name)
        logger.info(f"✅ {service_name} cleared from app.state.")

async def _cleanup_services(app: FastAPI) -> None:
    logger.info("🛑 Shutting down services...")
    # Cleanup in reverse order of initialization
    for service_name in reversed(ALL_SERVICE_KEYS):
        await _cleanup_single_service(app, service_name)
    logger.info("✅ All services cleanup attempted. Shutdown complete.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for service initialization and cleanup."""
    logger.info("🚀 Starting Face Recognition System...")
    settings = get_settings()

    try:
        if await initialize_services(app, settings):
            logger.info("✅ Startup complete - ready to serve requests")
        else:
            logger.error("❌ Startup failed - some services unavailable")
        yield
    except Exception as e:
        logger.error(f"❌ Lifespan startup error: {e}", exc_info=True)
        yield # Ensure yield for cleanup even if startup fails
    finally:
        await _cleanup_services(app)

def _register_routers(app: FastAPI):
    """Helper function to register API routers."""
    app.include_router(face_detection_router, prefix="/api", tags=["Face Detection"])
    app.include_router(
        face_recognition_router, prefix="/api", tags=["Face Recognition"]
    )
    app.include_router(face_analysis_router, prefix="/api", tags=["Face Analysis"])

def _root_redirect() -> RedirectResponse:
    """Redirects the root path to the API documentation."""
    return RedirectResponse(url="/docs")

async def _health_check_handler(request: Request) -> Dict[str, Any]:
    """Handles the health check endpoint."""
    s = get_settings()
    active_services = []
    for key in USER_FACING_SERVICE_KEYS:
        if hasattr(request.app.state, key) and \
           getattr(request.app.state, key) is not None:
            active_services.append(key.replace("_service", ""))

    return {
        "status": "healthy" if active_services else "degraded",
        "project": s.project_name,
        "version": s.version,
        "services": dict.fromkeys(active_services, "active"),
        "total_services": len(active_services)
    }

async def _system_info_handler(request: Request) -> Dict[str, Any]:
    """Handles the system information endpoint."""
    s = get_settings()
    try:
        vram_manager = getattr(request.app.state, VRAM_MANAGER_SERVICE, None)
        vram_status = {}
        if vram_manager and hasattr(vram_manager, "get_vram_status"):
            vram_status = await vram_manager.get_vram_status()

        loaded_services = []
        for key in ALL_SERVICE_KEYS:
            if hasattr(request.app.state, key) and \
               getattr(request.app.state, key) is not None:
                loaded_services.append(key.replace("_service", ""))

        return {
            "system": {
                "project": s.project_name,
                "version": s.version,
                "services_loaded": len(loaded_services)
            },
            "vram_status": vram_status,
            "services": loaded_services
        }
    except Exception as e:
        logger.error(f"System info error: {e}", exc_info=True)
        return {"error": f"System info error: {str(e)}"}

def _register_event_handlers(app: FastAPI):
    """Helper function to register event handlers (like root and health)."""
    app.get("/", include_in_schema=False)(_root_redirect)
    app.get("/health", tags=["System"])(_health_check_handler)
    app.get("/system/info", tags=["System"])(_system_info_handler)
    app.exception_handler(RequestValidationError)(
        _fastapi_native_validation_exception_handler
    )

def create_app() -> FastAPI:
    """Create FastAPI application."""
    current_settings = get_settings()

    app = FastAPI(
        title=current_settings.project_name,
        version=current_settings.version,
        description=current_settings.description,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=current_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_routers(app)
    _register_event_handlers(app)
    # Note: The custom validation_exception_handler_custom is defined
    # but not registered by default. The fastapi_native_validation_exception_handler
    # is registered by _register_event_handlers.

    return app

app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    logger.info(f"🚀 Starting server on {settings.host}:{settings.port}")
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level,
    )
