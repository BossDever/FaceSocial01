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
from typing import Any, AsyncGenerator, Dict, List
import asyncio # Added asyncio import

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Third-party imports
import uvicorn
from fastapi import FastAPI, Request # Added Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

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
    )

setup_logging()
logger = logging.getLogger(__name__)

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

    app.include_router(face_detection_router, prefix="/api", tags=["Face Detection"])
    app.include_router(
        face_recognition_router, prefix="/api", tags=["Face Recognition"]
    )
    app.include_router(face_analysis_router, prefix="/api", tags=["Face Analysis"])

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse: # Added -> RedirectResponse
        return RedirectResponse(url="/docs")

    @app.get("/health", tags=["System"])
    async def health_check(request: Request) -> Dict[str, Any]: # Used Dict from typing
        """System health check."""
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

    @app.get("/system/info", tags=["System"])
    async def system_info(request: Request) -> Dict[str, Any]: # Used Dict from typing
        """Detailed system information."""
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
