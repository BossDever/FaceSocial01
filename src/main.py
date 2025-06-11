#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face Recognition System - Main Application
SuwitBoss/wofk - Clean & Organized Version
"""

# Standard library imports
import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any, AsyncGenerator

# Third-party imports
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# Add project root to Python path FIRST
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Local imports (after sys.path modification) - noqa: E402
from src.core.config import get_settings  # noqa: E402
from src.ai_services.common.vram_manager import VRAMManager  # noqa: E402
from src.ai_services.face_detection.face_detection_service import (
    FaceDetectionService,
)  # noqa: E402
from src.ai_services.face_recognition.face_recognition_service import (
    FaceRecognitionService,
)  # noqa: E402
from src.ai_services.face_analysis.face_analysis_service import (
    FaceAnalysisService,
)  # noqa: E402
# Import routers at the top level
from src.api.complete_endpoints import ( # Updated import
    face_detection_router,
    face_recognition_router,
    face_analysis_router,
)  # noqa: E402

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/app.log")],
)

logger = logging.getLogger(__name__)

# Global service instances
services: Dict[str, Any] = {}


def _initialize_services(settings: Any) -> None:
    """Helper function to initialize all services."""
    # Create necessary directories
    for directory in [
        "logs",
        "output",
        "output/detection",
        "output/recognition",
        "output/analysis",
        "temp",
    ]:
        os.makedirs(directory, exist_ok=True)

    # Initialize VRAM Manager
    vram_manager = VRAMManager(settings.vram_config)
    services["vram_manager"] = vram_manager

    # Initialize Face Detection Service
    face_detection_service = FaceDetectionService(
        vram_manager=vram_manager, config=settings.detection_config
    )
    if face_detection_service.initialize_sync():  # Assuming a sync version for startup
        services["face_detection"] = face_detection_service
        logger.info("✅ Face Detection Service initialized")
        face_detection_router.face_detection_service = face_detection_service
    else:
        logger.error("❌ Failed to initialize Face Detection Service")

    # Initialize Face Recognition Service
    face_recognition_service = FaceRecognitionService(
        vram_manager=vram_manager, config=settings.recognition_config
    )
    if face_recognition_service.initialize_sync():  # Assuming a sync version
        services["face_recognition"] = face_recognition_service
        logger.info("✅ Face Recognition Service initialized")
        face_recognition_router.face_recognition_service = face_recognition_service
    else:
        logger.error("❌ Failed to initialize Face Recognition Service")

    # Initialize Face Analysis Service (Integration)
    if "face_detection" in services and "face_recognition" in services:
        face_analysis_service = FaceAnalysisService(
            vram_manager=vram_manager, config=settings.analysis_config
        )
        if face_analysis_service.initialize_sync():  # Assuming a sync version
            services["face_analysis"] = face_analysis_service
            logger.info("✅ Face Analysis Service initialized")
            face_analysis_router.face_analysis_service = face_analysis_service
        else:
            logger.error("❌ Failed to initialize Face Analysis Service")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager"""
    logger.info("🚀 Starting Face Recognition System...")
    settings = get_settings()
    try:
        # Using a synchronous helper for setup that doesn't need async
        _initialize_services(settings)
        logger.info("🎉 All services initialized successfully!")
        yield
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        # Decide if you want to yield even on failure, or raise to stop app
        yield  # Current behavior: app starts even if services fail

    # Shutdown
    logger.info("🛑 Shutting down Face Recognition System...")
    for service_name, service_instance in services.items():
        try:
            if hasattr(service_instance, "cleanup_sync"):  # Assuming sync cleanup
                service_instance.cleanup_sync()
            elif hasattr(service_instance, "cleanup"):  # Fallback to async if defined
                await service_instance.cleanup()
            logger.info(f"✅ {service_name} cleaned up")
        except Exception as e:
            logger.error(f"❌ Error cleaning up {service_name}: {e}")
    logger.info("✅ Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    # Initialize FastAPI app with lifespan manager
    app = FastAPI(
        title=settings.project_name,
        version=settings.version,
        description=settings.description,
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files directory (optional)
    # os.makedirs("static", exist_ok=True)
    # app.mount("/static", StaticFiles(directory="static"), name="static")

    # Include API routers
    app.include_router(face_detection_router, prefix="/api", tags=["Face Detection"])
    app.include_router(
        face_recognition_router, prefix="/api", tags=["Face Recognition"]
    )
    app.include_router(face_analysis_router, prefix="/api", tags=["Face Analysis"])

    # Root endpoint redirect or welcome message
    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/docs")

    @app.get("/health", tags=["System"])
    async def health_check():
        """System health check"""
        # Basic health check, can be expanded
        active_services = {
            name: "active" for name in services if services.get(name) is not None
        }
        return {
            "status": "healthy",
            "project_name": settings.project_name,
            "version": settings.version,
            "active_services": active_services,
        }

    logger.info(
        f"🚀 Application setup complete. Listening on http://{settings.host}:{settings.port}"
    )
    return app


app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "src.main:app",  # Point to the app instance
        host=settings.host,
        port=settings.port,
        reload=settings.debug,  # Enable reload only in debug mode
        log_level=settings.log_level.lower(),
    )
