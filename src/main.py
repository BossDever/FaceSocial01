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

# Import routers and inject services
from src.api.complete_endpoints import (  # noqa: E402
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


async def _initialize_services_async(settings: Any) -> bool:
    """Async helper function to initialize all services."""
    try:
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
        logger.info("✅ VRAM Manager initialized")

        # Initialize Face Detection Service
        face_detection_service = FaceDetectionService(
            vram_manager=vram_manager, config=settings.detection_config
        )

        detection_init = await face_detection_service.initialize()
        if detection_init:
            services["face_detection"] = face_detection_service
            logger.info("✅ Face Detection Service initialized")
            # Inject service into router
            face_detection_router.face_detection_service = face_detection_service
        else:
            logger.error("❌ Failed to initialize Face Detection Service")
            return False

        # Initialize Face Recognition Service
        face_recognition_service = FaceRecognitionService(
            vram_manager=vram_manager, config=settings.recognition_config
        )

        recognition_init = await face_recognition_service.initialize()
        if recognition_init:
            services["face_recognition"] = face_recognition_service
            logger.info("✅ Face Recognition Service initialized")
            # Inject service into router
            face_recognition_router.face_recognition_service = face_recognition_service
        else:
            logger.error("❌ Failed to initialize Face Recognition Service")
            return False

        # Initialize Face Analysis Service (Integration)
        if "face_detection" in services and "face_recognition" in services:
            face_analysis_service = FaceAnalysisService(
                vram_manager=vram_manager, config=settings.analysis_config
            )

            analysis_init = await face_analysis_service.initialize()
            if analysis_init:
                services["face_analysis"] = face_analysis_service
                logger.info("✅ Face Analysis Service initialized")
                # Inject service into router
                face_analysis_router.face_analysis_service = face_analysis_service
            else:
                logger.error("❌ Failed to initialize Face Analysis Service")
                return False

        return True

    except Exception as e:
        logger.error(f"❌ Error in service initialization: {e}", exc_info=True)
        return False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager"""
    logger.info("🚀 Starting Face Recognition System...")
    settings = get_settings()

    try:
        # Initialize services asynchronously
        init_success = await _initialize_services_async(settings)

        if init_success:
            logger.info("🎉 All services initialized successfully!")
        else:
            logger.error("❌ Some services failed to initialize")
            # You can decide whether to continue or raise an exception
            # For now, we'll continue to allow the app to start

        yield

    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}", exc_info=True)
        yield  # Continue even if there are initialization errors

    # Shutdown
    logger.info("🛑 Shutting down Face Recognition System...")
    for service_name, service_instance in services.items():
        try:
            if hasattr(service_instance, "cleanup"):
                if hasattr(service_instance.cleanup, "__call__"):
                    await service_instance.cleanup()
                logger.info(f"✅ {service_name} cleaned up")
        except Exception as e:
            logger.error(f"❌ Error cleaning up {service_name}: {e}")

    services.clear()
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

    # Include API routers
    app.include_router(face_detection_router, prefix="/api", tags=["Face Detection"])
    app.include_router(
        face_recognition_router, prefix="/api", tags=["Face Recognition"]
    )
    app.include_router(face_analysis_router, prefix="/api", tags=["Face Analysis"])

    # Root endpoint redirect
    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/docs")

    @app.get("/health", tags=["System"])
    async def health_check():
        """System health check"""
        active_services = {}
        for name, service in services.items():
            if service is not None:
                active_services[name] = "active"
            else:
                active_services[name] = "inactive"

        return {
            "status": "healthy" if services else "degraded",
            "project_name": settings.project_name,
            "version": settings.version,
            "active_services": active_services,
            "total_services": len(services),
        }

    @app.get("/system/info", tags=["System"])
    async def system_info():
        """Detailed system information"""
        try:
            vram_manager = services.get("vram_manager")
            vram_status = await vram_manager.get_vram_status() if vram_manager else {}

            return {
                "system": {
                    "project": settings.project_name,
                    "version": settings.version,
                    "services_loaded": len(services),
                },
                "vram_status": vram_status,
                "services": {
                    name: "active" if service else "inactive"
                    for name, service in services.items()
                },
            }
        except Exception as e:
            return {"error": f"Failed to get system info: {e}"}

    logger.info(
        f"🚀 Application setup complete. Will listen on http://{settings.host}:{settings.port}"
    )
    return app


app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "src.main:app",  # Point to the app instance
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level,
    )
