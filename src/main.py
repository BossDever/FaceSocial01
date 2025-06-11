#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face Recognition System - Main Application
SuwitBoss/wofk - Clean & Organized Version
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import core components
from src.core.config import get_settings
from src.core.vram_manager import VRAMManager
from src.services.face_detection_service import FaceDetectionService
from src.services.face_recognition_service import FaceRecognitionService
from src.services.face_analysis_service import FaceAnalysisService

# Import API routes
from src.api.face_detection import router as face_detection_router
from src.api.face_recognition import router as face_recognition_router
from src.api.face_analysis import router as face_analysis_router

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/app.log")
    ]
)

logger = logging.getLogger(__name__)

# Global service instances
services = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    
    # Startup
    logger.info("🚀 Starting Face Recognition System...")
    
    try:
        # Get settings
        settings = get_settings()
        
        # Create necessary directories
        for directory in ["logs", "output", "output/detection", "output/recognition"]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize VRAM Manager
        vram_manager = VRAMManager(settings.vram_config)
        services["vram_manager"] = vram_manager
        
        # Initialize Face Detection Service
        face_detection_service = FaceDetectionService(
            vram_manager=vram_manager,
            config=settings.detection_config
        )
        
        if await face_detection_service.initialize():
            services["face_detection"] = face_detection_service
            logger.info("✅ Face Detection Service initialized")
        else:
            logger.error("❌ Failed to initialize Face Detection Service")
        
        # Initialize Face Recognition Service
        face_recognition_service = FaceRecognitionService(
            vram_manager=vram_manager,
            config=settings.recognition_config
        )
        
        if await face_recognition_service.initialize():
            services["face_recognition"] = face_recognition_service
            logger.info("✅ Face Recognition Service initialized")
        else:
            logger.error("❌ Failed to initialize Face Recognition Service")
        
        # Initialize Face Analysis Service (Integration)
        if "face_detection" in services and "face_recognition" in services:
            face_analysis_service = FaceAnalysisService(
                detection_service=services["face_detection"],
                recognition_service=services["face_recognition"],
                config=settings.analysis_config
            )
            
            if await face_analysis_service.initialize():
                services["face_analysis"] = face_analysis_service
                logger.info("✅ Face Analysis Service initialized")
        
        # Inject services into API routers
        face_detection_router.service = services.get("face_detection")
        face_recognition_router.service = services.get("face_recognition")
        face_analysis_router.service = services.get("face_analysis")
        
        logger.info("🎉 All services initialized successfully!")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        yield
    
    # Shutdown
    logger.info("🛑 Shutting down Face Recognition System...")
    
    for service_name, service in services.items():
        try:
            if hasattr(service, 'cleanup'):
                await service.cleanup()
            logger.info(f"✅ {service_name} cleaned up")
        except Exception as e:
            logger.error(f"❌ Error cleaning up {service_name}: {e}")
    
    logger.info("✅ Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Face Recognition System API",
    description="Professional Face Detection, Recognition & Analysis System",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(face_detection_router, prefix="/api")
app.include_router(face_recognition_router, prefix="/api")
app.include_router(face_analysis_router, prefix="/api")

# Mount static files
app.mount("/output", StaticFiles(directory="output"), name="output")

@app.get("/")
async def root():
    """Root endpoint"""
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "services": {
                "face_detection": "face_detection" in services,
                "face_recognition": "face_recognition" in services,
                "face_analysis": "face_analysis" in services,
                "vram_manager": "vram_manager" in services
            },
            "version": "2.0.0"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "version": "2.0.0"
        }

@app.get("/system/info")
async def system_info():
    """System information endpoint"""
    try:
        vram_status = {}
        if "vram_manager" in services:
            vram_status = await services["vram_manager"].get_vram_status()
        
        return {
            "system": "Face Recognition System",
            "version": "2.0.0",
            "services_count": len(services),
            "vram_status": vram_status
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )