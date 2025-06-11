#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face Recognition System - Main Application
จุดเริ่มต้นของแอปพลิเคชัน Face Recognition System ที่แก้ไขแล้ว
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

# Import services
from src.core.vram_manager import VRAMManager
from src.services.face_detection_service import FaceDetectionService
from src.services.face_recognition_service import FaceRecognitionService
from src.services.face_analysis_service import FaceAnalysisService

# Import API routers
from src.api.face_detection_api import router as face_detection_router
from src.api.face_recognition_api import router as face_recognition_router
from src.api.face_analysis_api import router as face_analysis_router

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/face_recognition_system.log")
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Face Recognition System API",
    description="ระบบจดจำใบหน้าแบบครบวงจร รองรับการตรวจจับและจดจำใบหน้า",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances
vram_manager: VRAMManager = None
face_detection_service: FaceDetectionService = None
face_recognition_service: FaceRecognitionService = None
face_analysis_service: FaceAnalysisService = None

async def initialize_services():
    """Initialize all AI services"""
    global vram_manager, face_detection_service, face_recognition_service, face_analysis_service
    
    try:
        logger.info("🚀 Initializing Face Recognition System...")
        
        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("output", exist_ok=True)
        os.makedirs("temp", exist_ok=True)
        
        # Initialize VRAM Manager
        vram_config = {
            "reserved_vram_mb": 512,
            "model_vram_estimates": {
                # Face Detection Models
                "yolov9c-face": 512 * 1024 * 1024,
                "yolov9e-face": 2048 * 1024 * 1024, 
                "yolov11m-face": 2 * 1024 * 1024 * 1024,
                
                # Face Recognition Models  
                "adaface": 260 * 1024 * 1024,
                "arcface": 260 * 1024 * 1024,
                "facenet": 94 * 1024 * 1024,
            }
        }
        
        vram_manager = VRAMManager(vram_config)
        logger.info("✅ VRAM Manager initialized")
        
        # Initialize Face Detection Service
        detection_config = {
            'yolov9c_model_path': 'model/face-detection/yolov9c-face-lindevs.onnx',
            'yolov9e_model_path': 'model/face-detection/yolov9e-face-lindevs.onnx',
            'yolov11m_model_path': 'model/face-detection/yolov11m-face.pt',
            'conf_threshold': 0.5,
            'iou_threshold': 0.4,
            'min_quality_threshold': 40,
        }
        
        face_detection_service = FaceDetectionService(vram_manager, detection_config)
        detection_init = await face_detection_service.initialize()
        
        if not detection_init:
            logger.error("❌ Failed to initialize face detection service")
            return False
        
        logger.info("✅ Face Detection Service initialized")
        
        # Initialize Face Recognition Service
        recognition_config = {
            'preferred_model': 'facenet',
            'similarity_threshold': 0.6,
            'enable_gpu_optimization': True,
        }
        
        face_recognition_service = FaceRecognitionService(vram_manager, recognition_config)
        recognition_init = await face_recognition_service.initialize()
        
        if not recognition_init:
            logger.error("❌ Failed to initialize face recognition service")
            return False
        
        logger.info("✅ Face Recognition Service initialized")
        
        # Initialize Face Analysis Service (Integration Service)
        analysis_config = {
            "detection": detection_config,
            "recognition": recognition_config
        }
        
        face_analysis_service = FaceAnalysisService(
            vram_manager, 
            face_detection_service, 
            face_recognition_service,
            analysis_config
        )
        
        analysis_init = await face_analysis_service.initialize()
        
        if not analysis_init:
            logger.error("❌ Failed to initialize face analysis service")
            return False
        
        logger.info("✅ Face Analysis Service initialized")
        
        # Initialize API routers with services
        face_detection_router.face_detection_service = face_detection_service
        face_recognition_router.face_recognition_service = face_recognition_service
        face_analysis_router.face_analysis_service = face_analysis_service
        face_analysis_router.vram_manager = vram_manager
        
        logger.info("🎉 All services initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}", exc_info=True)
        return False

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    success = await initialize_services()
    if not success:
        logger.error("❌ Failed to start application - exiting")
        exit(1)

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    try:
        logger.info("🛑 Shutting down Face Recognition System...")
        
        if face_analysis_service:
            await face_analysis_service.cleanup()
            
        if face_recognition_service:
            await face_recognition_service.cleanup()
            
        if face_detection_service:
            await face_detection_service.cleanup()
        
        logger.info("✅ System shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

# Include API routers
app.include_router(face_detection_router, prefix="/api")
app.include_router(face_recognition_router, prefix="/api")
app.include_router(face_analysis_router, prefix="/api")

# Mount static files
app.mount("/output", StaticFiles(directory="output"), name="output")

@app.get("/")
async def root():
    """Root endpoint - redirect to docs"""
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "services": {
                "face_detection": face_detection_service is not None,
                "face_recognition": face_recognition_service is not None,
                "face_analysis": face_analysis_service is not None,
                "vram_manager": vram_manager is not None
            },
            "version": "2.0.0"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "version": "2.0.0"
        }

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8080, 
        reload=True,
        log_level="info"
    )