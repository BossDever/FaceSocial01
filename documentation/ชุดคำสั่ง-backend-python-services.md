# ชุดคำสั่ง: Backend Python Services
## การพัฒนาและจัดการบริการ Backend ด้วย Python

### 📋 สารบัญ
1.1 [ภาพรวมระบบ Backend](#11-ภาพรวมระบบ-backend)
1.2 [การติดตั้งและเริ่มต้นระบบ](#12-การติดตั้งและเริ่มต้นระบบ)
1.3 [FastAPI และ API Endpoints](#13-fastapi-และ-api-endpoints)
1.4 [AI Services Integration](#14-ai-services-integration)
1.5 [Dependency Injection](#15-dependency-injection)
1.6 [การจัดการ Database](#16-การจัดการ-database)
1.7 [Performance และ Monitoring](#17-performance-และ-monitoring)
1.8 [Testing และ Debugging](#18-testing-และ-debugging)

---

## 1.1 ภาพรวมระบบ Backend

ระบบ Backend ที่พัฒนาด้วย Python, FastAPI และ AI Services สำหรับ Face Recognition และ Social Media Platform

### 🏗️ สถาปัตยกรรม Backend
- **API Layer**: FastAPI REST endpoints และ WebSocket
- **AI Services**: Face Detection, Recognition, Anti-Spoofing  
- **Database**: PostgreSQL และ Redis caching
- **Infrastructure**: VRAM management และ monitoring

---

## 1.2 การติดตั้งและเริ่มต้นระบบ

### 1.2.1 การติดตั้งและเริ่มต้นระบบ

```python
#!/usr/bin/env python3
"""
Start script for Face Recognition System
"""
import os
import sys
import asyncio
import argparse
from pathlib import Path
import logging

# Setup logging configuration
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/startup.log", encoding='utf-8')
        ]
    )
    
    return logging.getLogger(__name__)

# Check system requirements
def check_system_requirements() -> bool:
    logger = logging.getLogger(__name__)
    
    if sys.version_info < (3, 8):
        logger.error(f"❌ Python 3.8+ required, found {sys.version}")
        return False
    
    critical_modules = [
        ("fastapi", "FastAPI framework"),
        ("uvicorn", "ASGI server"),
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("onnxruntime", "ONNX Runtime")
    ]
    
    missing_modules = []
    for module, description in critical_modules:
        try:
            __import__(module)
            logger.info(f"✅ {description}")
        except ImportError:
            logger.error(f"❌ {description} ({module}) not found")
            missing_modules.append(module)
    
    return len(missing_modules) == 0
```

### 1.3.1 ชุดคำสั่งการเริ่มต้น FastAPI Application

```python
#!/usr/bin/env python3
"""
Face Recognition System - Main Application
"""
import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List

# Third-party imports
import uvicorn
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

# Initialize logging
def setup_logging() -> None:
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

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown"""
    logger.info("🚀 Starting Face Recognition System...")
    
    try:
        # Initialize services
        await initialize_services()
        logger.info("✅ All services initialized successfully")
        yield
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise
    finally:
        logger.info("🛑 Shutting down Face Recognition System...")
        await cleanup_services()

# Create FastAPI application
app = FastAPI(
    title="Face Recognition System API",
    description="Professional Face Detection, Recognition & Analysis System",
    version="2.0.0",
    lifespan=lifespan
)
```

## 1.3 FastAPI และ API Endpoints

### 1.3.2 ชุดคำสั่ง Face Detection Models และ Configuration

```python
"""
Face Detection API Router
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional, Dict, Any, List
import cv2
import numpy as np
from pydantic import BaseModel
import base64
import logging

# Detection Request Model
class DetectionRequest(BaseModel):
    image_base64: str
    model_name: Optional[str] = "auto"
    conf_threshold: Optional[float] = 0.5
    iou_threshold: Optional[float] = 0.4
    max_faces: Optional[int] = 50
    min_quality_threshold: Optional[float] = 40.0

# Detection Configuration
class DetectionConfig(BaseModel):
    model_name: str = "auto"
    conf_threshold: float = 0.5
    iou_threshold: float = 0.4
    max_faces: int = 50
    min_quality_threshold: float = 40.0
    return_landmarks: bool = False
    use_fallback: bool = True

# Decode base64 image
def decode_base64_image(image_base64: str) -> np.ndarray:
    try:
        image_data = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
```

### 1.3.3 ชุดคำสั่ง Face Detection API Endpoints

```python
router = APIRouter()

# Basic face detection endpoint
@router.post("/detect", response_model=Dict[str, Any])
async def detect_faces(request: DetectionRequest):
    """
    ตรวจจับใบหน้าในภาพด้วย YOLO models
    """
    try:
        # Decode image
        image = decode_base64_image(request.image_base64)
        
        # Get face detection service
        detection_service = get_face_detection_service()
        
        # Configure detection parameters
        config = {
            "model_name": request.model_name,
            "conf_threshold": request.conf_threshold,
            "iou_threshold": request.iou_threshold,
            "max_faces": request.max_faces,
            "min_quality_threshold": request.min_quality_threshold
        }
        
        # Perform detection
        result = await detection_service.detect_faces(image, config)
        
        return {
            "success": True,
            "faces_detected": len(result.get("faces", [])),
            "faces": result.get("faces", []),
            "model_used": result.get("model_used", "unknown"),
            "processing_time": result.get("processing_time", 0),
            "image_info": {
                "width": image.shape[1],
                "height": image.shape[0],
                "channels": image.shape[2] if len(image.shape) > 2 else 1
            }
        }
        
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Upload file detection endpoint
@router.post("/detect-upload", response_model=Dict[str, Any])
async def detect_faces_upload(
    file: UploadFile = File(...),
    model_name: str = Form("auto"),
    conf_threshold: float = Form(0.5),
    max_faces: int = Form(50)
):
    """
    ตรวจจับใบหน้าในภาพที่อัปโหลด
    """
    try:
        # Validate file format
        if not validate_image_format(file):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file format. Supported: JPG, PNG, BMP, TIFF, WEBP"
            )
        
        # Read and decode image
        contents = await file.read()
        image = decode_uploaded_image(contents)
        
        # Get detection service
        detection_service = get_face_detection_service()
        
        # Configure detection
        config = create_detection_config(
            model_name=model_name,
            conf_threshold=conf_threshold,
            max_faces=max_faces
        )
        
        # Perform detection
        result = await detection_service.detect_faces(image, config)
        
        return {
            "success": True,
            "filename": file.filename,
            "faces_detected": len(result.get("faces", [])),
            "faces": result.get("faces", []),
            "model_used": result.get("model_used"),
            "processing_time": result.get("processing_time")
        }
        
    except Exception as e:
        logger.error(f"Upload detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

## 1.4 AI Services Integration

### 1.4.1 ชุดคำสั่ง Face Recognition Models และ Requests

```python
"""
Face Recognition API Router
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, Dict, Any, List, Union
import cv2
import numpy as np
from pydantic import BaseModel
import base64
import logging
import time
import uuid

# Add Face Request Model
class AddFaceRequest(BaseModel):
    person_name: str
    person_id: Optional[str] = None
    face_image_base64: str
    model_name: Optional[str] = "facenet"
    metadata: Optional[Dict[str, Any]] = None
    fast_mode: Optional[bool] = False
    processing_mode: Optional[str] = None

# Recognition Request Model
class RecognitionRequest(BaseModel):
    face_image_base64: str
    gallery: Optional[Dict[str, Any]] = None
    model_name: Optional[str] = "facenet"
    top_k: Optional[int] = 5
    similarity_threshold: Optional[float] = 0.5
    processing_mode: Optional[str] = None

# Compare Request Model
class CompareRequest(BaseModel):
    face1_image_base64: str
    face2_image_base64: str
    model_name: Optional[str] = "facenet"

# Multiple Registration Request
class MultipleRegistrationRequest(BaseModel):
    full_name: str
    employee_id: str
    department: Optional[str] = None
    position: Optional[str] = None  
    model_name: Optional[str] = "adaface"
    images: List[str]  # List of base64 encoded images
    metadata: Optional[Dict[str, Any]] = None
```

### 1.4.2 ชุดคำสั่ง Face Recognition API Endpoints

```python
router = APIRouter()

# Add face to gallery endpoint
@router.post("/add-face", response_model=Dict[str, Any])
async def add_face_to_gallery(request: AddFaceRequest):
    """
    เพิ่มใบหน้าใหม่เข้าสู่ระบบจดจำใบหน้า
    """
    try:
        # Decode image
        image = decode_base64_image(request.face_image_base64)
        
        # Generate person ID if not provided
        person_id = request.person_id or str(uuid.uuid4())
        
        # Get face recognition service
        recognition_service = get_face_recognition_service()
        
        # Add face to gallery
        result = await recognition_service.add_face(
            image=image,
            person_name=request.person_name,
            person_id=person_id,
            model_name=request.model_name,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "person_id": person_id,
            "person_name": request.person_name,
            "embedding_extracted": result.get("embedding_extracted", False),
            "model_used": result.get("model_used"),
            "face_quality": result.get("face_quality"),
            "processing_time": result.get("processing_time")
        }
        
    except Exception as e:
        logger.error(f"Add face failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Face recognition endpoint
@router.post("/recognize", response_model=Dict[str, Any])
async def recognize_face(request: RecognitionRequest):
    """
    จดจำใบหน้าและค้นหาในฐานข้อมูล
    """
    try:
        # Decode image
        image = decode_base64_image(request.face_image_base64)
        
        # Get recognition service
        recognition_service = get_face_recognition_service()
        
        # Perform recognition
        result = await recognition_service.recognize_face(
            image=image,
            gallery=request.gallery,
            model_name=request.model_name,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        return {
            "success": True,
            "matches_found": len(result.get("matches", [])),
            "matches": result.get("matches", []),
            "best_match": result.get("best_match"),
            "confidence_scores": result.get("confidence_scores", []),
            "model_used": result.get("model_used"),
            "processing_time": result.get("processing_time")
        }
        
    except Exception as e:
        logger.error(f"Face recognition failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Compare two faces endpoint
@router.post("/compare", response_model=Dict[str, Any])
async def compare_faces(request: CompareRequest):
    """
    เปรียบเทียบใบหน้าสองภาพ
    """
    try:
        # Decode images
        face1 = decode_base64_image(request.face1_image_base64)
        face2 = decode_base64_image(request.face2_image_base64)
        
        # Get recognition service
        recognition_service = get_face_recognition_service()
        
        # Compare faces
        result = await recognition_service.compare_faces(
            face1=face1,
            face2=face2,
            model_name=request.model_name
        )
        
        return {
            "success": True,
            "is_same_person": result.get("is_same_person", False),
            "similarity_score": result.get("similarity_score", 0.0),
            "confidence": result.get("confidence", 0.0),
            "model_used": result.get("model_used"),
            "processing_time": result.get("processing_time")
        }
        
    except Exception as e:
        logger.error(f"Face comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

## 1.5 Dependency Injection

### 1.5.1 ชุดคำสั่งการเริ่มต้นและจัดการ Models

```python
"""
AI Services Dependency Injection
"""
import logging
from typing import Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# Global service instances
_face_detection_service: Optional[Any] = None
_face_recognition_service: Optional[Any] = None
_anti_spoofing_service: Optional[Any] = None

async def initialize_services():
    """Initialize all AI services"""
    global _face_detection_service, _face_recognition_service, _anti_spoofing_service
    
    try:
        # Initialize Face Detection Service
        from src.ai_services.face_detection.enhanced_detector import EnhancedFaceDetector
        _face_detection_service = EnhancedFaceDetector()
        await _face_detection_service.initialize()
        logger.info("✅ Face Detection Service initialized")
        
        # Initialize Face Recognition Service
        from src.ai_services.face_recognition.face_recognition_service_enhanced import FaceRecognitionService
        _face_recognition_service = FaceRecognitionService()
        await _face_recognition_service.initialize()
        logger.info("✅ Face Recognition Service initialized")
        
        # Initialize Anti-Spoofing Service
        from src.ai_services.anti_spoofing.anti_spoofing_service import AntiSpoofingService
        _anti_spoofing_service = AntiSpoofingService()
        await _anti_spoofing_service.initialize()
        logger.info("✅ Anti-Spoofing Service initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

# Dependency functions
def get_face_detection_service():
    """Get face detection service instance"""
    if _face_detection_service is None:
        raise HTTPException(status_code=503, detail="Face detection service not initialized")
    return _face_detection_service

def get_face_recognition_service():
    """Get face recognition service instance"""
    if _face_recognition_service is None:
        raise HTTPException(status_code=503, detail="Face recognition service not initialized")
    return _face_recognition_service

def get_anti_spoofing_service():
    """Get anti-spoofing service instance"""
    if _anti_spoofing_service is None:
        raise HTTPException(status_code=503, detail="Anti-spoofing service not initialized")
    return _anti_spoofing_service
```

---

*เอกสารนี้แสดงชุดคำสั่งหลักสำหรับ Backend Python Services ของระบบการจดจำใบหน้า รวมถึงการตั้งค่า API endpoints และการจัดการ AI models*
