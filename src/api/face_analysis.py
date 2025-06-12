"""
Complete API Endpoints for Face Analysis System
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import cv2
import numpy as np
from pydantic import BaseModel
import base64

# === FACE DETECTION API ===
face_detection_router = APIRouter()
face_detection_service = None  # Will be injected by main.py

class DetectionRequest(BaseModel):
    image_base64: str
    model_name: Optional[str] = "auto"
    conf_threshold: Optional[float] = 0.5
    iou_threshold: Optional[float] = 0.4
    max_faces: Optional[int] = 50
    min_quality_threshold: Optional[float] = 40.0

@face_detection_router.get("/face-detection/health")
async def face_detection_health() -> Dict[str, Any]:
    """Health check for face detection service"""
    if face_detection_service is None:
        raise HTTPException(
            status_code=503, detail="Face detection service not available"
        )

    service_info = await face_detection_service.get_service_info()
    return {
        "status": "healthy",
        "service": "face_detection",
        "service_info": service_info
    }

@face_detection_router.post("/face-detection/detect")
async def detect_faces_endpoint(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form("auto"),
    conf_threshold: Optional[float] = Form(0.5),
    iou_threshold: Optional[float] = Form(0.4),
    max_faces: Optional[int] = Form(50),
    min_quality_threshold: Optional[float] = Form(40.0)
) -> JSONResponse:
    """Detect faces in uploaded image"""
    if face_detection_service is None:
        raise HTTPException(
            status_code=503, detail="Face detection service not available"
        )

    try:
        # Read image
        image_data = await file.read()
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Detect faces
        result = await face_detection_service.detect_faces(
            image=image,
            model_name=model_name,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_faces=max_faces,
            min_quality_threshold=min_quality_threshold
        )

        return JSONResponse(content=result.to_dict())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@face_detection_router.post("/face-detection/detect-base64")
async def detect_faces_base64(request: DetectionRequest) -> JSONResponse:
    """Detect faces in base64 encoded image"""
    if face_detection_service is None:
        raise HTTPException(
            status_code=503, detail="Face detection service not available"
        )

    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Detect faces
        result = await face_detection_service.detect_faces(
            image=image,
            model_name=request.model_name,
            conf_threshold=request.conf_threshold,
            iou_threshold=request.iou_threshold,
            max_faces=request.max_faces,
            min_quality_threshold=request.min_quality_threshold
        )

        return JSONResponse(content=result.to_dict())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

# === FACE RECOGNITION API ===
face_recognition_router = APIRouter()
face_recognition_service = None  # Will be injected by main.py

class RecognitionRequest(BaseModel):
    face_image_base64: str
    gallery: dict  # {"person_id": {"name": "...", "embeddings": [...]}
    model_name: Optional[str] = "facenet"
    top_k: Optional[int] = 5

class AddFaceRequest(BaseModel):
    person_id: str
    face_image_base64: str
    metadata: Optional[dict] = None

@face_recognition_router.get("/face-recognition/health")
async def face_recognition_health() -> Dict[str, Any]:
    """Health check for face recognition service"""
    if face_recognition_service is None:
        raise HTTPException(
            status_code=503, detail="Face recognition service not available"
        )

    stats = face_recognition_service.get_performance_stats()
    return {
        "status": "healthy",
        "service": "face_recognition",
        "performance_stats": stats.to_dict()
    }

@face_recognition_router.post("/face-recognition/extract-embedding")
async def extract_embedding_endpoint(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form("facenet")
) -> JSONResponse:
    """Extract face embedding from uploaded image"""
    if face_recognition_service is None:
        raise HTTPException(
            status_code=503, detail="Face recognition service not available"
        )

    try:
        # Read and process image
        image_data = await file.read()
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Extract embedding
        embedding = await face_recognition_service.extract_embedding(image, model_name)

        if embedding is None:
            raise HTTPException(
                status_code=400, detail="Failed to extract embedding"
            )

        return JSONResponse(content=embedding.to_dict())

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Embedding extraction failed: {str(e)}"
        )

@face_recognition_router.post("/face-recognition/recognize")
async def recognize_face_endpoint(request: RecognitionRequest) -> JSONResponse:
    """Recognize face against gallery"""
    if face_recognition_service is None:
        raise HTTPException(
            status_code=503, detail="Face recognition service not available"
        )

    try:
        # Decode image
        image_data = base64.b64decode(request.face_image_base64)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Recognize face
        result = await face_recognition_service.recognize_face(
            face_image=image,
            gallery=request.gallery,
            model_name=request.model_name,
            top_k=request.top_k
        )

        return JSONResponse(content=result.to_dict())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")

@face_recognition_router.post("/face-recognition/add-face")
async def add_face_to_database(request: AddFaceRequest) -> Dict[str, Any]:
    """Add face to internal database"""
    if face_recognition_service is None:
        raise HTTPException(
            status_code=503, detail="Face recognition service not available"
        )

    try:
        # Decode image
        image_data = base64.b64decode(request.face_image_base64)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Add face to database
        success = await face_recognition_service.add_face_to_database(
            person_id=request.person_id,
            face_image=image,
            metadata=request.metadata
        )

        if not success:
            raise HTTPException(
                status_code=400, detail="Failed to add face to database"
            )

        return {"success": True, "message": f"Face added for {request.person_id}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Add face failed: {str(e)}")

# === FACE ANALYSIS API ===
face_analysis_router = APIRouter()
face_analysis_service = None  # Will be injected by main.py

class AnalysisRequest(BaseModel):
    image_base64: str
    mode: str = "full_analysis"  # Options: "detection_only", "recognition_only",
                                # "full_analysis", "comprehensive"
    gallery: Optional[dict] = None
    config: Optional[dict] = None

@face_analysis_router.get("/face-analysis/health")
async def face_analysis_health() -> Dict[str, Any]:
    """Health check for face analysis service"""
    if face_analysis_service is None:
        raise HTTPException(
            status_code=503, detail="Face analysis service not available"
        )

    stats = face_analysis_service.get_performance_stats()
    available_models = await face_analysis_service.get_available_models()

    return {
        "status": "healthy",
        "service": "face_analysis",
        "performance_stats": stats,
        "available_models": available_models
    }

@face_analysis_router.post("/face-analysis/analyze")
async def analyze_faces_endpoint(
    file: UploadFile = File(...),
    mode: str = Form("full_analysis"),
    gallery_json: Optional[str] = Form(None),
    detection_model: Optional[str] = Form("auto"),
    recognition_model: Optional[str] = Form("facenet"),
    confidence_threshold: Optional[float] = Form(0.5),
    max_faces: Optional[int] = Form(50)
) -> JSONResponse:
    """Comprehensive face analysis"""
    if face_analysis_service is None:
        raise HTTPException(
            status_code=503, detail="Face analysis service not available"
        )

    try:
        # Read image
        image_data = await file.read()
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Parse gallery if provided
        gallery = None
        if gallery_json:
            import json
            gallery = json.loads(gallery_json)

        # Create analysis config
        from src.ai_services.face_analysis.models import (
            AnalysisConfig, AnalysisMode, QualityLevel
        )

        config = AnalysisConfig(
            mode=AnalysisMode(mode),
            detection_model=detection_model,
            recognition_model=recognition_model,
            confidence_threshold=confidence_threshold,
            max_faces=max_faces,
            enable_gallery_matching=gallery is not None,
            quality_level=QualityLevel.BALANCED
        )

        # Analyze faces
        result = await face_analysis_service.analyze_faces(
            image=image,
            config=config,
            gallery=gallery
        )

        return JSONResponse(content=result.to_dict())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@face_analysis_router.post("/face-analysis/analyze-base64")
async def analyze_faces_base64(request: AnalysisRequest) -> JSONResponse:
    """Comprehensive face analysis with base64 image"""
    if face_analysis_service is None:
        raise HTTPException(
            status_code=503, detail="Face analysis service not available"
        )

    try:
        # Decode image
        image_data = base64.b64decode(request.image_base64)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Create analysis config
        from src.ai_services.face_analysis.models import (
            AnalysisConfig, AnalysisMode, QualityLevel
        )

        config_dict = request.config or {}
        config = AnalysisConfig(
            mode=AnalysisMode(request.mode),
            detection_model=config_dict.get("detection_model", "auto"),
            recognition_model=config_dict.get("recognition_model", "facenet"),
            confidence_threshold=config_dict.get("confidence_threshold", 0.5),
            max_faces=config_dict.get("max_faces", 50),
            enable_gallery_matching=request.gallery is not None,
            quality_level=QualityLevel(config_dict.get("quality_level", "balanced"))
        )

        # Analyze faces
        result = await face_analysis_service.analyze_faces(
            image=image,
            config=config,
            gallery=request.gallery
        )

        return JSONResponse(content=result.to_dict())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@face_analysis_router.post("/face-analysis/batch")
async def batch_analyze_faces(
    files: List[UploadFile] = File(...),
    mode: str = Form("full_analysis"),
    gallery_json: Optional[str] = Form(None)
) -> JSONResponse:
    """Batch face analysis for multiple images"""
    if face_analysis_service is None:
        raise HTTPException(
            status_code=503, detail="Face analysis service not available"
        )

    try:
        # Read all images
        images = []
        for file_item in files: # Renamed 'file' to 'file_item' to avoid conflict
            image_data = await file_item.read()
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is not None:
                images.append(image)

        if not images:
            raise HTTPException(status_code=400, detail="No valid images provided")

        # Parse gallery if provided
        gallery = None
        if gallery_json:
            import json
            gallery = json.loads(gallery_json)

        # Create analysis config
        from src.ai_services.face_analysis.models import (
            AnalysisConfig, AnalysisMode, QualityLevel
        )

        config = AnalysisConfig(
            mode=AnalysisMode(mode),
            enable_gallery_matching=gallery is not None,
            quality_level=QualityLevel.BALANCED,
            parallel_processing=True
        )

        # Batch analyze
        result = await face_analysis_service.batch_analyze(
            images=images,
            config=config,
            gallery=gallery
        )

        return JSONResponse(content=result.to_dict())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

# Export routers
__all__ = ["face_detection_router", "face_recognition_router", "face_analysis_router"]
