"""
Face Analysis API Endpoints
Fixed version with proper dependency injection and error handling
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends, Request
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any, Union
import cv2
import numpy as np
from pydantic import BaseModel
import base64
import json
import logging

logger = logging.getLogger(__name__)

# === MODELS ===
class DetectionRequest(BaseModel):
    image_base64: str
    model_name: Optional[str] = "auto"
    conf_threshold: Optional[float] = 0.5
    iou_threshold: Optional[float] = 0.4
    max_faces: Optional[int] = 50
    min_quality_threshold: Optional[float] = 40.0

class RecognitionRequest(BaseModel):
    face_image_base64: str
    gallery: Dict[str, Any]
    model_name: Optional[str] = "facenet"
    top_k: Optional[int] = 5
    similarity_threshold: Optional[float] = 0.5  # Lowered from 0.6 to 0.5

class AddFaceRequest(BaseModel):
    person_id: str
    face_image_base64: str
    metadata: Optional[Dict[str, Any]] = None

class AnalysisRequest(BaseModel):
    image_base64: str
    mode: str = "full_analysis"
    gallery: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None

# === UTILITY FUNCTIONS ===
def decode_base64_image(image_base64: str) -> np.ndarray:
    """Decode a base64 encoded image to an OpenCV image"""
    try:
        image_data = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        return image
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

def decode_uploaded_image(image_data: bytes) -> np.ndarray:
    """Decode uploaded image file to an OpenCV image"""
    try:
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        return image
    except Exception as e:
        logger.error(f"Failed to decode uploaded image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

def create_analysis_config(
    mode: str = "full_analysis",
    config_dict: Optional[Dict[str, Any]] = None,
    detection_model: str = "auto",
    recognition_model: str = "facenet",
    confidence_threshold: float = 0.5,
    max_faces: int = 50
) -> Dict[str, Any]:
    """Create analysis config dictionary"""
    base_config = {
        "mode": mode,
        "detection_model": detection_model,
        "recognition_model": recognition_model,
        "confidence_threshold": confidence_threshold,
        "max_faces": max_faces,
        "enable_gallery_matching": True,
        "enable_database_matching": True,
        "quality_level": "balanced",
        "parallel_processing": True,
        "return_face_crops": False,
        "return_embeddings": False,
        "return_detailed_stats": True
    }
    
    if config_dict:
        base_config.update(config_dict)
    
    return base_config

# === FACE DETECTION API ===
face_detection_router = APIRouter()

def get_face_detection_service(request: Request):
    """Dependency to get face detection service from app.state"""
    service = getattr(request.app.state, "face_detection_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Face detection service not available"
        )
    return service

@face_detection_router.get("/face-detection/health")
async def face_detection_health(
    service = Depends(get_face_detection_service)
) -> Dict[str, Any]:
    """Health check for face detection service"""
    try:
        service_info = await service.get_service_info()
        return {
            "status": "healthy",
            "service": "face_detection",
            "service_info": service_info
        }
    except Exception as e:
        logger.error(f"Face detection health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@face_detection_router.post("/face-detection/detect")
async def detect_faces_endpoint(
    file: UploadFile = File(...),
    model_name: str = Form("auto"),
    conf_threshold: float = Form(0.5),
    iou_threshold: float = Form(0.4),
    max_faces: int = Form(50),
    min_quality_threshold: float = Form(40.0),
    service = Depends(get_face_detection_service)
) -> JSONResponse:
    """Detect faces in uploaded image"""
    try:
        # Read and decode image
        image_data = await file.read()
        image = decode_uploaded_image(image_data)

        # Detect faces
        result = await service.detect_faces(
            image_input=image,
            model_name=model_name,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_faces=max_faces,
            min_quality_threshold=min_quality_threshold
        )

        return JSONResponse(content=result.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@face_detection_router.post("/face-detection/detect-base64")
async def detect_faces_base64(
    request: DetectionRequest,
    service = Depends(get_face_detection_service)
) -> JSONResponse:
    """Detect faces in base64 encoded image"""
    try:
        # Decode base64 image
        image = decode_base64_image(request.image_base64)

        # Detect faces
        result = await service.detect_faces(
            image_input=image,
            model_name=request.model_name,
            conf_threshold=request.conf_threshold,
            iou_threshold=request.iou_threshold,
            max_faces=request.max_faces,
            min_quality_threshold=request.min_quality_threshold
        )

        return JSONResponse(content=result.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

# === FACE RECOGNITION API ===
face_recognition_router = APIRouter()

def get_face_recognition_service(request: Request):
    """Dependency to get face recognition service from app.state"""
    service = getattr(request.app.state, "face_recognition_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Face recognition service not available"
        )
    return service

@face_recognition_router.get("/face-recognition/health")
async def face_recognition_health(
    service = Depends(get_face_recognition_service)
) -> Dict[str, Any]:
    """Health check for face recognition service"""
    try:
        service_info = service.get_service_info()
        return {
            "status": "healthy",
            "service": "face_recognition",
            "service_info": service_info
        }
    except Exception as e:
        logger.error(f"Face recognition health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@face_recognition_router.post("/face-recognition/extract-embedding")
async def extract_embedding_endpoint(
    file: UploadFile = File(...),
    model_name: str = Form("facenet"),
    service = Depends(get_face_recognition_service)
) -> JSONResponse:
    """Extract face embedding from uploaded image"""
    try:
        # Read and decode image
        image_data = await file.read()
        image = decode_uploaded_image(image_data)        # Convert image to bytes for service
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()

        # Extract embedding using service method (WITHOUT adding to database)
        result = await service.extract_embedding_only(
            image_bytes=image_bytes,
            model_name=model_name
        )

        if not result.get('success'):
            raise HTTPException(
                status_code=400,
                detail=result.get('error', 'Failed to extract embedding')
            )

        return JSONResponse(content={
            "success": True,
            "embedding": result.get('embedding', []),
            "model_used": result.get('model_used', model_name),
            "vector": result.get('embedding', []),
            "dimension": result.get('dimension', 0),
            "embedding_preview": result.get('embedding_preview', [])[:5] if result.get('embedding_preview') else []
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {str(e)}")

@face_recognition_router.post("/face-recognition/recognize")
async def recognize_face_endpoint(
    request: RecognitionRequest,
    service = Depends(get_face_recognition_service)
) -> JSONResponse:
    """Recognize face against gallery"""
    try:
        image_data = base64.b64decode(request.face_image_base64)
        
        result_dict = await service.recognize_faces_with_gallery(
            image_bytes=image_data,
            gallery=request.gallery,
            model_name=request.model_name
        )
        
        return JSONResponse(content=result_dict)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recognition failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")

@face_recognition_router.post("/face-recognition/add-face")
async def add_face_to_database(
    request: AddFaceRequest,
    service = Depends(get_face_recognition_service)
) -> Dict[str, Any]:
    """Add face to internal database"""
    try:
        # Decode image
        image_data = base64.b64decode(request.face_image_base64)

        # Add face to database using service
        result = await service.add_face_from_image(
            image_bytes=image_data,
            person_name=request.person_id,
            person_id=request.person_id
        )

        if not result or not result.get('success'):
            error_msg = (result.get('error', 'Failed to add face')
                         if result else 'Failed to add face')
            raise HTTPException(status_code=400, detail=error_msg)

        return {
            "success": True,
            "message": f"Face added for {request.person_id}",
            "person_id": request.person_id,
            "face_ids": result.get('face_ids', []),
            "model_used": result.get('model_used'),
            "embedding_preview": result.get('embedding_preview', [])[:5]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add face failed: {e}")
        raise HTTPException(status_code=500, detail=f"Add face failed: {str(e)}")

# === FACE ANALYSIS API ===
face_analysis_router = APIRouter()

def get_face_analysis_service(request: Request):
    """Dependency to get face analysis service from app.state"""
    service = getattr(request.app.state, "face_analysis_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Face analysis service not available"
        )
    return service

@face_analysis_router.get("/face-analysis/health")
async def face_analysis_health(
    service = Depends(get_face_analysis_service)
) -> Dict[str, Any]:
    """Health check for face analysis service"""
    try:
        service_info = service.get_service_info()
        return {
            "status": "healthy",
            "service": "face_analysis",
            "service_info": service_info
        }
    except Exception as e:
        logger.error(f"Face analysis health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@face_analysis_router.post("/face-analysis/analyze")
async def analyze_faces_endpoint(
    file: UploadFile = File(...),
    mode: str = Form("full_analysis"),
    gallery_json: Optional[str] = Form(None),
    detection_model: str = Form("auto"),
    recognition_model: str = Form("facenet"),
    confidence_threshold: float = Form(0.5),
    max_faces: int = Form(50),
    service = Depends(get_face_analysis_service)
) -> JSONResponse:
    """Comprehensive face analysis"""
    try:
        # Read and decode image
        image_data = await file.read()
        image = decode_uploaded_image(image_data)

        # Parse gallery if provided
        gallery = None
        if gallery_json:
            try:
                gallery = json.loads(gallery_json)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400, detail="Invalid gallery JSON format"
                )

        # Create analysis config
        config_dict = create_analysis_config(
            mode=mode,
            detection_model=detection_model,
            recognition_model=recognition_model,
            confidence_threshold=confidence_threshold,
            max_faces=max_faces
        )

        # Try to import AnalysisConfig, fallback to dict if not available
        try:
            from src.ai_services.face_analysis.models import AnalysisConfig
            analysis_config = AnalysisConfig(**config_dict)
        except ImportError:
            # Use dict directly as fallback
            analysis_config = config_dict

        # Analyze faces
        result = await service.analyze_faces(
            image=image,
            config=analysis_config,
            gallery=gallery
        )

        return JSONResponse(content=result.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@face_analysis_router.post("/face-analysis/analyze-base64")
async def analyze_faces_base64(
    request: AnalysisRequest,
    service = Depends(get_face_analysis_service)
) -> JSONResponse:
    """Comprehensive face analysis with base64 image"""
    try:
        # Decode image
        image = decode_base64_image(request.image_base64)

        # Create analysis config
        config_dict = create_analysis_config(
            mode=request.mode,
            config_dict=request.config
        )

        # Try to import AnalysisConfig, fallback to dict if not available
        try:
            from src.ai_services.face_analysis.models import AnalysisConfig
            analysis_config = AnalysisConfig(**config_dict)
        except ImportError:
            # Use dict directly as fallback
            analysis_config = config_dict

        # Analyze faces
        result = await service.analyze_faces(
            image=image,
            config=analysis_config,
            gallery=request.gallery
        )

        return JSONResponse(content=result.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@face_analysis_router.post("/face-analysis/batch")
async def batch_analyze_faces(
    files: List[UploadFile] = File(...),
    mode: str = Form("full_analysis"),
    gallery_json: Optional[str] = Form(None),
    service = Depends(get_face_analysis_service)
) -> JSONResponse:
    """Batch face analysis for multiple images"""
    try:
        # Read all images
        images = []
        for file_item in files:
            try:
                image_data = await file_item.read()
                image = decode_uploaded_image(image_data)
                images.append(image)
            except Exception as e:
                logger.warning(f"Failed to read image {file_item.filename}: {e}")
                continue

        if not images:
            raise HTTPException(status_code=400, detail="No valid images provided")

        # Parse gallery if provided
        gallery = None
        if gallery_json:
            try:
                gallery = json.loads(gallery_json)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400, detail="Invalid gallery JSON format"
                )

        # Create analysis config
        config_dict = create_analysis_config(
            mode=mode,
            config_dict={"parallel_processing": True}
        )

        # Try to import AnalysisConfig, fallback to dict if not available
        try:
            from src.ai_services.face_analysis.models import AnalysisConfig
            analysis_config = AnalysisConfig(**config_dict)
        except ImportError:
            # Use dict directly as fallback
            analysis_config = config_dict

        # Process images sequentially (batch processing within service)
        results = []
        for i, image in enumerate(images):
            try:
                result = await service.analyze_faces(
                    image=image,
                    config=analysis_config,
                    gallery=gallery
                )
                results.append({
                    "image_index": i,
                    "success": True,
                    "result": result.to_dict()
                })
            except Exception as e:
                logger.error(f"Failed to analyze image {i}: {e}")
                results.append({
                    "image_index": i,
                    "success": False,
                    "error": str(e)
                })

        # Create batch summary
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]

        total_faces = sum(
            r["result"]["statistics"]["total_faces"] 
            for r in successful_results
        )
        total_identified = sum(
            r["result"]["statistics"]["identified_faces"] 
            for r in successful_results
        )

        batch_result = {
            "batch_summary": {
                "total_images": len(images),
                "successful_analyses": len(successful_results),
                "failed_analyses": len(failed_results),
                "total_faces_detected": total_faces,
                "total_faces_identified": total_identified,
                "overall_recognition_rate": (
                    total_identified / total_faces if total_faces > 0 else 0
                )
            },
            "individual_results": results
        }

        return JSONResponse(content=batch_result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

# Export routers
__all__ = ["face_detection_router", "face_recognition_router", "face_analysis_router", "router"]

# Main router for backward compatibility
router = face_analysis_router