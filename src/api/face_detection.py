"""
Face Detection API Router
Fixed version with proper dependency injection and comprehensive endpoints
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List
import cv2
import numpy as np
from pydantic import BaseModel
import base64
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter()

# === MODELS ===
class DetectionRequest(BaseModel):
    image_base64: str
    model_name: Optional[str] = "auto"
    conf_threshold: Optional[float] = 0.5
    iou_threshold: Optional[float] = 0.4
    max_faces: Optional[int] = 50
    min_quality_threshold: Optional[float] = 40.0

class DetectionConfig(BaseModel):
    model_name: str = "auto"
    conf_threshold: float = 0.5
    iou_threshold: float = 0.4
    max_faces: int = 50
    min_quality_threshold: float = 40.0
    return_landmarks: bool = False
    use_fallback: bool = True

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

def validate_image_format(file: UploadFile) -> bool:
    """Validate uploaded file format"""
    allowed_types = {
        "image/jpeg", "image/jpg", "image/png", 
        "image/bmp", "image/tiff", "image/webp"
    }
    
    if file.content_type not in allowed_types:
        return False
    
    # Check file extension
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        if ext not in allowed_extensions:
            return False
    
    return True

def create_detection_config(
    model_name: str = "auto",
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.4,
    max_faces: int = 50,
    min_quality_threshold: float = 40.0,
    **kwargs
) -> Dict[str, Any]:
    """Create detection configuration dictionary"""
    return {
        "model_name": model_name,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "max_faces": max_faces,
        "min_quality_threshold": min_quality_threshold,
        "return_landmarks": kwargs.get("return_landmarks", False),
        "use_fallback": kwargs.get("use_fallback", True)
    }

# === DEPENDENCY INJECTION ===
def get_face_detection_service(request: Request):
    """Dependency to get face detection service from app.state"""
    service = getattr(request.app.state, "face_detection_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Face detection service not available or not initialized properly."
        )
    return service

# === API ENDPOINTS ===
@router.get("/health")
async def health_check(
    service = Depends(get_face_detection_service)
) -> Dict[str, Any]:
    """Health check endpoint"""
    try:
        service_info = await service.get_service_info()
        return {
            "status": "healthy",
            "service": "face_detection",
            "service_info": service_info,
            "endpoints": [
                "/health",
                "/detect",
                "/detect-base64",
                "/detect-batch",
                "/models/status",
                "/models/available"
            ]
        }
    except Exception as e:
        logger.error(f"Face detection health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.post("/detect")
async def detect_faces_uploaded(
    file: UploadFile = File(...),
    model_name: str = Form("auto"),
    conf_threshold: float = Form(0.5),
    iou_threshold: float = Form(0.4),
    max_faces: int = Form(50),
    min_quality_threshold: float = Form(40.0),
    return_landmarks: bool = Form(False),
    use_fallback: bool = Form(True),
    service = Depends(get_face_detection_service)
) -> JSONResponse:
    """
    Detect faces in uploaded image file
    
    Parameters:
    - file: Image file to process
    - model_name: Detection model to use ("auto", "yolov9c", "yolov9e", "yolov11m")
    - conf_threshold: Confidence threshold for detection (0.0-1.0)
    - iou_threshold: IoU threshold for NMS (0.0-1.0)
    - max_faces: Maximum number of faces to detect
    - min_quality_threshold: Minimum quality score for face filtering
    - return_landmarks: Whether to return facial landmarks
    - use_fallback: Whether to use fallback detection if primary fails
    """
    try:
        # Validate file format
        if not validate_image_format(file):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file.content_type}"
            )

        # Read and decode image
        image_data = await file.read()
        image = decode_uploaded_image(image_data)

        logger.info(f"Processing image: {file.filename} ({image.shape})")

        # Detect faces
        result = await service.detect_faces(
            image_input=image,
            model_name=model_name,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_faces=max_faces,
            min_quality_threshold=min_quality_threshold,
            return_landmarks=return_landmarks,
            use_fallback=use_fallback
        )

        logger.info(f"Detection complete: {len(result.faces)} faces found")
        return JSONResponse(content=result.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face detection failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@router.post("/detect-base64")
async def detect_faces_base64(
    request: DetectionRequest,
    service = Depends(get_face_detection_service)
) -> JSONResponse:
    """
    Detect faces in base64 encoded image
    
    Body:
    - image_base64: Base64 encoded image data
    - model_name: Detection model to use
    - conf_threshold: Confidence threshold
    - iou_threshold: IoU threshold for NMS
    - max_faces: Maximum number of faces
    - min_quality_threshold: Minimum quality score
    """
    try:
        # Decode base64 image
        image = decode_base64_image(request.image_base64)
        
        logger.info(f"Processing base64 image ({image.shape})")

        # Detect faces
        result = await service.detect_faces(
            image_input=image,
            model_name=request.model_name,
            conf_threshold=request.conf_threshold,
            iou_threshold=request.iou_threshold,
            max_faces=request.max_faces,
            min_quality_threshold=request.min_quality_threshold
        )

        logger.info(f"Detection complete: {len(result.faces)} faces found")
        return JSONResponse(content=result.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@router.post("/detect-batch")
async def detect_faces_batch(
    files: List[UploadFile] = File(...),
    model_name: str = Form("auto"),
    conf_threshold: float = Form(0.5),
    iou_threshold: float = Form(0.4),
    max_faces: int = Form(50),
    min_quality_threshold: float = Form(40.0),
    service = Depends(get_face_detection_service)
) -> JSONResponse:
    """
    Batch detect faces in multiple uploaded images
    
    Parameters:
    - files: List of image files to process
    - model_name: Detection model to use
    - conf_threshold: Confidence threshold
    - iou_threshold: IoU threshold for NMS
    - max_faces: Maximum number of faces per image
    - min_quality_threshold: Minimum quality score
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        if len(files) > 20:  # Limit batch size
            raise HTTPException(
                status_code=400, 
                detail="Too many files. Maximum 20 files per batch."
            )

        results = []
        successful_count = 0
        failed_count = 0

        for i, file in enumerate(files):
            try:
                # Validate file format
                if not validate_image_format(file):
                    results.append({
                        "file_index": i,
                        "filename": file.filename,
                        "success": False,
                        "error": f"Unsupported file format: {file.content_type}"
                    })
                    failed_count += 1
                    continue

                # Read and decode image
                image_data = await file.read()
                image = decode_uploaded_image(image_data)

                # Detect faces
                detection_result = await service.detect_faces(
                    image_input=image,
                    model_name=model_name,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    max_faces=max_faces,
                    min_quality_threshold=min_quality_threshold
                )

                results.append({
                    "file_index": i,
                    "filename": file.filename,
                    "success": True,
                    "result": detection_result.to_dict()
                })
                successful_count += 1

            except Exception as e:
                logger.error(f"Failed to process file {file.filename}: {e}")
                results.append({
                    "file_index": i,
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
                failed_count += 1

        # Calculate batch statistics
        total_faces = sum(
            len(r["result"]["faces"]) 
            for r in results 
            if r["success"]
        )

        batch_summary = {
            "total_files": len(files),
            "successful_detections": successful_count,
            "failed_detections": failed_count,
            "total_faces_detected": total_faces,
            "average_faces_per_image": (
                total_faces / successful_count if successful_count > 0 else 0
            )
        }

        return JSONResponse(content={
            "batch_summary": batch_summary,
            "individual_results": results
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch detection failed: {str(e)}")

@router.get("/models/status")
async def get_models_status(
    service = Depends(get_face_detection_service)
) -> Dict[str, Any]:
    """Get status of all detection models"""
    try:
        service_info = await service.get_service_info()
        
        return {
            "models_loaded": service_info.get("models_loaded", False),
            "model_info": service_info.get("model_info", {}),
            "performance_stats": service_info.get("performance_stats", {}),
            "detection_config": service_info.get("detection_config", {}),
            "fallback_enabled": service_info.get("fallback_enabled", False)
        }
    except Exception as e:
        logger.error(f"Failed to get models status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get models status: {str(e)}")

@router.get("/models/available")
async def get_available_models(
    service = Depends(get_face_detection_service)
) -> Dict[str, Any]:
    """Get list of available detection models"""
    try:
        service_info = await service.get_service_info()
        model_info = service_info.get("model_info", {})
        
        available_models = []
        for model_name, info in model_info.items():
            if isinstance(info, dict) and info.get("model_loaded", False):
                available_models.append({
                    "name": model_name,
                    "loaded": info.get("model_loaded", False),
                    "device": info.get("device", "unknown"),
                    "performance": info.get("performance_stats", {})
                })
        
        return {
            "available_models": available_models,
            "total_models": len(available_models),
            "supported_formats": ["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
            "max_image_size": "No limit (memory dependent)",
            "recommended_models": {
                "speed": "yolov9c",
                "accuracy": "yolov9e", 
                "balanced": "yolov11m"
            }
        }
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get available models: {str(e)}")

@router.post("/test-detection")
async def test_detection_endpoint(
    model_name: str = Form("auto"),
    conf_threshold: float = Form(0.5),
    service = Depends(get_face_detection_service)
) -> Dict[str, Any]:
    """
    Test detection with a simple synthetic image
    Useful for testing model availability and performance
    """
    try:
        # Create a simple test image (solid color with some noise)
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some simple patterns to make it more realistic
        cv2.rectangle(test_image, (100, 100), (300, 300), (128, 128, 128), -1)
        cv2.circle(test_image, (200, 200), 50, (255, 255, 255), -1)
        
        logger.info(f"Testing detection with model: {model_name}")
        
        # Run detection
        result = await service.detect_faces(
            image_input=test_image,
            model_name=model_name,
            conf_threshold=conf_threshold,
            max_faces=10
        )
        
        return {
            "test_result": "success",
            "model_used": result.model_used,
            "processing_time_ms": result.total_processing_time,
            "faces_detected": len(result.faces),
            "fallback_used": result.fallback_used,
            "message": "Detection test completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Detection test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection test failed: {str(e)}")

@router.get("/performance/stats")
async def get_performance_stats(
    service = Depends(get_face_detection_service)
) -> Dict[str, Any]:
    """Get detailed performance statistics"""
    try:
        performance_stats = service.get_performance_stats()
        return {
            "performance_stats": performance_stats,
            "timestamp": "current",
            "service": "face_detection"
        }
    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance stats: {str(e)}")

# Export router
__all__ = ["router"]