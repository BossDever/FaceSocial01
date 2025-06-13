"""
Complete API Endpoints for Face Analysis System
Fixed version with proper error handling and type hints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends, Request
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, Union
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

class AddFaceJSONRequest(BaseModel):
    person_id: str
    person_name: Optional[str] = None
    face_image_base64: str
    model_name: Optional[str] = "facenet"
    metadata: Optional[Dict[str, Any]] = None

class FaceAnalysisJSONRequest(BaseModel):
    image_base64: str
    mode: Optional[str] = "full_analysis"
    config: Optional[Dict[str, Any]] = None
    gallery: Optional[Dict[str, Any]] = None

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

# === FACE DETECTION API ===
face_detection_router = APIRouter()

def get_face_detection_service(request: Request):
    """Dependency to get face detection service from app.state"""
    service = getattr(request.app.state, "face_detection_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Face detection service not available or not initialized properly."
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
            detail="Face recognition service not available or not initialized properly."
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
        image = decode_uploaded_image(image_data)

        # Convert image to bytes for service
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()

        # Extract embedding using service method
        result = await service.add_face_from_image(
            image_bytes=image_bytes,
            person_name="temp",
            person_id="temp",
            model_name=model_name
        )

        if not result.get('success'):
            raise HTTPException(
                status_code=400, 
                detail=result.get('error', 'Failed to extract embedding')
            )

        return JSONResponse(content={
            "success": True,
            "embedding": result.get('embedding_preview', []),
            "model_used": result.get('model_used', model_name),
            "vector": result.get('embedding_preview', [])
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {str(e)}")

@face_recognition_router.post("/face-recognition/recognize")
async def recognize_face_endpoint(
    request_data: RecognitionRequest,
    service = Depends(get_face_recognition_service)
) -> JSONResponse:
    """Recognize face against gallery"""
    try:
        image_data = base64.b64decode(request_data.face_image_base64)
        
        result_dict = await service.recognize_faces_with_gallery(
            image_bytes=image_data,
            gallery=request_data.gallery,
            model_name=request_data.model_name
        )
        
        return JSONResponse(content=result_dict)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recognition failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")

@face_recognition_router.post("/face-recognition/add-face")
async def add_face_to_database(
    file: UploadFile = File(None),
    person_name: str = Form(...),
    person_id: Optional[str] = Form(None),
    face_image_base64: Optional[str] = Form(None),
    service = Depends(get_face_recognition_service)
) -> Dict[str, Any]:
    """Add face to internal database - supports both file upload and base64"""
    try:
        image_bytes = None
        
        if face_image_base64:
            try:
                image_bytes = base64.b64decode(face_image_base64)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid base64 image data: {e}"
                )
        elif file:
            image_bytes = await file.read()
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'file' or 'face_image_base64' must be provided"
            )

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="No image data received"
            )

        # Use person_name as person_id if not provided
        if person_id is None:
            person_id = person_name

        # Add face to database using the service
        result = await service.add_face_from_image(
            image_bytes=image_bytes,
            person_name=person_name,
            person_id=person_id
        )

        if not result or not result.get('success'):
            error_msg = (result.get('error', 'Failed to add face')
                         if result else 'Failed to add face')
            raise HTTPException(status_code=400, detail=error_msg)

        return {
            "success": True,
            "message": f"Face added successfully for {person_name}",
            "person_id": person_id,
            "person_name": person_name,
            "face_ids": result.get('face_ids', []),
            "model_used": result.get('model_used'),
            "embedding_preview": result.get('embedding_preview', [])[:5]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding face for {person_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@face_recognition_router.post("/face-recognition/add-face-json")
async def add_face_json(
    request: AddFaceJSONRequest,
    service = Depends(get_face_recognition_service)
) -> Dict[str, Any]:
    """Add face to database using JSON request (for bulk operations)"""
    try:
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(request.face_image_base64)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 image data: {e}"
            )

        # Use person_id as person_name if not provided
        person_name = request.person_name or request.person_id

        # Add face to database
        result = await service.add_face_from_image(
            image_bytes=image_bytes,
            person_name=person_name,
            person_id=request.person_id,
            model_name=request.model_name
        )

        if not result or not result.get('success'):
            error_msg = (result.get('error', 'Failed to add face')
                         if result else 'Failed to add face')
            raise HTTPException(status_code=400, detail=error_msg)

        return {
            "success": True,
            "message": f"Face added successfully for {person_name}",
            "person_id": request.person_id,
            "person_name": person_name,
            "face_ids": result.get('face_ids', []),
            "model_used": result.get('model_used'),
            "embedding_preview": result.get('embedding_preview', [])[:5]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding face for {request.person_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@face_recognition_router.get("/face-recognition/get-gallery")
async def get_gallery_endpoint(
    service = Depends(get_face_recognition_service)
) -> JSONResponse:
    """Retrieve the current face gallery from the service."""
    try:
        database = service.face_database
        gallery_data_transformed: Dict[str, Dict[str, Any]] = {}
        
        if isinstance(database, dict):
            for person_id, embeddings_list in database.items():
                if isinstance(embeddings_list, list) and embeddings_list:
                    # Get name from first embedding, default to person_id
                    person_name = getattr(embeddings_list[0], 'person_name', person_id)
                    
                    embedding_vectors = []
                    for emb_obj in embeddings_list:
                        if hasattr(emb_obj, 'vector') and emb_obj.vector is not None:
                            embedding_vectors.append(emb_obj.vector.tolist())
                            
                    gallery_data_transformed[person_id] = {
                        "name": person_name,
                        "embeddings": embedding_vectors
                    }
                elif isinstance(embeddings_list, list):
                    gallery_data_transformed[person_id] = {
                        "name": person_id,
                        "embeddings": []
                    }
                    
        return JSONResponse(content=gallery_data_transformed)
        
    except Exception as e:
        logger.error(f"Failed to get gallery: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve gallery: {str(e)}"
        )

@face_recognition_router.get("/face-recognition/database-status")
async def get_database_status_endpoint(
    service = Depends(get_face_recognition_service)
) -> JSONResponse:
    """Get detailed database status for debugging"""
    try:
        database = service.face_database

        status: Dict[str, Any] = {
            "total_persons": len(database),
            "persons": {},
            "summary": {
                "total_embeddings": 0,
                "valid_embeddings": 0,
                "invalid_embeddings": 0
            }
        }

        total_embeddings = 0
        valid_embeddings = 0
        invalid_embeddings = 0

        for person_id, embeddings_data in database.items():
            person_info: Dict[str, Any] = {
                "embeddings_count": (
                    len(embeddings_data)
                    if isinstance(embeddings_data, list)
                    else 0
                ),
                "data_type": type(embeddings_data).__name__,
                "embedding_details": []
            }

            if isinstance(embeddings_data, list):
                for i, emb_obj in enumerate(embeddings_data):
                    total_embeddings += 1
                    emb_detail: Dict[str, Any] = {
                        "index": i,
                        "type": type(emb_obj).__name__,
                        "has_vector": hasattr(emb_obj, 'vector'),
                    }

                    if hasattr(emb_obj, 'vector'):
                        vector = emb_obj.vector
                        emb_detail.update({
                            "vector_type": type(vector).__name__,
                            "vector_shape": getattr(vector, 'shape', 'no_shape'),
                            "is_ndarray": isinstance(vector, np.ndarray)
                        })
                        if isinstance(vector, np.ndarray):
                            valid_embeddings += 1
                        else:
                            invalid_embeddings += 1
                    else:
                        invalid_embeddings += 1
                        emb_detail["error"] = "No vector attribute"
                    
                    person_info["embedding_details"].append(emb_detail)

            status["persons"][person_id] = person_info
        
        status["summary"]["total_embeddings"] = total_embeddings
        status["summary"]["valid_embeddings"] = valid_embeddings
        status["summary"]["invalid_embeddings"] = invalid_embeddings

        return JSONResponse(content=status)

    except Exception as e:
        logger.error(f"Failed to get database status: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get database status: {str(e)}"
        )

# === FACE ANALYSIS API ===
face_analysis_router = APIRouter()

def get_face_analysis_service(request: Request):
    """Dependency to get face analysis service from app.state"""
    service = getattr(request.app.state, "face_analysis_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Face analysis service not available or not initialized properly."
        )
    return service

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

@face_analysis_router.post("/face-analysis/analyze-json")
async def analyze_faces_json(
    request_data: FaceAnalysisJSONRequest,
    service = Depends(get_face_analysis_service),
) -> JSONResponse:
    """Analyze faces in a base64 encoded image using JSON request"""
    try:
        image = decode_base64_image(request_data.image_base64)
        
        # Create analysis config
        config_dict = create_analysis_config(
            mode=request_data.mode or "full_analysis",
            config_dict=request_data.config
        )
        
        # Import AnalysisConfig here to avoid circular imports
        try:
            from src.ai_services.face_analysis.models import AnalysisConfig
            analysis_config = AnalysisConfig(**config_dict)
        except ImportError:
            # Fallback: use dict directly
            analysis_config = config_dict
        
        result = await service.analyze_faces(
            image=image,
            config=analysis_config,
            gallery=request_data.gallery
        )
        
        return JSONResponse(content=result.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"JSON Face analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")

@face_analysis_router.post("/face-analysis/analyze")
async def analyze_faces_endpoint(
    file: UploadFile = File(...),
    mode: str = Form("full_analysis"),
    gallery_json: Optional[str] = Form(None),
    config_json: Optional[str] = Form(None),
    detection_model: str = Form("auto"),
    recognition_model: str = Form("facenet"),
    confidence_threshold: float = Form(0.5),
    max_faces: int = Form(50),
    service = Depends(get_face_analysis_service),
) -> JSONResponse:
    """Analyze faces in an uploaded image file (form data)"""
    try:
        # Read and decode image
        image_data = await file.read()
        image = decode_uploaded_image(image_data)

        # Parse gallery
        parsed_gallery = None
        if gallery_json:
            try:
                parsed_gallery = json.loads(gallery_json)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400, detail="Invalid gallery JSON format"
                )

        # Parse config
        parsed_config_dict = {}
        if config_json:
            try:
                parsed_config_dict = json.loads(config_json)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400, detail="Invalid config JSON format"
                )
        
        # Create analysis config
        config_dict = create_analysis_config(
            mode=mode,
            config_dict=parsed_config_dict,
            detection_model=detection_model,
            recognition_model=recognition_model,
            confidence_threshold=confidence_threshold,
            max_faces=max_faces
        )
        
        # Import AnalysisConfig here to avoid circular imports
        try:
            from src.ai_services.face_analysis.models import AnalysisConfig
            analysis_config = AnalysisConfig(**config_dict)
        except ImportError:
            # Fallback: use dict directly
            analysis_config = config_dict

        result = await service.analyze_faces(
            image=image,
            config=analysis_config,
            gallery=parsed_gallery
        )
        
        return JSONResponse(content=result.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Form-based Face analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")

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