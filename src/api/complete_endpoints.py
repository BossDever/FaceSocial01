"""
Complete API Endpoints for Face Analysis System
Updated version with JSON API support for bulk operations
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends, Request
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, cast
import cv2
import numpy as np
from pydantic import BaseModel
import base64
import json
import logging

from src.ai_services.face_recognition.models import RecognitionModel

# Import service types for dependency injection
from src.ai_services.face_detection.face_detection_service import \
    FaceDetectionService
from src.ai_services.face_recognition.face_recognition_service import \
    FaceRecognitionService
from src.ai_services.face_analysis.face_analysis_service import \
    FaceAnalysisService
from src.ai_services.face_analysis.models import (
    AnalysisConfig, AnalysisMode, QualityLevel
)

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
    gallery: dict
    model_name: Optional[str] = "facenet"
    top_k: Optional[int] = 5

class AddFaceRequest(BaseModel):
    person_id: str
    face_image_base64: str
    metadata: Optional[dict] = None

# NEW: JSON API Models for bulk operations
class AddFaceJSONRequest(BaseModel):
    person_id: str
    person_name: Optional[str] = None
    face_image_base64: str
    model_name: Optional[str] = "facenet"
    metadata: Optional[dict] = None

class AnalysisRequest(BaseModel):
    image_base64: str
    mode: str = "full_analysis"
    gallery: Optional[dict] = None
    config: Optional[dict] = None

class AnalysisJSONRequest(BaseModel):
    image_base64: str
    mode: str = "full_analysis"
    gallery: Optional[dict] = None
    config: Optional[dict] = None

# === UTILITY FUNCTIONS ===
def decode_base64_image(image_base64: str) -> np.ndarray:
    """Decode a base64 encoded image to an OpenCV image"""
    image_data = base64.b64decode(image_base64)
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

def decode_uploaded_image(image_data: bytes) -> np.ndarray:
    """Decode uploaded image file to an OpenCV image"""
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

# === FACE DETECTION API ===
face_detection_router = APIRouter()

def get_face_detection_service(request: Request) -> FaceDetectionService:
    """Dependency to get face detection service from app.state"""
    service = getattr(request.app.state, "face_detection_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Face detection service not available or not initialized properly."
        )
    return cast(FaceDetectionService, service)

@face_detection_router.get("/face-detection/health")
async def face_detection_health(
    service: FaceDetectionService = Depends(get_face_detection_service)
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
        detail = f"Health check failed: {str(e)}"
        raise HTTPException(status_code=500, detail=detail)

@face_detection_router.post("/face-detection/detect")
async def detect_faces_endpoint(
    file: UploadFile = File(...),
    model_name: str = Form("auto"),
    conf_threshold: float = Form(0.5),
    iou_threshold: float = Form(0.4),
    max_faces: int = Form(50),
    min_quality_threshold: float = Form(40.0),
    service: FaceDetectionService = Depends(get_face_detection_service)
) -> JSONResponse:
    """Detect faces in uploaded image"""
    try:
        # Read image
        image_data = await file.read()
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

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

    except Exception as e:
        detail = f"Detection failed: {str(e)}"
        raise HTTPException(status_code=500, detail=detail)

@face_detection_router.post("/face-detection/detect-base64")
async def detect_faces_base64(
    request: DetectionRequest,
    service: FaceDetectionService = Depends(get_face_detection_service)
) -> JSONResponse:
    """Detect faces in base64 encoded image"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

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

    except Exception as e:
        detail = f"Detection failed: {str(e)}"
        raise HTTPException(status_code=500, detail=detail)

# === FACE RECOGNITION API ===
face_recognition_router = APIRouter()

def get_face_recognition_service(request: Request) -> FaceRecognitionService:
    """Dependency to get face recognition service from app.state"""
    service = getattr(request.app.state, "face_recognition_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Face recognition service not available or not initialized properly."
        )
    return cast(FaceRecognitionService, service)

@face_recognition_router.get("/face-recognition/health")
async def face_recognition_health(
    service: FaceRecognitionService = Depends(get_face_recognition_service)
) -> Dict[str, Any]:
    """Health check for face recognition service"""
    try:
        # Check if service has get_performance_stats method
        if hasattr(service, 'get_performance_stats'):
            stats_result = service.get_performance_stats()
            performance_stats_dict = {}
            if isinstance(stats_result, dict):
                performance_stats_dict = stats_result
            elif hasattr(stats_result, 'to_dict') and callable(stats_result.to_dict):
                performance_stats_dict = stats_result.to_dict()
        else:
            performance_stats_dict = {"message": "Performance stats not available"}

        return {
            "status": "healthy",
            "service": "face_recognition",
            "performance_stats": performance_stats_dict
        }
    except Exception as e:
        detail = f"Health check failed: {str(e)}"
        raise HTTPException(status_code=500, detail=detail)

@face_recognition_router.post("/face-recognition/extract-embedding")
async def extract_embedding_endpoint(
    file: UploadFile = File(...),
    model_name: str = Form("facenet"),
    service: FaceRecognitionService = Depends(get_face_recognition_service)
) -> JSONResponse:
    """Extract face embedding from uploaded image"""
    try:
        # Read and process image
        image_data = await file.read()
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Convert model_name string to RecognitionModel enum
        try:
            model_enum = RecognitionModel(model_name.lower())
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid model name: {model_name}"
            )

        # Extract embedding
        embedding = await service._extract_embedding(image, model_enum)

        if embedding is None:
            raise HTTPException(status_code=400, detail="Failed to extract embedding")

        return JSONResponse(content=embedding.to_dict())

    except Exception as e:
        detail = f"Embedding extraction failed: {str(e)}"
        raise HTTPException(status_code=500, detail=detail)

@face_recognition_router.post("/face-recognition/recognize")
async def recognize_face_endpoint(
    request: RecognitionRequest,
    service: FaceRecognitionService = Depends(get_face_recognition_service)
) -> JSONResponse:
    """Recognize face against gallery"""
    try:
        # Decode image
        image_data = base64.b64decode(request.face_image_base64)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Recognize face
        top_k_value = request.top_k if request.top_k is not None else 5
        result = await service.recognize_faces(
            image=image,
            gallery=request.gallery,
            model_name=request.model_name,
            top_k=top_k_value
        )

        return JSONResponse(content=result.to_dict())

    except Exception as e:
        detail = f"Recognition failed: {str(e)}"
        raise HTTPException(status_code=500, detail=detail)

@face_recognition_router.post("/face-recognition/add-face")
async def add_face_to_database(
    file: UploadFile = File(None),
    person_name: str = Form(...),
    person_id: Optional[str] = Form(None),
    face_image_base64: Optional[str] = Form(None),
    service: FaceRecognitionService = Depends(get_face_recognition_service)
) -> Dict[str, Any]:
    """Add face to internal database - supports both file upload and base64"""
    try:
        # Handle both file upload and base64 input
        image_bytes = None
        
        if face_image_base64:
            # Base64 input
            try:
                image_bytes = base64.b64decode(face_image_base64)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid base64 image data: {e}"
                )
        elif file:
            # File upload input
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
            "face_id": result.get('face_id'),
            "model_used": result.get('model_used'),
            "embedding_preview": result.get('embedding_preview', [])[:5]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error adding face for {person_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# NEW: JSON API endpoint for bulk operations
@face_recognition_router.post("/face-recognition/add-face-json")
async def add_face_json(
    request: AddFaceJSONRequest,
    service: FaceRecognitionService = Depends(get_face_recognition_service)
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
            "face_id": result.get('face_id'),
            "model_used": result.get('model_used'),
            "embedding_preview": result.get('embedding_preview', [])[:5]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"❌ Error adding face for {request.person_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@face_recognition_router.get("/face-recognition/get-gallery")
async def get_gallery_endpoint(
    service: FaceRecognitionService = Depends(get_face_recognition_service)
) -> JSONResponse:
    """Retrieve the current face gallery from the service."""
    try:
        # Access face_database directly from service
        database = service.face_database
        
        gallery_data: Dict[str, Dict[str, Any]] = {}
        for person_id, embeddings_list in database.items():
            if isinstance(embeddings_list, list) and embeddings_list:
                # Get the first embedding for gallery format
                first_embedding = embeddings_list[0]
                gallery_data[person_id] = {
                    "name": getattr(first_embedding, 'person_name', person_id),
                    "embeddings": [
                        emb.vector.tolist()
                        for emb in embeddings_list if emb.vector is not None
                    ]
                }
        
        return JSONResponse(content=gallery_data)
    except Exception as e:
        logger.error(f"Failed to get gallery: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve gallery: {str(e)}"
        )

@face_recognition_router.get("/face-recognition/database-status")
async def get_database_status_endpoint(
    service: FaceRecognitionService = Depends(get_face_recognition_service)
) -> JSONResponse:
    """Get detailed database status for debugging"""
    try:
        # Access face_database directly from service
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
                "embedding_details": [] # Initialize as list
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
                    
                    # Ensure embedding_details is a list before appending
                    if isinstance(person_info["embedding_details"], list):
                        person_info["embedding_details"].append(emb_detail)

            status["persons"][person_id] = person_info
        
        if isinstance(status["summary"], dict):
            status["summary"]["total_embeddings"] = total_embeddings
            status["summary"]["valid_embeddings"] = valid_embeddings
            status["summary"]["invalid_embeddings"] = invalid_embeddings

        logger.info(f"Database status check: {status.get('summary')}")
        return JSONResponse(content=status)

    except Exception as e:
        logger.error(f"Failed to get database status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get database status: {str(e)}"
        )

# === FACE ANALYSIS API ===
face_analysis_router = APIRouter()

def get_face_analysis_service(request: Request) -> FaceAnalysisService:
    """Dependency to get face analysis service from app.state"""
    service = getattr(request.app.state, "face_analysis_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Face analysis service not available or not initialized properly."
        )
    return cast(FaceAnalysisService, service)

@face_analysis_router.get("/face-analysis/health")
async def face_analysis_health(
    service: FaceAnalysisService = Depends(get_face_analysis_service)
) -> Dict[str, Any]:
    """Health check for face analysis service"""
    try:
        service_info = await service.get_service_info()
        return {
            "status": "healthy",
            "service": "face_analysis",
            "service_info": service_info
        }
    except Exception as e:
        detail = f"Health check failed: {str(e)}"
        raise HTTPException(status_code=500, detail=detail)


def _parse_analysis_config(config_data: Optional[Dict[str, Any]]) -> AnalysisConfig:
    """Helper to parse analysis config from request data."""
    if not config_data:
        return AnalysisConfig() # Return default config

    # Map string quality level to enum if present
    quality_level_str = config_data.get("min_face_quality")
    if quality_level_str and isinstance(quality_level_str, str):
        try:
            config_data["min_face_quality"] = QualityLevel[quality_level_str.upper()]
        except KeyError:
            logger.warning(
                f"Invalid quality level: {quality_level_str}. Using default."
            )
            config_data.pop("min_face_quality", None)
    
    return AnalysisConfig(**config_data)

@face_analysis_router.post("/face-analysis/analyze")
async def analyze_faces_endpoint(
    file: UploadFile = File(None),
    image_base64: Optional[str] = Form(None),
    mode: str = Form("full_analysis"),
    gallery_json: Optional[str] = Form(None), # Gallery as JSON string
    config_json: Optional[str] = Form(None),   # Config as JSON string
    service: FaceAnalysisService = Depends(get_face_analysis_service)
) -> JSONResponse:
    """Analyze faces in an image (detection, quality, recognition, attributes)"""
    try:
        image_bytes: Optional[bytes] = None
        if image_base64:
            try:
                image_bytes = base64.b64decode(image_base64)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")
        elif file:
            image_bytes = await file.read()
        else:
            raise HTTPException(status_code=400, detail="No image provided.")

        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image data.")

        image = decode_uploaded_image(image_bytes)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format.")

        analysis_mode = AnalysisMode(mode)
        gallery = json.loads(gallery_json) if gallery_json else None
        config_dict = json.loads(config_json) if config_json else {}
        analysis_config = _parse_analysis_config(config_dict)

        result = await service.analyze_image(
            image=image,
            mode=analysis_mode,
            gallery=gallery,
            config=analysis_config
        )
        return JSONResponse(content=result.to_dict())

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {e}")
    except ValueError as e: # For invalid AnalysisMode
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")

# NEW: JSON API endpoint for bulk operations
@face_analysis_router.post("/face-analysis/analyze-json")
async def analyze_faces_json(
    request_data: AnalysisJSONRequest,
    service: FaceAnalysisService = Depends(get_face_analysis_service)
) -> JSONResponse:
    """Analyze faces using JSON request (for bulk operations)"""
    try:
        image = decode_base64_image(request_data.image_base64)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format.")

        analysis_mode = AnalysisMode(request_data.mode)
        analysis_config = _parse_analysis_config(request_data.config)

        result = await service.analyze_image(
            image=image,
            mode=analysis_mode,
            gallery=request_data.gallery,
            config=analysis_config
        )
        return JSONResponse(content=result.to_dict())

    except ValueError as e: # For invalid AnalysisMode or QualityLevel
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"JSON Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"JSON Analysis error: {e}")

# === MAIN ROUTER (Example, adjust as needed) ===
# You would typically include these routers in your main FastAPI app
# from fastapi import FastAPI
# app = FastAPI()
# app.include_router(face_detection_router, prefix="/api", tags=["Face Detection"])
# app.include_router(face_recognition_router, prefix="/api", tags=["Face Recognition"])
# app.include_router(face_analysis_router, prefix="/api", tags=["Face Analysis"])
