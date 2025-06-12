"""
Complete API Endpoints for Face Analysis System
Fixed version with proper service injection using app.state
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends, Request
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any, cast # Added cast
import cv2
import numpy as np
from pydantic import BaseModel
import base64
import json
import logging

from src.ai_services.face_recognition.models import RecognitionModel # Added this line

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

# Global service instances are removed; services will be accessed via app.state

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
    gallery: dict # type: ignore
    model_name: Optional[str] = "facenet"
    top_k: Optional[int] = 5

class AddFaceRequest(BaseModel):
    person_id: str
    face_image_base64: str
    metadata: Optional[dict] = None # type: ignore

class AnalysisRequest(BaseModel):
    image_base64: str
    mode: str = "full_analysis"
    gallery: Optional[dict] = None # type: ignore
    config: Optional[dict] = None # type: ignore

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
    model_name: str = Form("auto"), # Changed to str
    conf_threshold: float = Form(0.5), # Changed to float
    iou_threshold: float = Form(0.4), # Changed to float
    max_faces: int = Form(50), # Changed to int
    min_quality_threshold: float = Form(40.0), # Changed to float
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
            image_input=image, # Changed from image to image_input
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
            image_input=image, # Changed from image to image_input
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
        stats_result = await service.get_performance_stats()
        # Ensure stats_result is a dict or has to_dict before calling
        performance_stats_dict = {}
        if isinstance(stats_result, dict):
            performance_stats_dict = stats_result
        elif hasattr(stats_result, 'to_dict') and callable(stats_result.to_dict):
            performance_stats_dict = stats_result.to_dict()
        else:
            logger.warning(
                "Performance stats from face_recognition_service is not a dict "
                "and has no to_dict method."
            )
            # Fallback to an empty dict or a specific structure if preferred

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
    model_name: str = Form("facenet"), # Changed to str
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
            raise HTTPException(status_code=400, detail=f"Invalid model name: {model_name}")

        # Extract embedding
        embedding = await service.extract_embedding(image, model_enum)

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
        result = await service.recognize_face(
            image=image,  # Changed from face_image to image
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
    request: AddFaceRequest,
    service: FaceRecognitionService = Depends(get_face_recognition_service)
) -> Dict[str, Any]:
    """Add face to internal database"""
    try:
        # Decode image
        image_data = base64.b64decode(request.face_image_base64)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Add face to database
        success = await service.add_face_to_database(
            person_id=request.person_id,
            face_image=image,
            metadata=request.metadata
        )

        if not success:
            detail = "Failed to add face to database"
            raise HTTPException(status_code=400, detail=detail)

        return {"success": True, "message": f"Face added for {request.person_id}"}

    except Exception as e:
        detail = f"Add face failed: {str(e)}"
        raise HTTPException(status_code=500, detail=detail)

@face_recognition_router.get("/face-recognition/get-gallery")
async def get_gallery_endpoint(
    service: FaceRecognitionService = Depends(get_face_recognition_service)
) -> JSONResponse:
    """Retrieve the current face gallery from the service."""
    try:
        gallery_data = await service.get_gallery() # type: ignore
        # FastAPI will automatically convert the Pydantic model (FaceGallery)
        # or dict to a JSON response.
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

        status = {
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
            person_info = {
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
                    emb_detail = {
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
                    person_info["embedding_details"].append(emb_detail) # type: ignore

            status["persons"][person_id] = person_info # type: ignore
        status["summary"]["total_embeddings"] = total_embeddings # type: ignore
        status["summary"]["valid_embeddings"] = valid_embeddings # type: ignore
        status["summary"]["invalid_embeddings"] = invalid_embeddings # type: ignore

        logger.info(f"Database status check: {status['summary']}")
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
        stats = service.get_performance_stats()
        available_models = await service.get_available_models()

        return {
            "status": "healthy",
            "service": "face_analysis",
            "performance_stats": stats,
            "available_models": available_models
        }
    except Exception as e:
        detail = f"Health check failed: {str(e)}"
        raise HTTPException(status_code=500, detail=detail)

@face_analysis_router.post("/face-analysis/analyze")
async def analyze_faces_endpoint(
    file: UploadFile = File(...),
    mode: str = Form("full_analysis"),
    gallery_json: Optional[str] = Form(None),
    detection_model: str = Form("auto"), # Changed to str
    recognition_model: str = Form("facenet"), # Changed to str
    confidence_threshold: float = Form(0.5), # Changed to float
    max_faces: int = Form(50), # Changed to int
    service: FaceAnalysisService = Depends(get_face_analysis_service)
) -> JSONResponse:
    """Comprehensive face analysis"""
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
            gallery = json.loads(gallery_json)

        # Create analysis config
        # from src.ai_services.face_analysis.models import (
        #     AnalysisConfig, AnalysisMode, QualityLevel # Already imported at top
        # )

        try:
            config = AnalysisConfig(
                mode=AnalysisMode(mode),
                detection_model=detection_model,
                recognition_model=recognition_model,
                confidence_threshold=confidence_threshold,
                max_faces=max_faces,
                enable_gallery_matching=gallery is not None,
                quality_level=QualityLevel.BALANCED
            )
        except ValueError as e:
            logger.error(f"Invalid analysis configuration: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid analysis configuration: {str(e)}"
            )
        except Exception as e: # Catch any other unexpected error during config creation
            logger.error(f"Config creation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Config creation failed: {str(e)}"
            )

        # Analyze faces
        result = await service.analyze_faces(
            image=image,
            config=config,
            gallery=gallery
        )

        return JSONResponse(content=result.to_dict())

    except Exception as e:
        detail = f"Analysis failed: {str(e)}"
        raise HTTPException(status_code=500, detail=detail)

@face_analysis_router.post("/face-analysis/analyze-base64")
async def analyze_faces_base64(
    request: AnalysisRequest,
    service: FaceAnalysisService = Depends(get_face_analysis_service)
) -> JSONResponse:
    """Comprehensive face analysis with base64 image"""
    try:
        # Decode image
        image_data = base64.b64decode(request.image_base64)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Create analysis config
        # from src.ai_services.face_analysis.models import (
        #     AnalysisConfig, AnalysisMode, QualityLevel # Already imported at top
        # )

        config_dict = request.config or {}

        try:
            # Ensure defaults are of the correct type for AnalysisConfig
            conf_thresh_val = config_dict.get("confidence_threshold", 0.5)
            if not isinstance(conf_thresh_val, (float, int)):
                conf_thresh_val = 0.5

            max_faces_val = config_dict.get("max_faces", 50)
            if not isinstance(max_faces_val, int):
                max_faces_val = 50

            quality_level_str = config_dict.get("quality_level", "balanced")
            try:
                quality_level_val = QualityLevel(quality_level_str)
            except ValueError:
                quality_level_val = QualityLevel.BALANCED

            config = AnalysisConfig(
                mode=AnalysisMode(request.mode),
                detection_model=config_dict.get("detection_model", "auto"),
                recognition_model=config_dict.get("recognition_model", "facenet"),
                confidence_threshold=float(conf_thresh_val),
                max_faces=int(max_faces_val),
                enable_gallery_matching=request.gallery is not None,
                quality_level=quality_level_val
            )
        except ValueError as e:
            logger.error(f"Invalid analysis configuration: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid analysis configuration: {str(e)}"
            )
        except Exception as e: # Catch any other unexpected error during config creation
            logger.error(f"Config creation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Config creation failed: {str(e)}"
            )

        # Analyze faces
        result = await service.analyze_faces(
            image=image,
            config=config,
            gallery=request.gallery
        )

        return JSONResponse(content=result.to_dict())

    except Exception as e:
        detail = f"Analysis failed: {str(e)}"
        raise HTTPException(status_code=500, detail=detail)

@face_analysis_router.post("/face-analysis/batch")
async def batch_analyze_faces(
    files: List[UploadFile] = File(...),
    mode: str = Form("full_analysis"),
    gallery_json: Optional[str] = Form(None),
    service: FaceAnalysisService = Depends(get_face_analysis_service)
) -> JSONResponse:
    """Batch face analysis for multiple images"""
    try:
        # Read all images
        images = []
        for file_item in files:  # Renamed to avoid File type hint conflict
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
            gallery = json.loads(gallery_json)

        # Create analysis config
        # from src.ai_services.face_analysis.models import (
        #     AnalysisConfig, AnalysisMode, QualityLevel # Already imported at top
        # )

        try:
            config = AnalysisConfig(
                mode=AnalysisMode(mode),
                enable_gallery_matching=gallery is not None,
                quality_level=QualityLevel.BALANCED,
                parallel_processing=True
                # Add other fields with defaults if AnalysisConfig requires them
                # e.g., detection_model="auto", recognition_model="facenet",
                # confidence_threshold=0.5, max_faces=50,
            )
        except ValueError as e:
            logger.error(f"Invalid analysis configuration for batch: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid analysis configuration for batch: {str(e)}"
            )
        except Exception as e: # Catch any other unexpected error during config creation
            logger.error(f"Config creation failed for batch: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Config creation failed for batch: {str(e)}"
            )

        # Batch analyze
        result = await service.batch_analyze(
            images=images,
            config=config,
            gallery=gallery
        )

        return JSONResponse(content=result.to_dict())

    except Exception as e:
        detail = f"Batch analysis failed: {str(e)}"
        raise HTTPException(status_code=500, detail=detail)

# Export routers
__all__ = ["face_detection_router", "face_recognition_router", "face_analysis_router"]
