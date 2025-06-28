"""
Face Recognition API Router - Fixed Version
แก้ไขปัญหาการเรียกใช้ method และ handling ของ embedding extraction
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List, Union
import cv2
import numpy as np
from pydantic import BaseModel
import base64
import logging
import os
import time
import uuid

logger = logging.getLogger(__name__)

router = APIRouter()

# === MODELS ===
class AddFaceRequest(BaseModel):
    person_name: str
    person_id: Optional[str] = None
    face_image_base64: str
    model_name: Optional[str] = "facenet"
    metadata: Optional[Dict[str, Any]] = None
    fast_mode: Optional[bool] = False  # Add fast mode option
    
    # Simple Processing parameters
    processing_mode: Optional[str] = None  # "simple" or "enhanced"
    skip_preprocessing: Optional[bool] = None
    use_fast_detection: Optional[bool] = None
    
    # DeepFace specific parameters
    deepface_enforce_detection: Optional[bool] = None
    deepface_detector_backend: Optional[str] = None
    deepface_align: Optional[bool] = None
    deepface_normalization: Optional[str] = None

class RecognitionRequest(BaseModel):
    face_image_base64: str
    gallery: Optional[Dict[str, Any]] = None
    model_name: Optional[str] = "facenet"
    top_k: Optional[int] = 5
    similarity_threshold: Optional[float] = 0.5
    
    # Simple Processing parameters
    processing_mode: Optional[str] = None  # "simple" or "enhanced"
    skip_preprocessing: Optional[bool] = None
    use_fast_detection: Optional[bool] = None
    
    # DeepFace specific parameters
    deepface_enforce_detection: Optional[bool] = None
    deepface_detector_backend: Optional[str] = None
    deepface_align: Optional[bool] = None
    deepface_normalization: Optional[str] = None

class EmbeddingRequest(BaseModel):
    face_image_base64: str
    model_name: Optional[str] = "facenet"
    normalize: bool = True

class CompareRequest(BaseModel):
    face1_image_base64: str
    face2_image_base64: str
    model_name: Optional[str] = "facenet"

class GalleryUpdateRequest(BaseModel):
    gallery: Dict[str, Any]
    merge_with_existing: bool = False

class MultipleRegistrationRequest(BaseModel):
    full_name: str
    employee_id: str
    department: Optional[str] = None
    position: Optional[str] = None  
    model_name: Optional[str] = "adaface"
    images: List[str]  # List of base64 encoded images
    metadata: Optional[Dict[str, Any]] = None

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

def validate_model_name(model_name: str) -> str:
    """Validate and normalize model name"""
    # ONNX Models (primary)
    onnx_models = {"facenet", "adaface", "arcface"}
    
    # Framework Models (secondary)
    framework_models = {"deepface", "facenet_pytorch", "dlib", "insightface", "edgeface"}
    
    # All valid models
    valid_models = onnx_models | framework_models
    
    model_name = model_name.lower().strip()
    
    if model_name not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model name: {model_name}. Must be one of: {', '.join(valid_models)}"
        )
    
    return model_name

def validate_person_id(person_id: str) -> str:
    """Validate person ID format"""
    if not person_id or not person_id.strip():
        raise HTTPException(status_code=400, detail="Person ID cannot be empty")
    
    # Remove dangerous characters
    person_id = person_id.strip()
    if len(person_id) > 100:
        raise HTTPException(status_code=400, detail="Person ID too long (max 100 characters)")
    
    return person_id

# === DEPENDENCY INJECTION ===
def get_face_recognition_service(request: Request):
    """Dependency to get face recognition service from app.state"""
    service = getattr(request.app.state, "face_recognition_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Face recognition service not available or not initialized properly."
        )
    return service

# === API ENDPOINTS ===
@router.get("/health")
async def health_check(
    service = Depends(get_face_recognition_service)
) -> Dict[str, Any]:
    """Health check endpoint"""
    try:
        service_info = service.get_service_info()
        return {
            "status": "healthy",
            "service": "face_recognition",
            "service_info": service_info,
            "endpoints": [
                "/health", "/add-face", "/add-face-json", "/extract-embedding",
                "/recognize", "/compare", "/gallery/get", "/gallery/set",
                "/gallery/clear", "/database/status", "/models/available", "/performance/stats"
            ]
        }
    except Exception as e:
        logger.error(f"Face recognition health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.post("/add-face")
async def add_face_endpoint(
    person_name: str = Form(...),
    person_id: Optional[str] = Form(None),
    file: UploadFile = File(...),
    model_name: str = Form("facenet"),
    service = Depends(get_face_recognition_service)
) -> Dict[str, Any]:
    """Add a face to the recognition database using file upload"""
    try:
        # Validate inputs
        model_name = validate_model_name(model_name)
        person_id = validate_person_id(person_id or person_name)
        
        # Validate file format
        if not validate_image_format(file):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file.content_type}"
            )

        # Read image
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        logger.info(f"Adding face for {person_name} (ID: {person_id}) from file: {file.filename}")

        # Add face to database using the service
        result = await service.add_face_from_image(
            image_bytes=image_bytes,
            person_name=person_name,
            person_id=person_id,
            model_name=model_name
        )

        if not result or not result.get("success"):
            error_detail = (
                result.get("error", "Failed to add face due to an unknown error.")
                if result
                else "Failed to add face."
            )
            raise HTTPException(status_code=400, detail=error_detail)

        logger.info(f"Successfully added face for {person_name}")
        
        return {
            "success": True,
            "message": f"Face for {person_name} added successfully.",
            "person_name": person_name,
            "person_id": person_id,
            "face_ids": result.get("face_ids", []),
            "model_used": result.get("model_used"),
            "embeddings_count": result.get("embeddings_count", 1),
            "embedding_preview": result.get("embedding_preview", [])[:5]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in add_face_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/add-face-json")
async def add_face_json_endpoint(
    request: AddFaceRequest,
    service = Depends(get_face_recognition_service)
) -> Dict[str, Any]:
    """Add a face to the recognition database using JSON request"""
    try:
        # Validate inputs
        model_name = validate_model_name(request.model_name or "facenet")
        person_id = validate_person_id(request.person_id or request.person_name)        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(request.face_image_base64)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 image data: {e}"
            )

        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image data")

        logger.info(f"Adding face for {request.person_name} (ID: {person_id}) from JSON request (fast_mode: {request.fast_mode})")        # Add face to database with all parameters
        # If any "simple processing" parameters are set, enable fast_mode
        use_fast_mode = (
            request.fast_mode or 
            request.processing_mode == "simple" or 
            request.skip_preprocessing == True
        )
        
        # Prepare additional options for DeepFace
        additional_options = {}
        if model_name == "deepface":
            if request.deepface_enforce_detection is not None:
                additional_options["enforce_detection"] = request.deepface_enforce_detection
            if request.deepface_detector_backend is not None:
                additional_options["detector_backend"] = request.deepface_detector_backend
            if request.deepface_align is not None:
                additional_options["align"] = request.deepface_align
            if request.deepface_normalization is not None:
                additional_options["normalization"] = request.deepface_normalization
        
        result = await service.add_face_from_image(
            image_bytes=image_bytes,
            person_name=request.person_name,
            person_id=person_id,
            model_name=model_name,
            fast_mode=use_fast_mode,
            **additional_options
        )

        if not result or not result.get("success"):
            error_detail = (
                result.get("error", "Failed to add face due to an unknown error.")
                if result
                else "Failed to add face."
            )
            raise HTTPException(status_code=400, detail=error_detail)

        logger.info(f"Successfully added face for {request.person_name}")

        return {
            "success": True,
            "message": f"Face for {request.person_name} added successfully.",
            "person_name": request.person_name,
            "person_id": person_id,
            "face_ids": result.get("face_ids", []),
            "model_used": result.get("model_used"),
            "embeddings_count": result.get("embeddings_count", 1),
            "embedding_preview": result.get("embedding_preview", [])[:5]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in add_face_json_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/extract-embedding")
async def extract_embedding_endpoint(
    file: UploadFile = File(...),
    model_name: str = Form("facenet"),
    normalize: bool = Form(True),
    service = Depends(get_face_recognition_service)
) -> JSONResponse:
    """Extract face embedding from uploaded image - ไม่สร้าง database entries"""
    try:
        # Validate inputs
        model_name = validate_model_name(model_name)
        
        # Validate file format
        if not validate_image_format(file):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file.content_type}"
            )

        # Read image
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        logger.info(f"Extracting embedding from {file.filename} using {model_name}")

        # 🔧 แก้ไข: ใช้วิธีที่ไม่สร้าง database entries
        try:
            # วิธีที่ 1: ใช้ extract_embedding_only (ถ้ามี)
            if hasattr(service, 'extract_embedding_only'):
                embedding_result = await service.extract_embedding_only(
                    image_bytes=image_bytes,
                    model_name=model_name,
                    normalize=normalize
                )
            else:
                # วิธีที่ 2: ใช้ _extract_embedding_unified โดยตรง (ไม่ผ่าน database)
                img_np = service._decode_image(image_bytes, "embedding_extraction")
                if img_np is None:
                    raise HTTPException(status_code=400, detail="Failed to decode image")
                
                embedding_vector = await service._extract_embedding_unified(img_np, model_name)
                if embedding_vector is None:
                    raise HTTPException(status_code=400, detail="Failed to extract embedding")
                
                embedding_result = {
                    "success": True,
                    "embedding": embedding_vector.tolist(),
                    "full_embedding": embedding_vector.tolist(),
                    "embedding_preview": embedding_vector[:5].tolist(),
                    "model_used": model_name,
                    "dimension": len(embedding_vector),
                    "normalized": normalize
                }
                
        except Exception as e:
            logger.error(f"Direct embedding extraction failed: {e}")
            # 🚨 หลีกเลี่ยงการใช้ add_face_from_image ที่สร้าง temp
            raise HTTPException(status_code=400, detail=f"Embedding extraction failed: {str(e)}")

        if not embedding_result or not embedding_result.get("success"):
            raise HTTPException(status_code=400, detail="Failed to extract embedding")

        return JSONResponse(content={
            "success": True,
            "embedding": embedding_result.get("embedding", []),
            "model_used": embedding_result.get("model_used", model_name),
            "vector": embedding_result.get("embedding", []),
            "dimension": embedding_result.get("dimension", 0),
            "full_embedding": embedding_result.get("full_embedding", []),
            "embedding_preview": embedding_result.get("embedding_preview", []),
            "normalized": normalize,
            "extraction_only": True,  # 🔧 บอกว่าเป็น extraction อย่างเดียว
            "no_database_entry": True  # 🔧 ไม่สร้าง database entry
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {str(e)}")

@router.post("/recognize")
async def recognize_face_endpoint(
    request: RecognitionRequest,
    service = Depends(get_face_recognition_service)
) -> JSONResponse:
    """Recognize face against gallery or internal database"""
    try:
        # Validate inputs
        model_name = validate_model_name(request.model_name or "facenet")
        
        # Decode image
        try:
            image_bytes = base64.b64decode(request.face_image_base64)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 image data: {e}"
            )

        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image data")

        logger.info(f"Recognizing face using {model_name}")

        # Perform recognition
        if request.gallery:
            # Use external gallery
            result_dict = await service.recognize_faces_with_gallery(
                image_bytes=image_bytes,
                gallery=request.gallery,
                model_name=model_name
            )
        else:
            # Use internal database
            result_dict = await service.recognize_faces(
                image_bytes=image_bytes,
                model_name=model_name
            )        # แก้ไข: ตรวจสอบโครงสร้างของ result ให้ถูกต้อง
        if not result_dict or not result_dict.get("success", True):
            return JSONResponse(content={
                "success": False,
                "error": result_dict.get("error", "Recognition failed") if result_dict else "Recognition failed",
                "matches": [],
                "best_match": None
            })

        # ดึงผลลัพธ์ที่ถูกต้อง
        matches = result_dict.get("matches", result_dict.get("results", []))
        
        # ตรวจสอบว่า matches ไม่เป็น None
        if matches is None:
            matches = []
          # Filter results by top_k and similarity threshold
        if matches and request.top_k:
            matches = matches[:request.top_k]

        if request.similarity_threshold and matches:
            filtered_matches = [
                match for match in matches
                if match.get("similarity", match.get("confidence", 0)) >= request.similarity_threshold
            ]
            matches = filtered_matches

        # หา best match
        best_match = None
        if matches and len(matches) > 0:
            # Sort matches by similarity/confidence
            matches = sorted(matches, key=lambda x: x.get("similarity", x.get("confidence", 0)), reverse=True)
            best_match = matches[0]
            
            # ตรวจสอบ threshold
            if request.similarity_threshold:
                best_similarity = best_match.get("similarity", best_match.get("confidence", 0))
                if best_similarity < request.similarity_threshold:
                    best_match = None

        # ตรวจสอบให้แน่ใจว่า matches ไม่เป็น None ก่อนใช้ len()
        matches_count = len(matches) if matches else 0
        logger.info(f"Recognition complete: {matches_count} matches found")
        
        return JSONResponse(content={
            "success": True,
            "matches": matches or [],
            "best_match": best_match,
            "top_match": best_match,  # Legacy field
            "results": matches or [],  # Legacy field
            "message": f"Found {matches_count} potential match(es).",
            "query_embedding": result_dict.get("query_embedding", []),
            "processing_time": result_dict.get("processing_time", 0.0),
            "total_candidates": result_dict.get("total_candidates", 0)
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recognition failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")

@router.post("/compare")
async def compare_faces_endpoint(
    request: CompareRequest,
    service = Depends(get_face_recognition_service)
) -> JSONResponse:
    """Compare two faces for similarity - Fixed Version"""
    try:
        # Validate inputs
        model_name = validate_model_name(request.model_name or "facenet")
        
        # Decode images
        try:
            image1_bytes = base64.b64decode(request.face1_image_base64)
            image2_bytes = base64.b64decode(request.face2_image_base64)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 image data: {e}"
            )

        logger.info(f"Comparing two faces using {model_name}")

        # แก้ไข: ใช้ method เฉพาะสำหรับ comparison
        try:
            # ลองใช้ method compare_faces ถ้ามี
            comparison_result = await service.compare_faces(
                image1_bytes=image1_bytes,
                image2_bytes=image2_bytes,
                model_name=model_name
            )
            
            return JSONResponse(content=comparison_result)
            
        except AttributeError:
            # Fallback: ใช้ embedding extraction manual
            logger.info("Using manual embedding extraction for face comparison")
            
            # Extract embeddings for both faces
            embedding1 = await service.extract_embedding_only(
                image_bytes=image1_bytes,
                model_name=model_name
            )
            
            embedding2 = await service.extract_embedding_only(
                image_bytes=image2_bytes,
                model_name=model_name
            )
            
            if not (embedding1 and embedding1.get("success")) or not (embedding2 and embedding2.get("success")):
                raise HTTPException(
                    status_code=400,
                    detail="Failed to extract embeddings from one or both images"
                )

            # Calculate similarity
            emb1 = embedding1.get("embedding", [])
            emb2 = embedding2.get("embedding", [])
            
            if not emb1 or not emb2:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to extract valid embeddings"
                )

            # Simple cosine similarity calculation
            emb1_np = np.array(emb1, dtype=np.float32)
            emb2_np = np.array(emb2, dtype=np.float32)
            
            # Normalize embeddings
            emb1_norm = emb1_np / np.linalg.norm(emb1_np)
            emb2_norm = emb2_np / np.linalg.norm(emb2_np)
            
            # Calculate cosine similarity
            similarity = float(np.dot(emb1_norm, emb2_norm))
            
            # Determine if faces match (using optimized threshold)
            threshold = 0.5
            is_match = similarity >= threshold

            return JSONResponse(content={
                "success": True,
                "similarity": similarity,
                "is_match": is_match,
                "is_same_person": is_match,
                "threshold_used": threshold,
                "model_used": model_name,
                "confidence": similarity,
                "distance": 1.0 - similarity
            })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Face comparison failed: {str(e)}")

@router.get("/gallery/get")
async def get_gallery_endpoint(
    service = Depends(get_face_recognition_service)
) -> JSONResponse:
    """Get the current face gallery/database"""
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
                        "embeddings": embedding_vectors,
                        "embedding_count": len(embedding_vectors)
                    }
                elif isinstance(embeddings_list, list):
                    gallery_data_transformed[person_id] = {
                        "name": person_id,
                        "embeddings": [],
                        "embedding_count": 0
                    }
                    
        return JSONResponse(content=gallery_data_transformed)
        
    except Exception as e:
        logger.error(f"Failed to get gallery: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to retrieve gallery: {str(e)}"
        )

@router.post("/gallery/set")
async def set_gallery_endpoint(
    request: GalleryUpdateRequest,
    service = Depends(get_face_recognition_service)
) -> Dict[str, Any]:
    """Set/update the face gallery"""
    try:
        if not request.merge_with_existing:
            # Clear existing database
            service.face_database.clear()
            logger.info("Cleared existing face database")

        # This is a simplified implementation
        updated_count = 0
        for person_id, person_data in request.gallery.items():
            if isinstance(person_data, dict):
                updated_count += 1

        return {
            "success": True,
            "message": f"Gallery updated successfully",
            "updated_persons": updated_count,
            "total_persons": len(service.face_database),
            "merge_mode": request.merge_with_existing
        }
        
    except Exception as e:
        logger.error(f"Failed to set gallery: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to set gallery: {str(e)}"
        )

@router.delete("/gallery/clear")
async def clear_gallery_endpoint(
    service = Depends(get_face_recognition_service)
) -> Dict[str, Any]:
    """Clear the entire face gallery/database"""
    try:
        persons_count = len(service.face_database)
        service.face_database.clear()
        
        logger.info(f"Cleared face database ({persons_count} persons removed)")
        
        return {
            "success": True,
            "message": "Face gallery cleared successfully",
            "persons_removed": persons_count
        }
        
    except Exception as e:
        logger.error(f"Failed to clear gallery: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to clear gallery: {str(e)}"
        )

@router.get("/database/status")
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

@router.get("/models/available")
async def get_available_models(
    service = Depends(get_face_recognition_service)
) -> Dict[str, Any]:
    """Get list of available recognition models"""
    try:
        service_info = service.get_service_info()
        
        # Debug: Log service configuration
        logger.info(f"🔍 API Debug - Service instance: {id(service)}")
        logger.info(f"🔍 API Debug - Multi-framework enabled: {getattr(service, 'enable_multi_framework', False)}")
        logger.info(f"🔍 API Debug - Requested frameworks: {getattr(service, 'requested_frameworks', [])}")
        
        # Get available frameworks from the service
        try:
            available_frameworks = service.get_available_frameworks()
            logger.info(f"🔍 API Debug - Available frameworks returned: {available_frameworks}")
        except Exception as e:
            logger.error(f"🔍 API Debug - Framework detection failed: {e}")
            # Fallback to ONNX models if framework detection fails
            available_frameworks = ["facenet", "adaface", "arcface"]
            
        # Calculate total models
        total_models = len(available_frameworks)
        
        return {
            "available_models": available_frameworks,
            "total_models": total_models,
            "onnx_models": ["facenet", "adaface", "arcface"],
            "framework_models": ["deepface", "facenet_pytorch", "dlib", "insightface", "edgeface"],
            "current_model": service_info.get("model_info", {}).get("current_model"),
            "model_info": service_info.get("model_info", {}),
            "multi_framework_enabled": getattr(service, 'enable_multi_framework', False),
            "supported_formats": ["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
            "embedding_dimensions": {
                "facenet": 512,
                "adaface": 512,
                "arcface": 512,
                "deepface": 512,
                "facenet_pytorch": 512,
                "dlib": 128,
                "insightface": 512,
                "edgeface": 512
            },
            "recommended_models": {
                "general": "facenet",
                "accuracy": "adaface",
                "speed": "facenet"
            }
        }
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get available models: {str(e)}")

@router.get("/performance/stats")
async def get_performance_stats(
    service = Depends(get_face_recognition_service)
) -> Dict[str, Any]:
    """Get detailed performance statistics"""
    try:
        performance_stats = service.get_performance_stats()
        service_info = service.get_service_info()
        
        return {
            "performance_stats": performance_stats,
            "service_info": service_info,
            "timestamp": time.time(),
            "service": "face_recognition"
        }
    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance stats: {str(e)}")

@router.post("/register-multiple")
async def register_multiple_faces(
    request: MultipleRegistrationRequest,
    service = Depends(get_face_recognition_service)
) -> Dict[str, Any]:
    """
    ลงทะเบียนใบหน้าจากหลายภาพ (สำหรับ Real-time Face Registration)
    """
    try:
        start_time = time.time()
        logger.info(f"🔄 Starting multiple face registration for: {request.full_name}")
        
        if len(request.images) == 0:
            raise HTTPException(status_code=400, detail="No images provided")
        
        if len(request.images) > 20:
            raise HTTPException(status_code=400, detail="Too many images (max 20)")
        
        all_embeddings = []
        successful_images = 0
        failed_images = 0
          # ประมวลผลแต่ละภาพ
        for i, image_base64 in enumerate(request.images):
            try:
                logger.info(f"📸 Processing image {i+1}/{len(request.images)}")
                
                # Decode base64 image
                image_data = base64.b64decode(image_base64)
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    logger.warning(f"⚠️ Failed to decode image {i+1}")
                    failed_images += 1
                    continue
                    
                # Extract embedding using the specified model
                logger.info(f"🔍 Extracting embedding for image {i+1} using model: {request.model_name}")
                embedding_start_time = time.time()
                
                embedding_result = await service.extract_embedding_only(
                    image_bytes=cv2.imencode('.jpg', image)[1].tobytes(),
                    model_name=request.model_name
                )
                
                embedding_time = time.time() - embedding_start_time
                logger.info(f"⏱️ Embedding extraction took {embedding_time:.3f}s for image {i+1}")
                
                if embedding_result and embedding_result.get("success", False):
                    embedding = embedding_result.get("embedding")
                    if embedding is not None and len(embedding) > 0:
                        all_embeddings.append(embedding)
                        successful_images += 1
                        logger.info(f"✅ Successfully extracted embedding from image {i+1} (dim: {len(embedding)})")
                    else:
                        logger.warning(f"⚠️ Empty embedding from image {i+1}")
                        failed_images += 1
                else:
                    logger.warning(f"⚠️ Failed to extract embedding from image {i+1}: {embedding_result}")
                    failed_images += 1
                    
            except Exception as e:
                logger.error(f"❌ Error processing image {i+1}: {e}")
                failed_images += 1
                continue
        
        if len(all_embeddings) == 0:
            raise HTTPException(
                status_code=400, 
                detail="No valid embeddings could be extracted from any image"
            )
        
        # คำนวณ average embedding
        logger.info(f"🧮 Computing average embedding from {len(all_embeddings)} embeddings...")
        avg_embedding = np.mean(all_embeddings, axis=0)
        logger.info(f"✅ Average embedding computed (dimension: {len(avg_embedding)})")
        
        # สร้าง person_id ถ้าไม่มี
        person_id = request.employee_id or f"user_{int(time.time())}"
        
        # เตรียม metadata
        metadata = request.metadata or {}
        metadata.update({
            "full_name": request.full_name,
            "employee_id": request.employee_id,
            "department": request.department,
            "position": request.position,
            "registration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_images": len(request.images),
            "successful_images": successful_images,
            "failed_images": failed_images,
            "model_used": request.model_name,
            "embedding_dimension": len(avg_embedding),
            "registration_type": "multiple_images"
        })        # เพิ่มใบหน้าเข้า database ผ่าน direct embedding
        try:
            # สร้าง FaceEmbedding object
            from src.ai_services.face_recognition.models import FaceEmbedding, RecognitionModel
            
            # Convert model name to RecognitionModel enum
            model_type = None
            if request.model_name:
                model_type = RecognitionModel.from_string(request.model_name)
            
            face_embedding = FaceEmbedding(
                id=str(uuid.uuid4()),
                person_id=person_id,
                person_name=request.full_name,
                vector=avg_embedding,
                model_type=model_type,
                quality_score=90.0,  # High quality since it's averaged from multiple images
                metadata=metadata
            )
              # เพิ่มเข้า face_database โดยตรง
            if person_id not in service.face_database:
                service.face_database[person_id] = []
            
            service.face_database[person_id].append(face_embedding)
              # Debug: ตรวจสอบสถานะฐานข้อมูลหลังเพิ่มข้อมูล
            total_people = len(service.face_database)
            total_embeddings = sum(len(faces) for faces in service.face_database.values())
            logger.info(f"🗃️ Database status: {total_people} people, {total_embeddings} total embeddings")
            logger.info(f"🗃️ Added to database - Person ID: {person_id}, Total faces for this person: {len(service.face_database[person_id])}")
            logger.info(f"🔧 Service instance ID: {id(service)}, Database instance ID: {id(service.face_database)}")
            
            logger.info(f"✅ Successfully added face embedding to database for {person_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to add face to database: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to add face to database: {str(e)}"
            )
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Multiple face registration completed successfully in {processing_time:.2f}s")
        
        return {
            "success": True,
            "message": "Face registration completed successfully",
            "user_id": person_id,
            "person_name": request.full_name,
            "employee_id": request.employee_id,
            "processing_stats": {
                "total_images": len(request.images),
                "successful_images": successful_images,
                "failed_images": failed_images,
                "success_rate": f"{(successful_images/len(request.images)*100):.1f}%",
                "processing_time": f"{processing_time:.2f}s",
                "average_time_per_image": f"{processing_time/len(request.images):.2f}s"
            },
            "model_info": {
                "model_name": request.model_name,
                "embedding_dimension": len(avg_embedding),
                "total_embeddings_used": len(all_embeddings)
            },
            "metadata": metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Multiple face registration failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Face registration failed: {str(e)}"
        )

@router.get("/gallery/info")
async def get_gallery_info(
    service = Depends(get_face_recognition_service)
) -> Dict[str, Any]:
    """
    ดึงข้อมูลฐานข้อมูลใบหน้า (Gallery Information)
    """
    try:
        gallery_stats = await service.get_gallery_stats()
        service_info = service.get_service_info()        
        return {
            "success": True,
            "total_people": gallery_stats.get("total_persons", 0),
            "total_embeddings": gallery_stats.get("total_faces", 0),
            "current_model": service_info.get("model_info", {}).get("current_model", "unknown"),            "gallery_stats": gallery_stats,
            "model_info": service_info.get("model_info", {}),
            "last_updated": gallery_stats.get("last_updated"),
            "embedding_dimensions": gallery_stats.get("embedding_dimensions", {}),
            "people_list": gallery_stats.get("people_list", [])
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get gallery info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get gallery information: {str(e)}"
        )

@router.get("/person/{person_id}")
async def get_person_info(
    person_id: str,
    service = Depends(get_face_recognition_service)
) -> Dict[str, Any]:
    """
    ดึงข้อมูลผู้ใช้เฉพาะคนโดยใช้ person_id/UUID
    Get specific user information by person_id/UUID
    """
    try:
        # Validate person_id
        person_id = validate_person_id(person_id)
        
        # Get the database
        database = service.face_database
        
        if person_id not in database:
            raise HTTPException(
                status_code=404,
                detail=f"Person with ID '{person_id}' not found in database"
            )
        
        # Get person data
        embeddings_list = database[person_id]
        
        if not isinstance(embeddings_list, list) or not embeddings_list:
            raise HTTPException(
                status_code=404,
                detail=f"No valid data found for person '{person_id}'"
            )
        
        # Extract person information
        first_embedding = embeddings_list[0]
        person_name = getattr(first_embedding, 'person_name', person_id)
        
        # Count embeddings
        embedding_count = len(embeddings_list)
        valid_embeddings = 0
        embedding_details = []
        
        for i, emb_obj in enumerate(embeddings_list):
            if hasattr(emb_obj, 'vector') and emb_obj.vector is not None:
                valid_embeddings += 1
                embedding_details.append({
                    "index": i,
                    "face_id": getattr(emb_obj, 'face_id', f"face_{i}"),
                    "vector_shape": emb_obj.vector.shape if hasattr(emb_obj.vector, 'shape') else None,
                    "created_at": getattr(emb_obj, 'created_at', None)
                })
        
        return {
            "success": True,
            "person_id": person_id,
            "person_name": person_name,
            "total_embeddings": embedding_count,
            "valid_embeddings": valid_embeddings,
            "embedding_details": embedding_details,
            "found": True,
            "message": f"User information retrieved successfully for '{person_name}'"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get person info for {person_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve person information: {str(e)}"
        )

# Export router
__all__ = ["router"]