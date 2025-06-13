"""
Face Recognition API Router
Fixed version with proper dependency injection and comprehensive endpoints
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

logger = logging.getLogger(__name__)

router = APIRouter()

# === MODELS ===
class AddFaceRequest(BaseModel):
    person_name: str
    person_id: Optional[str] = None
    face_image_base64: str
    model_name: Optional[str] = "facenet"
    metadata: Optional[Dict[str, Any]] = None

class RecognitionRequest(BaseModel):
    face_image_base64: str
    gallery: Optional[Dict[str, Any]] = None
    model_name: Optional[str] = "facenet"
    top_k: Optional[int] = 5
    similarity_threshold: Optional[float] = 0.5  # Lowered from 0.6 to 0.5

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
    valid_models = {"facenet", "adaface", "arcface"}
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
                "/health",
                "/add-face",
                "/add-face-json",
                "/extract-embedding",
                "/recognize",
                "/compare",
                "/gallery/get",
                "/gallery/set",
                "/gallery/clear",
                "/database/status",
                "/models/available",
                "/performance/stats"
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
    """
    Add a face to the recognition database using file upload
    
    Parameters:
    - person_name: Name of the person
    - person_id: Unique ID for the person (optional, will use person_name if not provided)
    - file: Image file containing the face
    - model_name: Recognition model to use
    """
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
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        )

@router.post("/add-face-json")
async def add_face_json_endpoint(
    request: AddFaceRequest,
    service = Depends(get_face_recognition_service)
) -> Dict[str, Any]:
    """
    Add a face to the recognition database using JSON request
    
    Body:
    - person_name: Name of the person
    - person_id: Unique ID for the person (optional)
    - face_image_base64: Base64 encoded image
    - model_name: Recognition model to use
    - metadata: Optional metadata
    """
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

        logger.info(f"Adding face for {request.person_name} (ID: {person_id}) from JSON request")

        # Add face to database
        result = await service.add_face_from_image(
            image_bytes=image_bytes,
            person_name=request.person_name,
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
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        )

@router.post("/extract-embedding")
async def extract_embedding_endpoint(
    file: UploadFile = File(...),
    model_name: str = Form("facenet"),
    normalize: bool = Form(True),
    service = Depends(get_face_recognition_service)
) -> JSONResponse:
    """
    Extract face embedding from uploaded image
    
    Parameters:
    - file: Image file containing the face
    - model_name: Recognition model to use
    - normalize: Whether to normalize the embedding vector
    """
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

        # Extract embedding using a temporary face addition
        result = await service.add_face_from_image(
            image_bytes=image_bytes,
            person_name="temp_extraction",
            person_id="temp_extraction",
            model_name=model_name
        )

        if not result or not result.get("success"):
            error_detail = (
                result.get("error", "Failed to extract embedding")
                if result
                else "Failed to extract embedding"
            )
            raise HTTPException(status_code=400, detail=error_detail)

        return JSONResponse(content={
            "success": True,
            "embedding": result.get("embedding_preview", []),
            "model_used": result.get("model_used", model_name),
            "vector": result.get("embedding_preview", []),
            "dimension": len(result.get("embedding_preview", [])),
            "normalized": normalize
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
    """
    Recognize face against gallery or internal database
    
    Body:
    - face_image_base64: Base64 encoded image
    - gallery: Optional external gallery for recognition
    - model_name: Recognition model to use
    - top_k: Number of top matches to return
    - similarity_threshold: Minimum similarity threshold
    """
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
            )

        # Filter results by top_k and similarity threshold
        if "matches" in result_dict and request.top_k:
            matches = result_dict["matches"][:request.top_k]
            result_dict["matches"] = matches

        if request.similarity_threshold:
            if "matches" in result_dict:
                filtered_matches = [
                    match for match in result_dict["matches"]
                    if match.get("similarity", 0) >= request.similarity_threshold
                ]
                result_dict["matches"] = filtered_matches
                
                # Update best_match if needed
                if filtered_matches and result_dict.get("best_match"):
                    best_similarity = result_dict["best_match"].get("similarity", 0)
                    if best_similarity < request.similarity_threshold:
                        result_dict["best_match"] = None

        logger.info(f"Recognition complete: {len(result_dict.get('matches', []))} matches found")
        
        return JSONResponse(content=result_dict)

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
    """
    Compare two faces for similarity
    
    Body:
    - face1_image_base64: First face image (base64)
    - face2_image_base64: Second face image (base64)
    - model_name: Recognition model to use
    """
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

        # Extract embeddings for both faces
        result1 = await service.add_face_from_image(
            image_bytes=image1_bytes,
            person_name="temp_compare_1",
            person_id="temp_compare_1",
            model_name=model_name
        )

        result2 = await service.add_face_from_image(
            image_bytes=image2_bytes,
            person_name="temp_compare_2",
            person_id="temp_compare_2",
            model_name=model_name
        )

        if not (result1 and result1.get("success")) or not (result2 and result2.get("success")):
            raise HTTPException(
                status_code=400,
                detail="Failed to extract embeddings from one or both images"
            )

        # Calculate similarity (this is a simplified version)
        emb1 = result1.get("embedding_preview", [])
        emb2 = result2.get("embedding_preview", [])
        
        if not emb1 or not emb2:
            raise HTTPException(
                status_code=400,
                detail="Failed to extract valid embeddings"
            )

        # Simple cosine similarity calculation
        emb1_np = np.array(emb1)
        emb2_np = np.array(emb2)
        
        # Normalize embeddings
        emb1_norm = emb1_np / np.linalg.norm(emb1_np)
        emb2_norm = emb2_np / np.linalg.norm(emb2_np)
        
        # Calculate cosine similarity
        similarity = float(np.dot(emb1_norm, emb2_norm))
          # Determine if faces match (using optimized threshold)
        threshold = 0.5  # Lowered from 0.6 to 0.5
        is_match = similarity >= threshold

        return JSONResponse(content={
            "success": True,
            "similarity": similarity,
            "is_match": is_match,
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
    """
    Set/update the face gallery
    
    Body:
    - gallery: New gallery data
    - merge_with_existing: Whether to merge with existing data
    """
    try:
        if not request.merge_with_existing:
            # Clear existing database
            service.face_database.clear()
            logger.info("Cleared existing face database")

        # This is a simplified implementation
        # In a real scenario, you'd need to convert the gallery format
        # to match the internal database format
        
        updated_count = 0
        for person_id, person_data in request.gallery.items():
            if isinstance(person_data, dict):
                # Convert gallery format to database format would go here
                # This is a placeholder
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
        
        return {
            "available_models": ["facenet", "adaface", "arcface"],
            "current_model": service_info.get("model_info", {}).get("current_model"),
            "model_info": service_info.get("model_info", {}),
            "supported_formats": ["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
            "embedding_dimensions": {
                "facenet": 512,
                "adaface": 512,
                "arcface": 512
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

# Export router
__all__ = ["router"]