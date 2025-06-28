"""
Age and Gender Detection API Endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import logging
import time
import base64
import io
from PIL import Image
import asyncio

from ..ai_services.age_gender import (
    get_age_gender_service,
    AgeGenderRequest,
    AgeGenderResult,
    BatchAgeGenderRequest,
    BatchAgeGenderResult,
    AgeGenderStats
)

logger = logging.getLogger(__name__)

# Create router
age_gender_router = APIRouter(
    prefix="/age-gender",
    tags=["Age & Gender Detection"],
    responses={404: {"description": "Not found"}}
)

@age_gender_router.post("/analyze", response_model=AgeGenderResult)
async def analyze_age_gender(
    file: UploadFile = File(..., description="Image file to analyze")
) -> AgeGenderResult:
    """
    Analyze age and gender from uploaded image
    
    - **file**: Image file (JPG, PNG, WebP)
    - Returns age and gender information for all detected faces
    """
    try:
        start_time = time.time()
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPG, PNG, WebP)"
            )
        
        # Read image data
        image_data = await file.read()
        
        if len(image_data) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        # Get service and analyze
        service = await get_age_gender_service()
        result = await service.analyze_age_gender(image_data)
        
        # Add processing time
        result.processing_time = time.time() - start_time
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Age and gender analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@age_gender_router.post("/analyze-base64", response_model=AgeGenderResult)
async def analyze_age_gender_base64(
    request: AgeGenderRequest
) -> AgeGenderResult:
    """
    Analyze age and gender from base64 encoded image
    
    - **image**: Base64 encoded image data
    - **detect_multiple**: Whether to detect multiple faces (default: true)
    - **detector_backend**: Face detector backend (opencv, ssd, dlib, mtcnn, retinaface, mediapipe)
    """
    try:
        start_time = time.time()
        
        # Decode base64 image
        try:
            # Remove data URL prefix if present
            if request.image.startswith('data:'):
                request.image = request.image.split(',')[1]
            
            image_data = base64.b64decode(request.image)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 image data: {str(e)}"
            )
        
        # Validate image
        try:
            img = Image.open(io.BytesIO(image_data))
            img.verify()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format: {str(e)}"
            )
        
        # Get service and analyze
        service = await get_age_gender_service()
        
        # Update detector backend if specified
        if hasattr(service, 'detector_backend'):
            service.detector_backend = request.detector_backend
        
        result = await service.analyze_age_gender(image_data)
        
        # Add processing time
        result.processing_time = time.time() - start_time
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Base64 age and gender analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@age_gender_router.post("/batch-analyze", response_model=BatchAgeGenderResult)
async def batch_analyze_age_gender(
    files: List[UploadFile] = File(..., description="List of image files to analyze")
) -> BatchAgeGenderResult:
    """
    Analyze age and gender from multiple uploaded images
    
    - **files**: List of image files (JPG, PNG, WebP)
    - Returns age and gender information for all detected faces in all images
    """
    try:
        start_time = time.time()
        
        if not files:
            raise HTTPException(
                status_code=400,
                detail="No files provided"
            )
        
        if len(files) > 10:  # Limit batch size
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 files allowed per batch"
            )
        
        # Get service
        service = await get_age_gender_service()
        
        # Process all images
        results = []
        total_faces = 0
        
        for i, file in enumerate(files):
            try:
                # Validate file type
                if not file.content_type or not file.content_type.startswith('image/'):
                    logger.warning(f"Skipping non-image file: {file.filename}")
                    continue
                
                # Read and analyze
                image_data = await file.read()
                result = await service.analyze_age_gender(image_data)
                
                results.append(result)
                total_faces += result.total_faces
                
            except Exception as e:
                logger.error(f"Failed to process file {i}: {e}")
                # Add failed result
                results.append(AgeGenderResult(
                    success=False,
                    message=f"Failed to process image: {str(e)}",
                    analyses=[],
                    total_faces=0
                ))
        
        # Create batch result
        batch_result = BatchAgeGenderResult(
            success=True,
            message=f"Processed {len(results)} images",
            results=results,
            total_images=len(results),
            total_faces=total_faces,
            processing_time=time.time() - start_time
        )
        
        return batch_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch age and gender analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysis failed: {str(e)}"
        )

@age_gender_router.post("/batch-analyze-base64", response_model=BatchAgeGenderResult)
async def batch_analyze_age_gender_base64(
    request: BatchAgeGenderRequest
) -> BatchAgeGenderResult:
    """
    Analyze age and gender from multiple base64 encoded images
    
    - **images**: List of base64 encoded image data
    - **detect_multiple**: Whether to detect multiple faces per image
    - **detector_backend**: Face detector backend to use
    """
    try:
        start_time = time.time()
        
        if not request.images:
            raise HTTPException(
                status_code=400,
                detail="No images provided"
            )
        
        if len(request.images) > 10:  # Limit batch size
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 images allowed per batch"
            )
        
        # Get service
        service = await get_age_gender_service()
        
        # Update detector backend
        if hasattr(service, 'detector_backend'):
            service.detector_backend = request.detector_backend
        
        # Process all images
        results = []
        total_faces = 0
        
        for i, image_b64 in enumerate(request.images):
            try:
                # Decode base64
                if image_b64.startswith('data:'):
                    image_b64 = image_b64.split(',')[1]
                
                image_data = base64.b64decode(image_b64)
                
                # Analyze
                result = await service.analyze_age_gender(image_data)
                results.append(result)
                total_faces += result.total_faces
                
            except Exception as e:
                logger.error(f"Failed to process image {i}: {e}")
                results.append(AgeGenderResult(
                    success=False,
                    message=f"Failed to process image {i}: {str(e)}",
                    analyses=[],
                    total_faces=0
                ))
        
        # Create batch result
        batch_result = BatchAgeGenderResult(
            success=True,
            message=f"Processed {len(results)} images",
            results=results,
            total_images=len(results),
            total_faces=total_faces,
            processing_time=time.time() - start_time
        )
        
        return batch_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch base64 age and gender analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysis failed: {str(e)}"
        )

@age_gender_router.get("/stats", response_model=AgeGenderStats)
async def get_age_gender_stats(
    results: List[AgeGenderResult]
) -> AgeGenderStats:
    """
    Calculate statistics from age and gender analysis results
    
    - **results**: List of analysis results
    - Returns aggregated statistics
    """
    try:
        if not results:
            raise HTTPException(
                status_code=400,
                detail="No results provided"
            )
        
        # Collect all analyses
        all_analyses = []
        for result in results:
            if result.success:
                all_analyses.extend(result.analyses)
        
        if not all_analyses:
            raise HTTPException(
                status_code=400,
                detail="No successful analyses found"
            )
        
        # Calculate statistics
        ages = [analysis.age for analysis in all_analyses]
        genders = [analysis.gender for analysis in all_analyses]
        
        # Age statistics
        average_age = sum(ages) / len(ages)
        age_range = {"min": min(ages), "max": max(ages)}
        
        # Gender distribution
        gender_dist = {}
        for gender in genders:
            gender_dist[gender] = gender_dist.get(gender, 0) + 1
        
        stats = AgeGenderStats(
            average_age=average_age,
            age_range=age_range,
            gender_distribution=gender_dist,
            total_analyzed=len(all_analyses)
        )
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Statistics calculation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Statistics calculation failed: {str(e)}"
        )

@age_gender_router.get("/service-info")
async def get_service_info() -> Dict[str, Any]:
    """
    Get age and gender detection service information
    
    Returns service details, configuration, and capabilities
    """
    try:
        service = await get_age_gender_service()
        return await service.get_service_info()
        
    except Exception as e:
        logger.error(f"Failed to get service info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get service info: {str(e)}"
        )

@age_gender_router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for age and gender detection service
    """
    try:
        service = await get_age_gender_service()
        
        return {
            "status": "healthy",
            "service": "age_gender_detection",
            "initialized": service._is_initialized,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "age_gender_detection",
            "error": str(e),
            "timestamp": time.time()
        }
