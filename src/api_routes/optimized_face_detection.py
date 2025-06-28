"""
Advanced optimized face detection API endpoints
"""
import time
import asyncio
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import logging

# Import the optimized service
from ...optimized_face_detection_service import optimized_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/face-detection",
    tags=["face-detection-optimized"]
)

@router.on_event("startup")
async def startup_optimized_service():
    """Initialize the optimized service on startup"""
    try:
        await optimized_service.initialize()
        logger.info("Optimized face detection service started successfully")
    except Exception as e:
        logger.error(f"Failed to start optimized service: {e}")

@router.on_event("shutdown")
async def shutdown_optimized_service():
    """Cleanup on shutdown"""
    try:
        await optimized_service.cleanup()
        logger.info("Optimized face detection service cleaned up")
    except Exception as e:
        logger.error(f"Error during optimized service cleanup: {e}")

@router.post("/detect-ultra-fast")
async def detect_faces_ultra_fast(
    file: UploadFile = File(...),
    model_name: str = Form("yolov11m"),
    conf_threshold: float = Form(0.3),
    max_faces: int = Form(15),
    min_quality_threshold: float = Form(18.0)
):
    """
    Ultra-fast face detection with advanced optimizations:
    - Adaptive quality control
    - Intelligent caching
    - Frame skipping
    - Performance monitoring
    """
    start_time = time.time()
    
    try:
        # Validate input
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await file.read()
        
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Process with optimized service
        result = await optimized_service.detect_faces_optimized(
            image_data=image_data,
            conf_threshold=conf_threshold,
            max_faces=max_faces,
            min_quality_threshold=min_quality_threshold
        )
        
        # Add endpoint metadata
        result.update({
            "endpoint": "detect-ultra-fast",
            "total_api_time": time.time() - start_time,
            "model_name": model_name,
            "parameters": {
                "conf_threshold": conf_threshold,
                "max_faces": max_faces,
                "min_quality_threshold": min_quality_threshold
            }
        })
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ultra-fast detection: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Internal server error: {str(e)}",
                "faces": [],
                "processing_time": time.time() - start_time
            }
        )

@router.get("/performance-stats")
async def get_performance_stats():
    """Get current performance statistics"""
    try:
        stats = optimized_service.get_performance_stats()
        return JSONResponse(content={
            "success": True,
            "stats": stats,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Error getting performance stats: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

@router.post("/reset-stats")
async def reset_performance_stats():
    """Reset performance statistics"""
    try:
        optimized_service.reset_performance_stats()
        return JSONResponse(content={
            "success": True,
            "message": "Performance statistics reset"
        })
    except Exception as e:
        logger.error(f"Error resetting stats: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

@router.post("/clear-cache")
async def clear_detection_cache():
    """Clear the detection result cache"""
    try:
        optimized_service.clear_cache()
        return JSONResponse(content={
            "success": True,
            "message": "Detection cache cleared"
        })
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

@router.get("/benchmark-optimized")
async def benchmark_optimized():
    """
    Benchmark the optimized detection service
    """
    try:
        import numpy as np
        from PIL import Image
        import io
        
        # Create test image (640x480 with random noise)
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='JPEG', quality=80)
        image_data = img_byte_arr.getvalue()
        
        # Run multiple detections for benchmarking
        num_tests = 10
        results = []
        
        logger.info(f"Starting optimized benchmark with {num_tests} iterations")
        
        for i in range(num_tests):
            start_time = time.time()
            
            result = await optimized_service.detect_faces_optimized(
                image_data=image_data,
                conf_threshold=0.3,
                max_faces=15,
                min_quality_threshold=18.0
            )
            
            end_time = time.time()
            iteration_time = end_time - start_time
            
            results.append({
                "iteration": i + 1,
                "processing_time": iteration_time,
                "success": result.get("success", False),
                "faces_detected": len(result.get("faces", [])),
                "cached": result.get("cached", False),
                "skipped": result.get("skipped", False),
                "adaptive_quality": result.get("adaptive_quality", 0),
                "scale_factor": result.get("scale_factor", 1.0)
            })
            
            # Small delay between iterations
            await asyncio.sleep(0.01)
        
        # Calculate statistics
        processing_times = [r["processing_time"] for r in results]
        successful_detections = sum(1 for r in results if r["success"])
        cached_results = sum(1 for r in results if r.get("cached", False))
        skipped_results = sum(1 for r in results if r.get("skipped", False))
        
        avg_time = sum(processing_times) / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        fps = 1 / avg_time if avg_time > 0 else 0
        
        # Get current performance stats
        perf_stats = optimized_service.get_performance_stats()
        
        benchmark_result = {
            "success": True,
            "benchmark_type": "optimized",
            "test_iterations": num_tests,
            "successful_detections": successful_detections,
            "cached_results": cached_results,
            "skipped_results": skipped_results,
            "performance": {
                "average_time": round(avg_time, 4),
                "min_time": round(min_time, 4),
                "max_time": round(max_time, 4),
                "estimated_fps": round(fps, 2),
                "total_time": round(sum(processing_times), 4)
            },
            "optimization_stats": perf_stats,
            "detailed_results": results,
            "test_image_size": "640x480",
            "timestamp": time.time()
        }
        
        logger.info(f"Optimized benchmark completed: {fps:.2f} FPS average")
        
        return JSONResponse(content=benchmark_result)
        
    except Exception as e:
        logger.error(f"Benchmark error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Benchmark failed: {str(e)}"
            }
        )

@router.get("/status-optimized")
async def get_optimized_status():
    """Get status of the optimized face detection service"""
    try:
        # Get performance stats
        perf_stats = optimized_service.get_performance_stats()
        
        return JSONResponse(content={
            "success": True,
            "service": "optimized_face_detection",
            "status": "active",
            "model": "yolov11m",
            "features": [
                "adaptive_quality_control",
                "intelligent_caching", 
                "frame_skipping",
                "performance_monitoring",
                "memory_management"
            ],
            "performance": perf_stats,
            "timestamp": time.time()
        })
        
    except Exception as e:
        logger.error(f"Error getting optimized status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )
