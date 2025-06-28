"""
Fast Face Detection API Endpoint using Multi-Model Service
Supports YOLOv11m and YOLOv11n for performance comparison
Optimized for real-time CCTV and webcam usage
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import time
import logging
from typing import Optional
import cv2
import numpy as np
from PIL import Image
import io

# Import our multi-model service
from src.ai_services.face_detection.multi_model_service import MultiModelService, ModelType

logger = logging.getLogger(__name__)

# Global service instance (singleton pattern for performance)
_face_detector_instance: Optional[MultiModelService] = None

async def get_face_detector() -> MultiModelService:
    """Get or create the global face detector instance"""
    global _face_detector_instance
    
    if _face_detector_instance is None:
        logger.info("🚀 Initializing Multi-Model Face Detection Service...")
        _face_detector_instance = MultiModelService()
        
        if not await _face_detector_instance.initialize():
            logger.error("❌ Failed to initialize Multi-Model Service")
            raise HTTPException(status_code=500, detail="Face detection service initialization failed")
        
        logger.info("✅ Multi-Model Face Detection Service initialized successfully")
    
    return _face_detector_instance

router = APIRouter()

@router.post("/detect-fast")
async def detect_faces_fast(
    file: UploadFile = File(...),
    model_name: str = Form("yolov11m"),  # Support yolov11m or yolov11n
    conf_threshold: float = Form(0.3),
    iou_threshold: float = Form(0.4),
    max_faces: Optional[int] = Form(20),
    min_quality_threshold: float = Form(20.0),
):
    """
    Fast face detection endpoint with multi-model support
    
    Args:
        file: Image file to process
        model_name: Model to use (yolov11m or yolov11n)
        conf_threshold: Confidence threshold (0.0-1.0)
        iou_threshold: IoU threshold for NMS (0.0-1.0)
        max_faces: Maximum number of faces to return
        min_quality_threshold: Minimum quality score for faces
    
    Returns:
        JSON response with detected faces and performance metrics
    """
    start_time = time.time()
    
    try:
        # Get service instance
        detector = await get_face_detector()
        
        # Read and validate image
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Convert to OpenCV format
        image = Image.open(io.BytesIO(contents))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Determine model type
        try:
            if model_name.lower() == "yolov11n":
                model_type = ModelType.YOLO11N
            else:
                model_type = ModelType.YOLO11M  # Default
        except:
            model_type = ModelType.YOLO11M
        
        # Run detection
        result = await detector.detect_faces(
            image=image_cv,
            model_type=model_type,
            conf_threshold=conf_threshold,
            max_faces=max_faces,
            min_quality=min_quality_threshold
        )
        
        if result.error:
            logger.error(f"Detection error: {result.error}")
            raise HTTPException(status_code=500, detail=f"Detection failed: {result.error}")
        
        # Format response
        faces_data = []
        for face in result.faces:
            x1, y1, x2, y2 = face.bbox
            faces_data.append({
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1), 
                    "x2": int(x2),
                    "y2": int(y2)
                },
                "confidence": round(face.confidence, 3),
                "quality_score": round(face.quality_score, 1) if face.quality_score else None
            })
        
        total_time_ms = (time.time() - start_time) * 1000
        
        return JSONResponse({
            "success": True,
            "faces": faces_data,
            "count": len(faces_data),
            "processing_time_ms": round(result.processing_time_ms, 2),
            "total_time_ms": round(total_time_ms, 2),
            "model_used": result.model_used,
            "image_shape": {
                "height": result.image_shape[0],
                "width": result.image_shape[1],
                "channels": result.image_shape[2]
            },
            "parameters": {
                "model_name": model_name,
                "conf_threshold": conf_threshold,
                "max_faces": max_faces,
                "min_quality_threshold": min_quality_threshold
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in detect_faces_fast: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/status-fast")
async def get_status_fast():
    """Get fast detection service status with multi-model info"""
    try:
        detector = await get_face_detector()
        
        # Get performance stats and model info
        perf_stats = detector.get_performance_stats()
        model_info = detector.get_model_info()
        
        return JSONResponse({
            "status": "ready",
            "service": "Multi-Model Face Detection Service",
            "current_model": model_info["current_model"],
            "available_models": model_info["available_models"],
            "loaded_models": model_info["loaded_models"],
            "device": model_info["device"],
            "performance": perf_stats,
            "capabilities": {
                "supports_model_switching": True,
                "real_time_optimized": True,
                "max_concurrent_requests": 1
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting status: {e}")
        return JSONResponse({
            "status": "error",
            "error": str(e)
        }, status_code=500)

@router.post("/benchmark")
async def benchmark_detection(
    iterations: int = Form(50),
    model_name: str = Form("yolov11m"),
    image_width: int = Form(640),
    image_height: int = Form(480)
):
    """
    Benchmark face detection performance with specified model
    
    Args:
        iterations: Number of detection iterations
        model_name: Model to benchmark (yolov11m or yolov11n)
        image_width: Test image width
        image_height: Test image height
    
    Returns:
        Benchmark results with timing statistics
    """
    try:
        detector = await get_face_detector()
        
        # Determine model type
        try:
            if model_name.lower() == "yolov11n":
                model_type = ModelType.YOLO11N
            else:
                model_type = ModelType.YOLO11M
        except:
            model_type = ModelType.YOLO11M
        
        # Run benchmark
        logger.info(f"🚀 Starting benchmark: {model_name} ({iterations} iterations)")
        
        results = await detector.benchmark(
            iterations=iterations,
            model_type=model_type,
            image_size=(image_height, image_width)
        )
        
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])
        
        logger.info(f"✅ Benchmark completed: {results['fps']:.2f} FPS")
        
        return JSONResponse({
            "success": True,
            "benchmark_results": results,
            "summary": {
                "model": results["model"],
                "fps": results["fps"],
                "avg_time_ms": results["average_time_ms"],
                "device": results["device"]
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Benchmark error: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")

@router.post("/switch-model")
async def switch_model(model_name: str = Form(...)):
    """
    Switch to a different model
    
    Args:
        model_name: Model to switch to (yolov11m or yolov11n)
    
    Returns:
        Success status and new model info
    """
    try:
        detector = await get_face_detector()
        
        # Determine model type
        try:
            if model_name.lower() == "yolov11n":
                model_type = ModelType.YOLO11N
            elif model_name.lower() == "yolov11m":
                model_type = ModelType.YOLO11M
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported model: {model_name}")
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid model name: {model_name}")
        
        # Switch model
        success = await detector.switch_model(model_type)
        
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to switch to {model_name}")
        
        # Get updated model info
        model_info = detector.get_model_info()
        
        logger.info(f"✅ Successfully switched to {model_name}")
        
        return JSONResponse({
            "success": True,
            "message": f"Successfully switched to {model_name}",
            "model_info": model_info
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Model switch error: {e}")
        raise HTTPException(status_code=500, detail=f"Model switch failed: {str(e)}")

@router.get("/models")
async def get_available_models():
    """Get list of available models"""
    try:
        detector = await get_face_detector()
        model_info = detector.get_model_info()
        
        return JSONResponse({
            "success": True,
            "available_models": model_info["available_models"],
            "current_model": model_info["current_model"],
            "loaded_models": model_info["loaded_models"],
            "model_paths": model_info["model_paths"]
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")
