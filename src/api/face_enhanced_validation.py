"""
Enhanced Face Validation API
API สำหรับตรวจสอบคุณภาพใบหน้าแบบครบวงจร
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import logging
import asyncio
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# Import enhanced services
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..ai_services.face_detection.landmark_detector import FaceLandmarkDetector
    from ..ai_services.face_detection.enhanced_quality_assessment import EnhancedFaceQualityAssessment
    from ..ai_services.face_analysis.enhanced_registration_service import EnhancedFaceRegistrationService
except ImportError:
    # Fallback for development
    class FaceLandmarkDetector:
        def detect_landmarks(self, image, detailed=False):
            return []
        def cleanup(self):
            pass
    
    class EnhancedFaceQualityAssessment:
        def assess_face_quality(self, image, face_bbox=None, use_landmarks=True):
            return type('QualityMetrics', (), {
                'overall_score': 0.5,
                'is_acceptable': False,
                'rejection_reasons': ['service_unavailable']
            })()
        def get_quality_summary(self, metrics):
            return {'error': 'Service unavailable'}
        def cleanup(self):
            pass
    
    class EnhancedFaceRegistrationService:
        def __init__(self, config=None):
            pass
        async def validate_registration_image(self, image_bytes, person_id, strict_mode=True):
            return {'success': False, 'error': 'Service unavailable'}
        def cleanup(self):
            pass

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/face-enhanced", tags=["Enhanced Face Validation"])

# Global services (will be initialized)
landmark_detector = None
quality_assessor = None
registration_service = None

def initialize_enhanced_services():
    """Initialize enhanced services"""
    global landmark_detector, quality_assessor, registration_service
    
    try:
        if landmark_detector is None:
            landmark_detector = FaceLandmarkDetector()
            logger.info("✅ Landmark detector initialized")
        
        if quality_assessor is None:
            quality_assessor = EnhancedFaceQualityAssessment({
                "sharpness_min": 120.0,
                "frontal_min": 0.8,
                "overall_min": 0.75
            })
            logger.info("✅ Quality assessor initialized")
        
        if registration_service is None:
            registration_service = EnhancedFaceRegistrationService({
                "minimum_quality_score": 0.8,
                "strict_frontal_check": True,
                "require_landmarks": True,
                "max_pose_angle": 12.0
            })
            logger.info("✅ Registration service initialized")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize enhanced services: {e}")
        return False

def cleanup_enhanced_services():
    """Cleanup enhanced services"""
    global landmark_detector, quality_assessor, registration_service
    
    try:
        if landmark_detector:
            landmark_detector.cleanup()
        if quality_assessor:
            quality_assessor.cleanup()
        if registration_service:
            registration_service.cleanup()
        logger.info("🧹 Enhanced services cleaned up")
    except Exception as e:
        logger.warning(f"⚠️ Cleanup warning: {e}")

async def process_uploaded_image(file: UploadFile) -> np.ndarray:
    """Process uploaded image file"""
    try:
        # Read file content
        content = await file.read()
        
        # Convert to numpy array
        img_buffer = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Cannot decode image")
        
        return image, content
    except Exception as e:
        logger.error(f"❌ Image processing failed: {e}")
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

@router.post("/validate-quality")
async def validate_face_quality(
    file: UploadFile = File(...),
    person_id: Optional[str] = Form(None),
    strict_mode: bool = Form(True),
    use_landmarks: bool = Form(True)
):
    """
    ตรวจสอบคุณภาพใบหน้าแบบครบวงจร
    
    Args:
        file: ไฟล์รูปภาพใบหน้า
        person_id: ID ของบุคคล (optional)
        strict_mode: โหมดเข้มงวด
        use_landmarks: ใช้การตรวจจับ landmarks
    
    Returns:
        ผลการตรวจสอบคุณภาพและคำแนะนำ
    """
    try:
        # Initialize services
        if not initialize_enhanced_services():
            raise HTTPException(status_code=500, detail="Failed to initialize enhanced services")
        
        # Process image
        image, image_bytes = await process_uploaded_image(file)
        
        # Validate with registration service
        if registration_service and person_id:
            validation_result = await registration_service.validate_registration_image(
                image_bytes, person_id, strict_mode
            )
            return JSONResponse(content=validation_result)
        
        # Basic quality assessment
        if quality_assessor:
            quality_metrics = quality_assessor.assess_face_quality(
                image, use_landmarks=use_landmarks
            )
            quality_summary = quality_assessor.get_quality_summary(quality_metrics)
            
            # Add landmark analysis
            landmark_analysis = {}
            if landmark_detector and use_landmarks:
                landmark_results = landmark_detector.detect_landmarks(image, detailed=True)
                if landmark_results:
                    best_face = landmark_results[0]
                    landmark_analysis = {
                        "landmarks_found": True,
                        "landmark_count": best_face.get("landmark_count", 0),
                        "is_frontal": landmark_detector.is_frontal_face(best_face, threshold=0.8 if strict_mode else 0.6),
                        "pose_estimation": landmark_detector.get_face_pose(best_face)
                    }
                else:
                    landmark_analysis = {"landmarks_found": False}
            
            return JSONResponse(content={
                "success": quality_metrics.is_acceptable,
                "quality_assessment": quality_summary,
                "landmark_analysis": landmark_analysis,
                "strict_mode": strict_mode,
                "person_id": person_id,
                "processing_info": {
                    "image_size": image.shape[:2],
                    "use_landmarks": use_landmarks,
                    "file_name": file.filename
                }
            })
        
        else:
            raise HTTPException(status_code=500, detail="Quality assessment service unavailable")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Quality validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect-landmarks")
async def detect_face_landmarks(
    file: UploadFile = File(...),
    detailed: bool = Form(True),
    return_pose: bool = Form(True)
):
    """
    ตรวจจับจุดสำคัญบนใบหน้า
    
    Args:
        file: ไฟล์รูปภาพใบหน้า
        detailed: ตรวจจับ landmarks แบบละเอียด (468 จุด)
        return_pose: คืนค่าการประมาณมุมใบหน้า
    
    Returns:
        ข้อมูล landmarks และการวิเคราะห์ใบหน้า
    """
    try:
        # Initialize services
        if not initialize_enhanced_services():
            raise HTTPException(status_code=500, detail="Failed to initialize landmark detection")
        
        # Process image
        image, _ = await process_uploaded_image(file)
        
        # Detect landmarks
        if landmark_detector:
            landmark_results = landmark_detector.detect_landmarks(image, detailed=detailed)
            
            if not landmark_results:
                return JSONResponse(content={
                    "success": False,
                    "message": "No face landmarks detected",
                    "landmarks_found": 0
                })
            
            # Process results
            processed_results = []
            for i, face_data in enumerate(landmark_results):
                result = {
                    "face_index": i,
                    "landmark_count": face_data.get("landmark_count", 0),
                    "type": face_data.get("type", "unknown"),
                    "quality_metrics": face_data.get("quality_metrics", {}),
                    "is_frontal": landmark_detector.is_frontal_face(face_data)
                }
                
                # Add pose estimation if requested
                if return_pose:
                    pose_data = landmark_detector.get_face_pose(face_data)
                    result["pose_estimation"] = pose_data
                
                # Add bounding box if available
                if "bbox" in face_data:
                    result["bounding_box"] = face_data["bbox"]
                
                # Add confidence if available
                if "confidence" in face_data:
                    result["confidence"] = face_data["confidence"]
                
                processed_results.append(result)
            
            return JSONResponse(content={
                "success": True,
                "faces_found": len(processed_results),
                "faces": processed_results,
                "processing_info": {
                    "detailed_landmarks": detailed,
                    "pose_estimation": return_pose,
                    "image_size": image.shape[:2]
                }
            })
        
        else:
            raise HTTPException(status_code=500, detail="Landmark detection service unavailable")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Landmark detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/register-enhanced")
async def register_face_enhanced(
    file: UploadFile = File(...),
    person_name: str = Form(...),
    person_id: str = Form(...),
    strict_validation: bool = Form(True)
):
    """
    ลงทะเบียนใบหน้าพร้อมการตรวจสอบคุณภาพแบบครบวงจร
    
    Args:
        file: ไฟล์รูปภาพใบหน้า
        person_name: ชื่อบุคคล
        person_id: ID ของบุคคล
        strict_validation: ใช้การตรวจสอบแบบเข้มงวด
    
    Returns:
        ผลการลงทะเบียนและข้อมูลการตรวจสอบ
    """
    try:
        # Initialize services
        if not initialize_enhanced_services():
            raise HTTPException(status_code=500, detail="Failed to initialize registration service")
        
        # Process image
        image, image_bytes = await process_uploaded_image(file)
        
        # Enhanced registration process
        if registration_service:
            registration_result = await registration_service.process_registration_with_enhanced_checks(
                image_bytes, person_name, person_id
            )
            
            return JSONResponse(content=registration_result)
        
        else:
            raise HTTPException(status_code=500, detail="Registration service unavailable")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Enhanced registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/service-info")
async def get_enhanced_service_info():
    """
    ข้อมูลเกี่ยวกับ Enhanced Face Services
    """
    try:
        # Initialize services
        if not initialize_enhanced_services():
            return JSONResponse(content={
                "status": "unavailable",
                "message": "Enhanced services not available"
            })
        
        service_info = {
            "status": "online",
            "services": {
                "landmark_detection": {
                    "available": landmark_detector is not None,
                    "features": [
                        "468 detailed landmarks (MediaPipe Face Mesh)",
                        "6 key landmarks (MediaPipe Face Detection)",
                        "Frontal face detection",
                        "Pose estimation (pitch, yaw, roll)"
                    ]
                },
                "quality_assessment": {
                    "available": quality_assessor is not None,
                    "features": [
                        "Sharpness evaluation",
                        "Brightness optimization",
                        "Contrast analysis",
                        "Frontal face validation",
                        "Symmetry assessment"
                    ]
                },
                "enhanced_registration": {
                    "available": registration_service is not None,
                    "features": [
                        "Comprehensive validation pipeline",
                        "Strict pose angle restrictions",
                        "Detailed feedback system",
                        "Quality-based rejection"
                    ]
                }
            },
            "supported_formats": ["JPEG", "PNG", "BMP", "WEBP"],
            "max_image_size": "10MB",
            "api_version": "1.0.0"
        }
        
        # Get detailed service info if available
        if registration_service:
            detailed_info = registration_service.get_service_info()
            service_info["configuration"] = detailed_info
        
        return JSONResponse(content=service_info)
    
    except Exception as e:
        logger.error(f"❌ Service info failed: {e}")
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        })

@router.post("/test-connectivity")
async def test_enhanced_services():
    """
    ทดสอบการเชื่อมต่อ Enhanced Services
    """
    try:
        results = {
            "timestamp": "2025-06-21T15:35:00Z",
            "services": {}
        }
          # Test landmark detector
        try:
            global landmark_detector
            if not landmark_detector:
                landmark_detector = FaceLandmarkDetector()
            results["services"]["landmark_detector"] = {"status": "ok", "message": "Initialized successfully"}
        except Exception as e:
            results["services"]["landmark_detector"] = {"status": "error", "message": str(e)}
        
        # Test quality assessor
        try:
            global quality_assessor
            if not quality_assessor:
                quality_assessor = EnhancedFaceQualityAssessment()
            results["services"]["quality_assessor"] = {"status": "ok", "message": "Initialized successfully"}
        except Exception as e:
            results["services"]["quality_assessor"] = {"status": "error", "message": str(e)}
        
        # Test registration service
        try:
            global registration_service
            if not registration_service:
                registration_service = EnhancedFaceRegistrationService()
            results["services"]["registration_service"] = {"status": "ok", "message": "Initialized successfully"}
        except Exception as e:
            results["services"]["registration_service"] = {"status": "error", "message": str(e)}
        
        # Overall status
        all_ok = all(service["status"] == "ok" for service in results["services"].values())
        results["overall_status"] = "ok" if all_ok else "partial"
        
        return JSONResponse(content=results)
    
    except Exception as e:
        logger.error(f"❌ Connectivity test failed: {e}")
        return JSONResponse(content={
            "overall_status": "error",
            "message": str(e),
            "timestamp": "2025-06-21T15:35:00Z"
        })

# Cleanup on shutdown
import atexit
atexit.register(cleanup_enhanced_services)
