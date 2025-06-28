"""
Example: Integrating Enhanced Quality Assessment with Existing APIs
ตัวอย่างการผสาน Enhanced Quality Assessment เข้ากับ API ที่มีอยู่
"""

from typing import Dict, Any, Optional
import logging
import numpy as np
import cv2

# Import existing services
from ..face_detection.enhanced_quality_assessment import EnhancedFaceQualityAssessment
from ..face_detection.landmark_detector import FaceLandmarkDetector

logger = logging.getLogger(__name__)


class EnhancedFaceRegistrationService:
    """Enhanced face registration with comprehensive quality checks"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize enhanced quality assessment
        self.quality_assessor = EnhancedFaceQualityAssessment(self.config.get("quality", {}))
        
        # Initialize landmark detector
        self.landmark_detector = FaceLandmarkDetector()
        
        # Enhanced thresholds
        self.enhanced_thresholds = {
            "minimum_quality_score": self.config.get("minimum_quality_score", 0.75),
            "strict_frontal_check": self.config.get("strict_frontal_check", True),
            "require_landmarks": self.config.get("require_landmarks", True),
            "max_pose_angle": self.config.get("max_pose_angle", 15.0),  # degrees
        }
        
        logger.info("Enhanced Face Registration Service initialized")
    
    async def validate_registration_image(
        self, 
        image_bytes: bytes, 
        person_id: str,
        strict_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive validation for face registration images
        
        Args:
            image_bytes: Input image bytes
            person_id: Person identifier
            strict_mode: Whether to apply strict quality checks
            
        Returns:
            Validation result with detailed feedback
        """
        try:
            # Decode image
            img_buffer = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
            
            if image is None:
                return {
                    "success": False,
                    "error": "Failed to decode image",
                    "can_retry": True
                }
            
            validation_result = {
                "success": False,
                "person_id": person_id,
                "quality_assessment": {},
                "landmark_analysis": {},
                "pose_analysis": {},
                "recommendations": [],
                "can_retry": True,
                "strict_mode": strict_mode
            }
            
            # Step 1: Basic quality assessment
            quality_metrics = self.quality_assessor.assess_face_quality(
                image, 
                use_landmarks=self.enhanced_thresholds["require_landmarks"]
            )
            
            quality_summary = self.quality_assessor.get_quality_summary(quality_metrics)
            validation_result["quality_assessment"] = quality_summary
            
            # Step 2: Landmark analysis if required
            if self.enhanced_thresholds["require_landmarks"]:
                landmark_results = self.landmark_detector.detect_landmarks(image, detailed=True)
                
                if landmark_results:
                    best_face = landmark_results[0]
                    validation_result["landmark_analysis"] = {
                        "landmarks_found": True,
                        "landmark_count": best_face.get("landmark_count", 0),
                        "quality_metrics": best_face.get("quality_metrics", {}),
                        "is_frontal": self.landmark_detector.is_frontal_face(
                            best_face, 
                            threshold=0.8 if strict_mode else 0.6
                        )
                    }
                    
                    # Step 3: Pose analysis
                    pose_data = self.landmark_detector.get_face_pose(best_face)
                    validation_result["pose_analysis"] = pose_data
                    
                    # Check pose angles
                    max_angle = self.enhanced_thresholds["max_pose_angle"]
                    pose_acceptable = (
                        abs(pose_data.get("yaw", 0)) <= max_angle and
                        abs(pose_data.get("pitch", 0)) <= max_angle and
                        abs(pose_data.get("roll", 0)) <= max_angle * 2  # More lenient for roll
                    )
                    validation_result["pose_analysis"]["pose_acceptable"] = pose_acceptable
                
                else:
                    validation_result["landmark_analysis"] = {
                        "landmarks_found": False,
                        "error": "No facial landmarks detected"
                    }
            
            # Step 4: Final validation decision
            validation_passed = self._make_validation_decision(
                validation_result, 
                quality_metrics, 
                strict_mode
            )
            
            validation_result["success"] = validation_passed
            
            # Step 5: Generate recommendations
            validation_result["recommendations"] = self._generate_enhanced_recommendations(
                validation_result, 
                quality_metrics
            )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Enhanced validation failed for {person_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "person_id": person_id,
                "can_retry": True
            }
    
    def _make_validation_decision(
        self, 
        validation_result: Dict[str, Any], 
        quality_metrics, 
        strict_mode: bool
    ) -> bool:
        """Make final validation decision based on all criteria"""
        try:
            # Check overall quality score
            overall_score = quality_metrics.overall_score
            min_score = self.enhanced_thresholds["minimum_quality_score"]
            if strict_mode:
                min_score = max(min_score, 0.8)  # Stricter in strict mode
            
            if overall_score < min_score:
                return False
            
            # Check basic acceptability
            if not quality_metrics.is_acceptable:
                return False
            
            # Check landmark requirements
            if self.enhanced_thresholds["require_landmarks"]:
                landmark_analysis = validation_result.get("landmark_analysis", {})
                
                if not landmark_analysis.get("landmarks_found", False):
                    return False
                
                if self.enhanced_thresholds["strict_frontal_check"]:
                    if not landmark_analysis.get("is_frontal", False):
                        return False
                
                # Check pose if available
                pose_analysis = validation_result.get("pose_analysis", {})
                if not pose_analysis.get("pose_acceptable", True):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation decision failed: {e}")
            return False
    
    def _generate_enhanced_recommendations(
        self, 
        validation_result: Dict[str, Any], 
        quality_metrics
    ) -> list[str]:
        """Generate enhanced recommendations for improvement"""
        recommendations = []
        
        try:
            # Get basic quality recommendations
            basic_recs = quality_metrics.rejection_reasons
            if basic_recs:
                rec_mapping = {
                    "image_too_blurry": "Hold the camera steady and ensure proper focus",
                    "poor_lighting": "Improve lighting - avoid shadows and backlighting",
                    "low_contrast": "Ensure good contrast between face and background",
                    "not_frontal": "Look directly at the camera",
                    "asymmetric_face": "Center your face in the camera view",
                    "face_too_small": "Move closer to the camera",
                    "overall_quality_poor": "Improve overall image quality"
                }
                
                for reason in basic_recs:
                    if reason in rec_mapping:
                        recommendations.append(rec_mapping[reason])
            
            # Add landmark-specific recommendations
            landmark_analysis = validation_result.get("landmark_analysis", {})
            if not landmark_analysis.get("landmarks_found", True):
                recommendations.append("Ensure your face is clearly visible and well-lit")
            
            if not landmark_analysis.get("is_frontal", True):
                recommendations.append("Face the camera directly - avoid turning your head")
            
            # Add pose-specific recommendations
            pose_analysis = validation_result.get("pose_analysis", {})
            if not pose_analysis.get("pose_acceptable", True):
                yaw = pose_analysis.get("yaw", 0)
                pitch = pose_analysis.get("pitch", 0)
                roll = pose_analysis.get("roll", 0)
                
                if abs(yaw) > 15:
                    direction = "right" if yaw > 0 else "left"
                    recommendations.append(f"Turn your head less to the {direction}")
                
                if abs(pitch) > 15:
                    direction = "up" if pitch > 0 else "down"
                    recommendations.append(f"Tilt your head less {direction}")
                
                if abs(roll) > 30:
                    recommendations.append("Keep your head level - avoid tilting")
            
            # Default recommendation if all passed
            if not recommendations:
                recommendations.append("Image quality is excellent!")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return ["Unable to generate specific recommendations"]
    
    async def process_registration_with_enhanced_checks(
        self, 
        image_bytes: bytes, 
        person_name: str, 
        person_id: str
    ) -> Dict[str, Any]:
        """
        Complete registration process with enhanced quality checks
        """
        try:
            # Step 1: Validate image quality
            validation_result = await self.validate_registration_image(
                image_bytes, 
                person_id, 
                strict_mode=True
            )
            
            if not validation_result["success"]:
                return {
                    "success": False,
                    "stage": "validation",
                    "validation_result": validation_result,
                    "message": "Image quality validation failed"
                }
            
            # Step 2: If validation passed, proceed with registration
            # (This would call your existing face registration service)
            
            # For now, return success with validation details
            return {
                "success": True,
                "stage": "completed",
                "person_id": person_id,
                "person_name": person_name,
                "validation_result": validation_result,
                "message": "Registration completed successfully with enhanced quality checks"
            }
            
        except Exception as e:
            logger.error(f"Enhanced registration failed: {e}")
            return {
                "success": False,
                "stage": "processing",
                "error": str(e),
                "message": "Registration processing failed"
            }
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get enhanced service information"""
        return {
            "service_name": "Enhanced Face Registration Service",
            "features": [
                "Comprehensive quality assessment",
                "Facial landmark detection",
                "Pose estimation",
                "Strict frontal face validation",
                "Detailed feedback and recommendations"
            ],
            "quality_thresholds": self.enhanced_thresholds,
            "landmark_detection": "MediaPipe-based",
            "supported_formats": ["JPEG", "PNG", "BMP"],
            "max_image_size": "10MB"
        }
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'quality_assessor'):
                self.quality_assessor.cleanup()
            if hasattr(self, 'landmark_detector'):
                self.landmark_detector.cleanup()
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


# Usage example for API integration
async def example_api_usage():
    """Example of how to integrate with existing APIs"""
    
    # Initialize enhanced service
    enhanced_service = EnhancedFaceRegistrationService({
        "quality": {
            "sharpness_min": 120.0,  # Stricter sharpness
            "frontal_min": 0.8,      # Stricter frontal requirement
            "overall_min": 0.75      # Higher overall quality
        },
        "minimum_quality_score": 0.8,
        "strict_frontal_check": True,
        "require_landmarks": True,
        "max_pose_angle": 12.0  # Stricter pose requirement
    })
    
    # Example image bytes (would come from your API)
    # image_bytes = open("test_face.jpg", "rb").read()
    
    # Process registration with enhanced checks
    # result = await enhanced_service.process_registration_with_enhanced_checks(
    #     image_bytes=image_bytes,
    #     person_name="John Doe",
    #     person_id="user_123"
    # )
    
    # Get service information
    service_info = enhanced_service.get_service_info()
    print("Enhanced Service Info:", service_info)
    
    # Cleanup
    enhanced_service.cleanup()


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_api_usage())
