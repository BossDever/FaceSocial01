"""
Enhanced Face Quality Assessment Service
ระบบประเมินคุณภาพใบหน้าแบบครอบคลุม ใช้ทั้ง YOLO และ MediaPipe
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Import existing services
from .landmark_detector import FaceLandmarkDetector

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Face quality metrics container"""
    sharpness_score: float = 0.0
    brightness_score: float = 0.0
    contrast_score: float = 0.0
    frontal_score: float = 0.0
    symmetry_score: float = 0.0
    size_score: float = 0.0
    overall_score: float = 0.0
    is_acceptable: bool = False
    rejection_reasons: List[str] = None

    def __post_init__(self):
        if self.rejection_reasons is None:
            self.rejection_reasons = []


class EnhancedFaceQualityAssessment:
    """Enhanced face quality assessment using multiple detection methods"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Quality thresholds
        self.thresholds = {
            "sharpness_min": self.config.get("sharpness_min", 100.0),
            "brightness_min": self.config.get("brightness_min", 50.0),
            "brightness_max": self.config.get("brightness_max", 200.0),
            "contrast_min": self.config.get("contrast_min", 30.0),
            "frontal_min": self.config.get("frontal_min", 0.7),
            "symmetry_min": self.config.get("symmetry_min", 0.6),
            "size_min": self.config.get("size_min", (80, 80)),
            "overall_min": self.config.get("overall_min", 0.7)
        }
        
        # Weights for overall score calculation
        self.weights = {
            "sharpness": self.config.get("weight_sharpness", 0.25),
            "brightness": self.config.get("weight_brightness", 0.15),
            "contrast": self.config.get("weight_contrast", 0.15),
            "frontal": self.config.get("weight_frontal", 0.25),
            "symmetry": self.config.get("weight_symmetry", 0.10),
            "size": self.config.get("weight_size", 0.10)
        }
        
        # Initialize landmark detector
        self.landmark_detector = FaceLandmarkDetector()
        
        logger.info("Enhanced Face Quality Assessment initialized")

    def assess_face_quality(
        self, 
        image: np.ndarray, 
        face_bbox: Optional[Dict[str, int]] = None,
        use_landmarks: bool = True
    ) -> QualityMetrics:
        """
        Comprehensive face quality assessment
        
        Args:
            image: Input face image (BGR format)
            face_bbox: Optional bounding box {x, y, width, height}
            use_landmarks: Whether to use landmark-based analysis
            
        Returns:
            QualityMetrics object with detailed assessment
        """
        try:
            # Extract face region if bbox provided
            if face_bbox:
                x, y, w, h = face_bbox['x'], face_bbox['y'], face_bbox['width'], face_bbox['height']
                face_region = image[y:y+h, x:x+w]
            else:
                face_region = image
            
            metrics = QualityMetrics()
            
            # Basic quality assessments
            metrics.sharpness_score = self._calculate_sharpness(face_region)
            metrics.brightness_score = self._calculate_brightness_score(face_region)
            metrics.contrast_score = self._calculate_contrast_score(face_region)
            metrics.size_score = self._calculate_size_score(face_region)
            
            # Landmark-based assessments
            if use_landmarks:
                landmark_metrics = self._assess_with_landmarks(image, face_bbox)
                metrics.frontal_score = landmark_metrics.get("frontal_score", 0.0)
                metrics.symmetry_score = landmark_metrics.get("symmetry_score", 0.0)
            else:
                # Fallback basic assessments
                metrics.frontal_score = self._estimate_frontal_basic(face_region)
                metrics.symmetry_score = self._estimate_symmetry_basic(face_region)
            
            # Calculate overall score
            metrics.overall_score = self._calculate_overall_score(metrics)
            
            # Determine acceptability and reasons
            metrics.is_acceptable, metrics.rejection_reasons = self._evaluate_acceptability(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return QualityMetrics(rejection_reasons=["assessment_failed"])

    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            
            # Normalize to 0-1 scale
            normalized_score = min(variance / 500.0, 1.0)  # 500 is empirical max
            return float(normalized_score)
            
        except Exception as e:
            logger.error(f"Sharpness calculation failed: {e}")
            return 0.0

    def _calculate_brightness_score(self, image: np.ndarray) -> float:
        """Calculate brightness quality score"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            mean_brightness = np.mean(gray)
            
            # Optimal brightness range: 80-170
            optimal_min, optimal_max = 80, 170
            
            if optimal_min <= mean_brightness <= optimal_max:
                score = 1.0
            elif mean_brightness < optimal_min:
                score = mean_brightness / optimal_min
            else:  # too bright
                score = (255 - mean_brightness) / (255 - optimal_max)
            
            return float(max(0.0, min(1.0, score)))
            
        except Exception as e:
            logger.error(f"Brightness calculation failed: {e}")
            return 0.0

    def _calculate_contrast_score(self, image: np.ndarray) -> float:
        """Calculate contrast quality score"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            contrast = np.std(gray)
            
            # Normalize contrast score
            normalized_score = min(contrast / 80.0, 1.0)  # 80 is empirical good contrast
            return float(normalized_score)
            
        except Exception as e:
            logger.error(f"Contrast calculation failed: {e}")
            return 0.0

    def _calculate_size_score(self, image: np.ndarray) -> float:
        """Calculate size adequacy score"""
        try:
            height, width = image.shape[:2]
            min_width, min_height = self.thresholds["size_min"]
            
            # Score based on size adequacy
            width_score = min(width / min_width, 1.0) if min_width > 0 else 1.0
            height_score = min(height / min_height, 1.0) if min_height > 0 else 1.0
            
            size_score = (width_score + height_score) / 2
            return float(size_score)
            
        except Exception as e:
            logger.error(f"Size calculation failed: {e}")
            return 0.0

    def _assess_with_landmarks(self, image: np.ndarray, face_bbox: Optional[Dict[str, int]] = None) -> Dict[str, float]:
        """Assess quality using landmark detection"""
        try:
            # Detect landmarks
            landmark_results = self.landmark_detector.detect_landmarks(image, detailed=True)
            
            if not landmark_results:
                return {"frontal_score": 0.0, "symmetry_score": 0.0}
            
            # Use the first (best) detection
            best_result = landmark_results[0]
            quality_metrics = best_result.get("quality_metrics", {})
            
            frontal_score = quality_metrics.get("frontal_score", 0.0)
            symmetry_score = quality_metrics.get("nose_symmetry", 0.0)
            
            return {
                "frontal_score": frontal_score,
                "symmetry_score": symmetry_score
            }
            
        except Exception as e:
            logger.error(f"Landmark-based assessment failed: {e}")
            return {"frontal_score": 0.0, "symmetry_score": 0.0}

    def _estimate_frontal_basic(self, image: np.ndarray) -> float:
        """Basic frontal estimation without landmarks"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            height, width = gray.shape
            
            # Simple symmetry check using pixel intensity
            left_half = gray[:, :width//2]
            right_half = gray[:, width//2:]
            right_half_flipped = np.fliplr(right_half)
            
            # Resize to match if needed
            if left_half.shape != right_half_flipped.shape:
                min_width = min(left_half.shape[1], right_half_flipped.shape[1])
                left_half = left_half[:, :min_width]
                right_half_flipped = right_half_flipped[:, :min_width]
            
            # Calculate similarity between halves
            difference = np.abs(left_half.astype(float) - right_half_flipped.astype(float))
            similarity = 1.0 - (np.mean(difference) / 255.0)
            
            return float(max(0.0, similarity))
            
        except Exception as e:
            logger.error(f"Basic frontal estimation failed: {e}")
            return 0.0

    def _estimate_symmetry_basic(self, image: np.ndarray) -> float:
        """Basic symmetry estimation"""
        try:
            # Use the same method as frontal for simplicity
            return self._estimate_frontal_basic(image)
            
        except Exception as e:
            logger.error(f"Basic symmetry estimation failed: {e}")
            return 0.0

    def _calculate_overall_score(self, metrics: QualityMetrics) -> float:
        """Calculate weighted overall quality score"""
        try:
            score = (
                metrics.sharpness_score * self.weights["sharpness"] +
                metrics.brightness_score * self.weights["brightness"] +
                metrics.contrast_score * self.weights["contrast"] +
                metrics.frontal_score * self.weights["frontal"] +
                metrics.symmetry_score * self.weights["symmetry"] +
                metrics.size_score * self.weights["size"]
            )
            
            return float(max(0.0, min(1.0, score)))
            
        except Exception as e:
            logger.error(f"Overall score calculation failed: {e}")
            return 0.0

    def _evaluate_acceptability(self, metrics: QualityMetrics) -> tuple[bool, List[str]]:
        """Evaluate if face meets quality requirements"""
        reasons = []
        
        try:
            # Check each criterion
            if metrics.sharpness_score < (self.thresholds["sharpness_min"] / 500.0):
                reasons.append("image_too_blurry")
            
            if metrics.brightness_score < 0.5:  # Adjusted threshold
                reasons.append("poor_lighting")
            
            if metrics.contrast_score < (self.thresholds["contrast_min"] / 80.0):
                reasons.append("low_contrast")
            
            if metrics.frontal_score < self.thresholds["frontal_min"]:
                reasons.append("not_frontal")
            
            if metrics.symmetry_score < self.thresholds["symmetry_min"]:
                reasons.append("asymmetric_face")
            
            if metrics.size_score < 0.8:  # Require good size
                reasons.append("face_too_small")
            
            if metrics.overall_score < self.thresholds["overall_min"]:
                reasons.append("overall_quality_poor")
            
            is_acceptable = len(reasons) == 0
            
            return is_acceptable, reasons
            
        except Exception as e:
            logger.error(f"Acceptability evaluation failed: {e}")
            return False, ["evaluation_failed"]

    def get_quality_summary(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """Get human-readable quality summary"""
        try:
            return {
                "overall_score": f"{metrics.overall_score:.2f}",
                "quality_grade": self._get_quality_grade(metrics.overall_score),
                "is_acceptable": metrics.is_acceptable,
                "detailed_scores": {
                    "sharpness": f"{metrics.sharpness_score:.2f}",
                    "brightness": f"{metrics.brightness_score:.2f}",
                    "contrast": f"{metrics.contrast_score:.2f}",
                    "frontal": f"{metrics.frontal_score:.2f}",
                    "symmetry": f"{metrics.symmetry_score:.2f}",
                    "size": f"{metrics.size_score:.2f}"
                },
                "issues": metrics.rejection_reasons,
                "recommendations": self._get_recommendations(metrics)
            }
            
        except Exception as e:
            logger.error(f"Quality summary generation failed: {e}")
            return {"error": str(e)}

    def _get_quality_grade(self, score: float) -> str:
        """Convert numeric score to grade"""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Good"
        elif score >= 0.7:
            return "Acceptable"
        elif score >= 0.5:
            return "Poor"
        else:
            return "Unacceptable"

    def _get_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """Get improvement recommendations"""
        recommendations = []
        
        try:
            if metrics.sharpness_score < 0.5:
                recommendations.append("Hold camera steady and ensure proper focus")
            
            if metrics.brightness_score < 0.5:
                recommendations.append("Improve lighting conditions")
            
            if metrics.contrast_score < 0.5:
                recommendations.append("Avoid flat lighting, add some contrast")
            
            if metrics.frontal_score < 0.7:
                recommendations.append("Look directly at the camera")
            
            if metrics.size_score < 0.8:
                recommendations.append("Move closer to camera or crop face larger")
            
            if not recommendations:
                recommendations.append("Face quality is good!")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendations generation failed: {e}")
            return ["Unable to generate recommendations"]

    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'landmark_detector'):
                self.landmark_detector.cleanup()
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


# Usage example
if __name__ == "__main__":
    quality_assessor = EnhancedFaceQualityAssessment()
    
    # Test with an image
    import cv2
    image = cv2.imread("test_face.jpg")
    
    if image is not None:
        metrics = quality_assessor.assess_face_quality(image)
        summary = quality_assessor.get_quality_summary(metrics)
        
        print("Face Quality Assessment:")
        print(f"Overall Score: {summary['overall_score']}")
        print(f"Grade: {summary['quality_grade']}")
        print(f"Acceptable: {summary['is_acceptable']}")
        
        if summary['issues']:
            print(f"Issues: {', '.join(summary['issues'])}")
        
        print("Recommendations:")
        for rec in summary['recommendations']:
            print(f"- {rec}")
    
    quality_assessor.cleanup()
