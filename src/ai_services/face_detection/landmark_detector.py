"""
Enhanced Landmark Detection Service using MediaPipe
เพิ่มการตรวจจับจุดสำคัญบนใบหน้าด้วย MediaPipe
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class FaceLandmarkDetector:
    """Face Landmark Detection using MediaPipe"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize face mesh for detailed landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize face detection for basic landmarks
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for close-range, 1 for full-range
            min_detection_confidence=0.5
        )
    
    def detect_landmarks(self, image: np.ndarray, detailed: bool = False) -> List[Dict[str, Any]]:
        """
        Detect facial landmarks
        
        Args:
            image: Input image (BGR format)
            detailed: If True, returns 468 landmarks; if False, returns 6 key landmarks
            
        Returns:
            List of face landmarks with quality metrics
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            
            faces_data = []
            
            if detailed:
                # Use face mesh for detailed landmarks
                results = self.face_mesh.process(rgb_image)
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        landmarks = []
                        for landmark in face_landmarks.landmark:
                            x = int(landmark.x * width)
                            y = int(landmark.y * height)
                            z = landmark.z  # Relative depth
                            landmarks.append([x, y, z])
                        
                        # Calculate face quality metrics
                        quality_metrics = self._calculate_face_quality(landmarks, width, height)
                        
                        faces_data.append({
                            "landmarks": landmarks,
                            "landmark_count": len(landmarks),
                            "type": "detailed_mesh",
                            "quality_metrics": quality_metrics
                        })
            else:
                # Use face detection for basic landmarks
                results = self.face_detection.process(rgb_image)
                
                if results.detections:
                    for detection in results.detections:
                        # Get bounding box
                        bbox = detection.location_data.relative_bounding_box
                        
                        # Get key landmarks (6 points)
                        key_landmarks = []
                        if detection.location_data.relative_keypoints:
                            for keypoint in detection.location_data.relative_keypoints:
                                x = int(keypoint.x * width)
                                y = int(keypoint.y * height)
                                key_landmarks.append([x, y])
                        
                        # Calculate basic quality metrics
                        quality_metrics = self._calculate_basic_quality(key_landmarks, bbox, width, height)
                        
                        faces_data.append({
                            "landmarks": key_landmarks,
                            "landmark_count": len(key_landmarks),
                            "type": "key_points",
                            "bbox": {
                                "x": int(bbox.xmin * width),
                                "y": int(bbox.ymin * height),
                                "width": int(bbox.width * width),
                                "height": int(bbox.height * height)
                            },
                            "confidence": detection.score[0],
                            "quality_metrics": quality_metrics
                        })
            
            return faces_data
            
        except Exception as e:
            logger.error(f"Landmark detection failed: {e}")
            return []
    
    def _calculate_face_quality(self, landmarks: List[List[float]], width: int, height: int) -> Dict[str, float]:
        """Calculate face quality metrics from detailed landmarks"""
        try:
            if len(landmarks) < 468:
                return {"frontal_score": 0.0, "quality_score": 0.0}
            
            # Key landmark indices for face analysis
            LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            NOSE_TIP = 1
            MOUTH = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
            
            # Convert to numpy arrays
            landmarks_np = np.array(landmarks)
            
            # Calculate frontal score based on facial symmetry
            left_eye_center = np.mean([landmarks_np[i][:2] for i in LEFT_EYE], axis=0)
            right_eye_center = np.mean([landmarks_np[i][:2] for i in RIGHT_EYE], axis=0)
            nose_tip = landmarks_np[NOSE_TIP][:2]
            
            # Eye distance and symmetry
            eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
            eye_center = (left_eye_center + right_eye_center) / 2
            
            # Nose should be centered between eyes
            nose_offset = abs(nose_tip[0] - eye_center[0])
            nose_symmetry = 1.0 - min(nose_offset / (eye_distance / 2), 1.0)
            
            # Face pose estimation (simplified)
            face_width = max(landmarks_np[:, 0]) - min(landmarks_np[:, 0])
            face_height = max(landmarks_np[:, 1]) - min(landmarks_np[:, 1])
            aspect_ratio = face_width / face_height if face_height > 0 else 0
            
            # Frontal faces typically have aspect ratio around 0.75-0.85
            aspect_score = 1.0 - abs(aspect_ratio - 0.8) / 0.3 if aspect_ratio > 0 else 0
            aspect_score = max(0, min(1, aspect_score))
            
            # Overall frontal score
            frontal_score = (nose_symmetry * 0.6 + aspect_score * 0.4)
            
            # Quality score based on landmark distribution
            landmark_spread = np.std(landmarks_np[:, :2], axis=0)
            spread_score = min(np.mean(landmark_spread) / 50.0, 1.0)  # Normalize spread
            
            quality_score = (frontal_score * 0.7 + spread_score * 0.3)
            
            return {
                "frontal_score": float(frontal_score),
                "quality_score": float(quality_score),
                "nose_symmetry": float(nose_symmetry),
                "aspect_ratio": float(aspect_ratio),
                "aspect_score": float(aspect_score),
                "eye_distance": float(eye_distance),
                "face_width": float(face_width),
                "face_height": float(face_height)
            }
            
        except Exception as e:
            logger.error(f"Quality calculation failed: {e}")
            return {"frontal_score": 0.0, "quality_score": 0.0}
    
    def _calculate_basic_quality(self, landmarks: List[List[int]], bbox, width: int, height: int) -> Dict[str, float]:
        """Calculate basic quality metrics from key landmarks"""
        try:
            if len(landmarks) < 6:
                return {"frontal_score": 0.0, "quality_score": 0.0}
            
            # Basic symmetry check using eye positions
            if len(landmarks) >= 2:
                left_eye = np.array(landmarks[0])
                right_eye = np.array(landmarks[1])
                
                # Eye level check (should be roughly horizontal)
                eye_level_diff = abs(left_eye[1] - right_eye[1])
                eye_distance = np.linalg.norm(right_eye - left_eye)
                
                if eye_distance > 0:
                    level_score = 1.0 - min(eye_level_diff / (eye_distance * 0.1), 1.0)
                else:
                    level_score = 0.0
                
                # Face size relative to image
                face_area = bbox.width * bbox.height * width * height
                image_area = width * height
                size_ratio = face_area / image_area if image_area > 0 else 0
                size_score = min(size_ratio * 10, 1.0)  # Optimal face size ~10% of image
                
                frontal_score = level_score
                quality_score = (frontal_score * 0.6 + size_score * 0.4)
                
                return {
                    "frontal_score": float(frontal_score),
                    "quality_score": float(quality_score),
                    "eye_level_score": float(level_score),
                    "size_score": float(size_score),
                    "eye_distance": float(eye_distance),
                    "face_size_ratio": float(size_ratio)
                }
            
            return {"frontal_score": 0.0, "quality_score": 0.0}
            
        except Exception as e:
            logger.error(f"Basic quality calculation failed: {e}")
            return {"frontal_score": 0.0, "quality_score": 0.0}
    
    def is_frontal_face(self, landmarks_data: Dict[str, Any], threshold: float = 0.7) -> bool:
        """
        Check if face is frontal based on landmark analysis
        
        Args:
            landmarks_data: Output from detect_landmarks
            threshold: Minimum frontal score (0.0-1.0)
            
        Returns:
            True if face is considered frontal
        """
        quality_metrics = landmarks_data.get("quality_metrics", {})
        frontal_score = quality_metrics.get("frontal_score", 0.0)
        return frontal_score >= threshold
    
    def get_face_pose(self, landmarks_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimate face pose (pitch, yaw, roll) from landmarks
        Simplified implementation for basic pose estimation
        """
        try:
            if landmarks_data.get("type") != "detailed_mesh":
                return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "confidence": 0.0}
            
            landmarks = landmarks_data.get("landmarks", [])
            if len(landmarks) < 468:
                return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "confidence": 0.0}
            
            # Simplified pose estimation using key landmarks
            landmarks_np = np.array(landmarks)
            
            # Nose tip and nose bridge
            nose_tip = landmarks_np[1]
            nose_bridge = landmarks_np[6]
            
            # Eye centers
            left_eye = np.mean([landmarks_np[33], landmarks_np[133]], axis=0)
            right_eye = np.mean([landmarks_np[362], landmarks_np[263]], axis=0)
            
            # Calculate angles (simplified)
            eye_line = right_eye - left_eye
            roll = np.arctan2(eye_line[1], eye_line[0]) * 180 / np.pi
            
            # Yaw estimation based on nose position relative to eyes
            eye_center = (left_eye + right_eye) / 2
            nose_offset = nose_tip[0] - eye_center[0]
            eye_distance = np.linalg.norm(right_eye - left_eye)
            yaw = (nose_offset / eye_distance) * 30 if eye_distance > 0 else 0  # Rough estimate
            
            # Pitch estimation based on nose bridge angle
            nose_vector = nose_tip - nose_bridge
            pitch = np.arctan2(nose_vector[1], nose_vector[2] if len(nose_vector) > 2 else 1) * 180 / np.pi
            
            confidence = min(landmarks_data.get("quality_metrics", {}).get("quality_score", 0.0), 1.0)
            
            return {
                "pitch": float(pitch),
                "yaw": float(yaw),
                "roll": float(roll),
                "confidence": float(confidence)
            }
            
        except Exception as e:
            logger.error(f"Pose estimation failed: {e}")
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "confidence": 0.0}
    
    def cleanup(self):
        """Clean up MediaPipe resources"""
        try:
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
            if hasattr(self, 'face_detection'):
                self.face_detection.close()
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

# Usage example
if __name__ == "__main__":
    detector = FaceLandmarkDetector()
    
    # Load test image
    import cv2
    image = cv2.imread("test_face.jpg")
    
    if image is not None:
        # Detect detailed landmarks
        detailed_results = detector.detect_landmarks(image, detailed=True)
        
        for face_data in detailed_results:
            print(f"Found {face_data['landmark_count']} landmarks")
            print(f"Frontal score: {face_data['quality_metrics']['frontal_score']:.2f}")
            print(f"Is frontal: {detector.is_frontal_face(face_data)}")
            
            # Get pose estimation
            pose = detector.get_face_pose(face_data)
            print(f"Pose - Yaw: {pose['yaw']:.1f}°, Pitch: {pose['pitch']:.1f}°, Roll: {pose['roll']:.1f}°")
    
    detector.cleanup()
