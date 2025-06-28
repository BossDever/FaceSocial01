"""
Simple YOLOv11m Face Detection Service for Real-time CCTV
Single model, optimized for speed and simplicity
"""

import time
import logging
import cv2
import numpy as np
import os
import asyncio
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SimpleFaceDetection:
    """Simple face detection result"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    quality_score: Optional[float] = None


@dataclass 
class SimpleDetectionResult:
    """Simple detection result"""
    faces: List[SimpleFaceDetection]
    processing_time_ms: float
    image_shape: Tuple[int, int, int]
    model_used: str
    error: Optional[str] = None


class SimpleYOLOv11mService:
    """
    Simple YOLOv11m Face Detection Service
    Single model, optimized for real-time CCTV
    """
    
    def __init__(self, model_path: str = "model/face-detection/yolov11m-face.pt"):
        self.model_path = model_path
        self.model = None
        self.device = "cpu"
        self.model_loaded = False
        
        # Performance tracking
        self.total_detections = 0
        self.total_processing_time = 0.0
        self.last_processing_time = 0.0
        
        # Configuration
        self.input_size = 640
        self.default_conf = 0.3
        self.default_iou = 0.4
        
    def _check_dependencies(self) -> bool:
        """Check required dependencies"""
        try:
            from ultralytics import YOLO
            import torch
            return True
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return False
    
    def _determine_device(self) -> str:
        """Determine best device to use"""
        try:
            import torch
            
            if torch.cuda.is_available():
                try:
                    # Test CUDA functionality
                    test_tensor = torch.tensor([1.0]).cuda()
                    _ = test_tensor.cpu()
                    
                    device_name = torch.cuda.get_device_name()
                    logger.info(f"🎯 Using GPU: {device_name}")
                    return "cuda"
                    
                except Exception as cuda_error:
                    logger.warning(f"⚠️ CUDA test failed: {cuda_error}")
                    return "cpu"
            else:
                logger.info("ℹ️ Using CPU (CUDA not available)")
                return "cpu"
                
        except Exception as e:
            logger.warning(f"⚠️ Device check failed: {e}")
            return "cpu"
    
    async def initialize(self) -> bool:
        """Initialize the YOLOv11m model"""
        try:
            # Check dependencies
            if not self._check_dependencies():
                logger.error("❌ Required dependencies not available")
                return False
            
            # Check model file
            if not os.path.exists(self.model_path):
                logger.error(f"❌ Model file not found: {self.model_path}")
                return False
            
            # Determine device
            self.device = self._determine_device()
            
            logger.info(f"⏳ Loading YOLOv11m model: {self.model_path}")
            
            # Load model
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.model_loaded = True
            
            logger.info(f"✅ YOLOv11m loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize YOLOv11m: {e}")
            self.model_loaded = False
            return False
    
    def _calculate_quality_score(self, bbox: Tuple[int, int, int, int], confidence: float) -> float:
        """Calculate simple quality score"""
        x1, y1, x2, y2 = bbox
        
        # Face size score (larger faces = better quality)
        face_area = (x2 - x1) * (y2 - y1)
        size_score = min(face_area / 10000.0, 1.0) * 50  # Max 50 points
        
        # Confidence score
        conf_score = confidence * 50  # Max 50 points
        
        return min(size_score + conf_score, 100.0)
    
    def _process_yolo_results(self, results: Any, image_shape: Tuple[int, int, int]) -> List[SimpleFaceDetection]:
        """Process YOLO detection results"""
        detections = []
        
        if not results or not hasattr(results[0], 'boxes') or results[0].boxes is None:
            return detections
        
        h, w = image_shape[:2]
        
        for box_data in results[0].boxes:
            if box_data.xyxyn is not None and len(box_data.xyxyn) > 0:
                # Get normalized coordinates
                x1_norm, y1_norm, x2_norm, y2_norm = box_data.xyxyn[0].tolist()
                
                # Convert to absolute coordinates
                x1 = int(x1_norm * w)
                y1 = int(y1_norm * h)
                x2 = int(x2_norm * w)
                y2 = int(y2_norm * h)
                
                # Get confidence
                confidence = float(box_data.conf[0]) if box_data.conf is not None else 0.0
                
                # Validate bounding box
                if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
                    bbox = (x1, y1, x2, y2)
                    quality_score = self._calculate_quality_score(bbox, confidence)
                    
                    detection = SimpleFaceDetection(
                        bbox=bbox,
                        confidence=confidence,
                        quality_score=quality_score
                    )
                    detections.append(detection)
        
        return detections
    
    async def detect_faces(
        self,
        image: Union[str, np.ndarray],
        conf_threshold: float = None,
        iou_threshold: float = None,
        max_faces: Optional[int] = None,
        min_quality: float = 30.0
    ) -> SimpleDetectionResult:
        """
        Detect faces in image with YOLOv11m
        
        Args:
            image: Image file path or numpy array
            conf_threshold: Confidence threshold (default: 0.3)
            iou_threshold: IoU threshold for NMS (default: 0.4)
            max_faces: Maximum number of faces to return
            min_quality: Minimum quality score threshold
            
        Returns:
            SimpleDetectionResult with detected faces
        """
        start_time = time.time()
        
        # Validation
        if not self.model_loaded:
            return SimpleDetectionResult(
                faces=[],
                processing_time_ms=0.0,
                image_shape=(0, 0, 0),
                model_used="YOLOv11m",
                error="Model not loaded"
            )
        
        # Process image input
        try:
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image file not found: {image}")
                img_array = cv2.imread(image)
                if img_array is None:
                    raise ValueError(f"Cannot read image file: {image}")
            elif isinstance(image, np.ndarray):
                img_array = image.copy()
            else:
                raise TypeError("Image must be file path or numpy array")
            
            if img_array.size == 0:
                raise ValueError("Empty image provided")
                
        except Exception as e:
            return SimpleDetectionResult(
                faces=[],
                processing_time_ms=time.time() - start_time,
                image_shape=(0, 0, 0),
                model_used="YOLOv11m",
                error=f"Image processing error: {e}"
            )
        
        # Set thresholds
        conf_thresh = conf_threshold if conf_threshold is not None else self.default_conf
        iou_thresh = iou_threshold if iou_threshold is not None else self.default_iou
        
        try:
            # Run YOLO detection
            results = self.model.predict(
                source=img_array,
                conf=conf_thresh,
                iou=iou_thresh,
                device=self.device,
                verbose=False,
                save=False,
                show=False,
                imgsz=self.input_size
            )
            
            # Process results
            detections = self._process_yolo_results(results, img_array.shape)
            
            # Filter by quality
            if min_quality > 0:
                detections = [d for d in detections if d.quality_score >= min_quality]
            
            # Sort by quality (best first)
            detections.sort(key=lambda x: x.quality_score or 0, reverse=True)
            
            # Limit number of faces
            if max_faces is not None and len(detections) > max_faces:
                detections = detections[:max_faces]
            
            # Update performance stats
            processing_time = (time.time() - start_time) * 1000  # ms
            self.last_processing_time = processing_time
            self.total_processing_time += processing_time
            self.total_detections += 1
            
            logger.debug(f"YOLOv11m detected {len(detections)} faces in {processing_time:.1f}ms")
            
            return SimpleDetectionResult(
                faces=detections,
                processing_time_ms=processing_time,
                image_shape=img_array.shape,
                model_used="YOLOv11m",
                error=None
            )
            
        except Exception as e:
            error_msg = f"Detection failed: {e}"
            logger.error(error_msg)
            
            return SimpleDetectionResult(
                faces=[],
                processing_time_ms=(time.time() - start_time) * 1000,
                image_shape=img_array.shape,
                model_used="YOLOv11m",
                error=error_msg
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_time = self.total_processing_time / max(self.total_detections, 1)
        fps = 1000.0 / max(avg_time, 1.0)  # Convert ms to FPS
        
        return {
            "model": "YOLOv11m",
            "device": self.device,
            "total_detections": self.total_detections,
            "total_processing_time_ms": self.total_processing_time,
            "average_processing_time_ms": avg_time,
            "last_processing_time_ms": self.last_processing_time,
            "fps": fps,
            "model_loaded": self.model_loaded
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.total_detections = 0
        self.total_processing_time = 0.0
        self.last_processing_time = 0.0


# Example usage for real-time CCTV
class RealTimeCCTVDetector:
    """Real-time CCTV face detector using simple YOLOv11m"""
    
    def __init__(self, model_path: str = "model/face-detection/yolov11m-face.pt"):
        self.detector = SimpleYOLOv11mService(model_path)
        self.is_running = False
        
    async def initialize(self) -> bool:
        """Initialize the detector"""
        return await self.detector.initialize()
    
    async def process_cctv_stream(
        self,
        camera_source: Union[str, int] = 0,
        conf_threshold: float = 0.3,
        target_fps: float = 10.0
    ):
        """
        Process CCTV stream with face detection
        
        Args:
            camera_source: Camera source (0 for webcam, or IP camera URL)
            conf_threshold: Detection confidence threshold
            target_fps: Target processing FPS
        """
        frame_interval = 1.0 / target_fps
        
        cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera: {camera_source}")
            return
        
        logger.info(f"🎥 Starting CCTV processing (target: {target_fps} FPS)")
        self.is_running = True
        
        try:
            last_process_time = 0
            
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break
                
                current_time = time.time()
                
                # Skip frame if processing too fast (maintain target FPS)
                if current_time - last_process_time < frame_interval:
                    continue
                
                # Detect faces
                result = await self.detector.detect_faces(
                    frame,
                    conf_threshold=conf_threshold,
                    max_faces=20  # Limit for performance
                )
                
                # Draw results on frame
                if result.faces:
                    for face in result.faces:
                        x1, y1, x2, y2 = face.bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add confidence and quality text
                        text = f"Conf: {face.confidence:.2f} Q: {face.quality_score:.0f}"
                        cv2.putText(frame, text, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Add performance info
                stats = self.detector.get_performance_stats()
                info_text = f"FPS: {stats['fps']:.1f} | Faces: {len(result.faces)} | Time: {result.processing_time_ms:.1f}ms"
                cv2.putText(frame, info_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow('CCTV Face Detection', frame)
                
                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                last_process_time = current_time
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.is_running = False
            logger.info("🛑 CCTV processing stopped")
    
    def stop(self):
        """Stop CCTV processing"""
        self.is_running = False


# Test function
async def test_simple_detector():
    """Test the simple YOLOv11m detector"""
    print("🧪 Testing Simple YOLOv11m Face Detector")
    
    detector = SimpleYOLOv11mService()
    
    # Initialize
    if not await detector.initialize():
        print("❌ Failed to initialize detector")
        return False
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Run detection
    result = await detector.detect_faces(test_image, conf_threshold=0.1)
    
    print(f"📊 Detection result:")
    print(f"  - Faces found: {len(result.faces)}")
    print(f"  - Processing time: {result.processing_time_ms:.1f}ms")
    print(f"  - Model used: {result.model_used}")
    print(f"  - Error: {result.error}")
    
    # Show performance stats
    stats = detector.get_performance_stats()
    print(f"📈 Performance stats: {stats}")
    
    return True

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_simple_detector())
