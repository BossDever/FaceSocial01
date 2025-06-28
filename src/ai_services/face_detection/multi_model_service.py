"""
Multi-Model YOLO Face Detection Service for Real-time CCTV
Supports both YOLOv11m (.pt) and YOLOv11n (.onnx) models for performance comparison
"""

import time
import logging
import cv2
import numpy as np
import os
import asyncio
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Check for ONNX runtime availability
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    logger.info("✅ ONNX Runtime available")
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("⚠️ ONNX Runtime not available - yolov11n model will not work")


class ModelType(Enum):
    """Available model types"""
    YOLO11M = "yolov11m"
    YOLO11N = "yolov11n"


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


class MultiModelService:
    """
    Multi-Model YOLO Face Detection Service
    Supports YOLOv11m (.pt) and YOLOv11n (.onnx) for performance comparison
    """
    
    def __init__(self):
        # Model paths (relative to project root)
        self.model_paths = {
            ModelType.YOLO11M: "model/face-detection/yolov11m-face.pt",
            ModelType.YOLO11N: "model/face-detection/yolov11n-face.onnx"
        }
        
        # Model instances (stores both YOLO and ONNX models)
        self.models = {}
        self.onnx_sessions = {}  # For ONNX runtime sessions
        self.current_model_type = ModelType.YOLO11M
        self.device = None  # Will be determined later
        self.models_loaded = {}
        
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
            
            logger.info(f"🔍 Checking CUDA availability...")
            logger.info(f"🔍 torch.cuda.is_available(): {torch.cuda.is_available()}")
            
            if torch.cuda.is_available():
                try:
                    device_count = torch.cuda.device_count()
                    logger.info(f"🔍 CUDA device count: {device_count}")
                    
                    # Test CUDA functionality
                    test_tensor = torch.tensor([1.0]).cuda()
                    _ = test_tensor.cpu()
                    
                    device_name = torch.cuda.get_device_name()
                    logger.info(f"🎯 Multi-Model Service using GPU: {device_name}")
                    return "cuda"
                    
                except Exception as cuda_error:
                    logger.warning(f"⚠️ CUDA test failed in multi-model: {cuda_error}")
                    return "cpu"
            else:
                logger.info("ℹ️ Multi-Model Service using CPU (CUDA not available)")
                return "cpu"
                
        except Exception as e:
            logger.warning(f"⚠️ Device check failed in multi-model: {e}")
            return "cpu"

    def _preprocess_image_for_onnx(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ONNX model"""
        # Resize image to model input size
        resized = cv2.resize(image, (self.input_size, self.input_size))
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Convert to CHW format (Channel, Height, Width)
        chw_image = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batched = np.expand_dims(chw_image, axis=0)
        
        return batched

    def _postprocess_onnx_results(self, outputs: np.ndarray, image_shape: Tuple[int, int, int], conf_threshold: float) -> List[SimpleFaceDetection]:
        """Postprocess ONNX detection results"""
        detections = []
        
        if outputs is None or len(outputs) == 0:
            return detections
        
        # outputs shape: [1, 5, 8400] - [batch, 5 values (x, y, w, h, conf), anchors]
        output = outputs[0]  # Remove batch dimension: [5, 8400]
        
        h, w = image_shape[:2]
        
        # Extract predictions
        # output format: [x_center, y_center, width, height, confidence]
        predictions = output.T  # Transpose to [8400, 5]
        
        for pred in predictions:
            confidence = pred[4]
            
            # Filter by confidence
            if confidence < conf_threshold:
                continue
            
            # Extract box coordinates (normalized)
            x_center, y_center, width, height = pred[:4]
            
            # Convert from center format to corner format
            x1_norm = x_center - width / 2
            y1_norm = y_center - height / 2
            x2_norm = x_center + width / 2
            y2_norm = y_center + height / 2
            
            # Convert to pixel coordinates
            x1 = max(0, int(x1_norm * w))
            y1 = max(0, int(y1_norm * h))
            x2 = min(w, int(x2_norm * w))
            y2 = min(h, int(y2_norm * h))
            
            # Calculate quality score
            quality_score = self._calculate_quality_score((x1, y1, x2, y2), confidence)
            
            detection = SimpleFaceDetection(
                bbox=(x1, y1, x2, y2),
                confidence=float(confidence),
                quality_score=quality_score
            )
            
            detections.append(detection)
        
        return detections

    async def load_model(self, model_type: ModelType) -> bool:
        """Load a specific model (either ultralytics YOLO or ONNX runtime)"""
        try:
            if model_type in self.models_loaded and self.models_loaded[model_type]:
                logger.info(f"✅ Model {model_type.value} already loaded")
                return True
                
            model_path = self.model_paths[model_type]
            
            # Check model file
            if not os.path.exists(model_path):
                logger.error(f"❌ Model file not found: {model_path}")
                return False
            
            # Determine device if not set
            if not hasattr(self, 'device') or self.device is None:
                self.device = self._determine_device()
            
            logger.info(f"⏳ Loading {model_type.value} model: {model_path}")
            
            if model_type == ModelType.YOLO11M:
                # Load YOLOv11m with ultralytics
                if not self._check_dependencies():
                    logger.error("❌ Ultralytics dependencies not available")
                    return False
                
                from ultralytics import YOLO
                model = YOLO(model_path)
                
                # Move model to the determined device
                if hasattr(model, 'to'):
                    model.to(self.device)
                
                self.models[model_type] = model
                
            elif model_type == ModelType.YOLO11N:
                # Load YOLOv11n with ONNX runtime
                if not ONNX_AVAILABLE:
                    logger.error("❌ ONNX Runtime not available for yolov11n")
                    return False
                
                import onnxruntime as ort
                
                # Set up ONNX runtime providers
                providers = []
                if self.device == "cuda":
                    providers.append('CUDAExecutionProvider')
                providers.append('CPUExecutionProvider')
                
                # Create ONNX session
                session = ort.InferenceSession(
                    model_path,
                    providers=providers
                )
                
                self.onnx_sessions[model_type] = session
                
                # Log provider info
                logger.info(f"🔧 ONNX providers: {session.get_providers()}")
            
            self.models_loaded[model_type] = True
            logger.info(f"✅ {model_type.value} loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_type.value}: {e}")
            self.models_loaded[model_type] = False
            return False

    async def initialize(self, default_model: ModelType = ModelType.YOLO11M) -> bool:
        """Initialize with default model"""
        try:
            success = await self.load_model(default_model)
            if success:
                self.current_model_type = default_model
            return success
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {e}")
            return False

    async def switch_model(self, model_type: ModelType) -> bool:
        """Switch to a different model"""
        try:
            if model_type not in self.models_loaded or not self.models_loaded[model_type]:
                success = await self.load_model(model_type)
                if not success:
                    return False
            
            self.current_model_type = model_type
            logger.info(f"🔄 Switched to {model_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to switch to {model_type.value}: {e}")
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
                
                # Convert to pixel coordinates
                x1 = max(0, int(x1_norm * w))
                y1 = max(0, int(y1_norm * h))
                x2 = min(w, int(x2_norm * w))
                y2 = min(h, int(y2_norm * h))
                
                # Get confidence
                confidence = float(box_data.conf[0])
                
                # Calculate quality score
                quality_score = self._calculate_quality_score((x1, y1, x2, y2), confidence)
                
                detection = SimpleFaceDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    quality_score=quality_score
                )
                
                detections.append(detection)
        
        return detections

    async def detect_faces(
        self,
        image: np.ndarray,
        model_type: Optional[ModelType] = None,
        conf_threshold: float = None,
        max_faces: int = 20,
        min_quality: float = 0.0
    ) -> SimpleDetectionResult:
        """
        Detect faces in image using specified model
        """
        start_time = time.time()
        
        try:
            # Use specified model or current model
            target_model = model_type or self.current_model_type
            
            # Switch model if needed
            if target_model != self.current_model_type:
                switch_success = await self.switch_model(target_model)
                if not switch_success:
                    return SimpleDetectionResult(
                        faces=[],
                        processing_time_ms=0.0,
                        image_shape=image.shape,
                        model_used=target_model.value,
                        error=f"Failed to switch to {target_model.value}"
                    )
            
            # Check if model is loaded
            if not self.models_loaded.get(target_model, False):
                return SimpleDetectionResult(
                    faces=[],
                    processing_time_ms=0.0,
                    image_shape=image.shape,
                    model_used=target_model.value,
                    error=f"Model {target_model.value} not loaded"
                )
            
            # Use provided threshold or default
            if conf_threshold is None:
                conf_threshold = self.default_conf
            
            # Run detection based on model type
            detection_start = time.time()
            
            if target_model == ModelType.YOLO11M:
                # Use ultralytics YOLO
                model = self.models[target_model]
                results = model(
                    image,
                    conf=conf_threshold,
                    iou=self.default_iou,
                    imgsz=self.input_size,
                    verbose=False,
                    device=self.device
                )
                detections = self._process_yolo_results(results, image.shape)
                
            elif target_model == ModelType.YOLO11N:
                # Use ONNX runtime
                session = self.onnx_sessions[target_model]
                
                # Preprocess image
                input_tensor = self._preprocess_image_for_onnx(image)
                
                # Get input name
                input_name = session.get_inputs()[0].name
                
                # Run inference
                outputs = session.run(None, {input_name: input_tensor})
                
                # Postprocess results
                detections = self._postprocess_onnx_results(outputs[0], image.shape, conf_threshold)
            
            else:
                raise ValueError(f"Unknown model type: {target_model}")
            
            detection_time = time.time() - detection_start
            
            # Filter by quality if specified
            if min_quality > 0.0:
                detections = [d for d in detections if d.quality_score and d.quality_score >= min_quality]
            
            # Limit max faces
            if max_faces > 0:
                # Sort by confidence and take top N
                detections.sort(key=lambda x: x.confidence, reverse=True)
                detections = detections[:max_faces]
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Update performance tracking
            self.total_detections += 1
            self.total_processing_time += processing_time_ms
            self.last_processing_time = processing_time_ms
            
            return SimpleDetectionResult(
                faces=detections,
                processing_time_ms=processing_time_ms,
                image_shape=image.shape,
                model_used=target_model.value,
                error=None
            )
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Face detection failed: {e}")
            
            return SimpleDetectionResult(
                faces=[],
                processing_time_ms=processing_time_ms,
                image_shape=image.shape if 'image' in locals() else (0, 0, 0),
                model_used=target_model.value if 'target_model' in locals() else "unknown",
                error=str(e)
            )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_time = (
            self.total_processing_time / self.total_detections 
            if self.total_detections > 0 else 0.0
        )
        
        return {
            "total_detections": self.total_detections,
            "average_processing_time_ms": round(avg_time, 2),
            "last_processing_time_ms": round(self.last_processing_time, 2),
            "estimated_fps": round(1000.0 / avg_time, 2) if avg_time > 0 else 0.0,
            "device": self.device,
            "current_model": self.current_model_type.value,
            "loaded_models": [model_type.value for model_type, loaded in self.models_loaded.items() if loaded]
        }

    async def benchmark(
        self, 
        iterations: int = 100,
        model_type: Optional[ModelType] = None,
        image_size: Tuple[int, int] = (640, 480)
    ) -> Dict[str, Any]:
        """
        Run benchmark test
        """
        target_model = model_type or self.current_model_type
        
        # Switch to target model if needed
        if target_model != self.current_model_type:
            switch_success = await self.switch_model(target_model)
            if not switch_success:
                return {"error": f"Failed to switch to {target_model.value}"}
        
        logger.info(f"🚀 Starting benchmark with {target_model.value} ({iterations} iterations)")
        
        # Create test image
        test_image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        
        times = []
        
        # Warmup
        for _ in range(5):
            await self.detect_faces(test_image, target_model)
        
        # Benchmark
        start_time = time.time()
        
        for i in range(iterations):
            result = await self.detect_faces(test_image, target_model)
            times.append(result.processing_time_ms)
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        
        fps = 1000.0 / avg_time if avg_time > 0 else 0.0
        
        return {
            "model": target_model.value,
            "iterations": iterations,
            "total_time_seconds": round(total_time, 2),
            "average_time_ms": round(avg_time, 2),
            "min_time_ms": round(min_time, 2),
            "max_time_ms": round(max_time, 2),
            "std_time_ms": round(std_time, 2),
            "fps": round(fps, 2),
            "device": self.device,
            "image_size": image_size
        }

    def is_ready(self, model_type: Optional[ModelType] = None) -> bool:
        """Check if service is ready"""
        target_model = model_type or self.current_model_type
        return self.models_loaded.get(target_model, False)

    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return [model_type.value for model_type in ModelType]

    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information"""
        return {
            "current_model": self.current_model_type.value,
            "available_models": self.get_available_models(),
            "loaded_models": [model_type.value for model_type, loaded in self.models_loaded.items() if loaded],
            "model_paths": {model_type.value: path for model_type, path in self.model_paths.items()},
            "device": self.device,
            "onnx_available": ONNX_AVAILABLE
        }


# Global service instance
multi_model_service = MultiModelService()