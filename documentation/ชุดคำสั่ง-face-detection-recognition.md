# ชุดคำสั่ง: การตรวจจับและจดจำใบหน้า
## Face Detection และ Face Recognition Systems

### 📋 สารบัญ
3.1 [ภาพรวม Face Detection System](#31-ภาพรวม-face-detection-system)
3.2 [YOLO Models Implementation](#32-yolo-models-implementation)
3.3 [Face Recognition Service](#33-face-recognition-service)
3.4 [Face Embeddings Management](#34-face-embeddings-management)
3.5 [API Integration](#35-api-integration)
3.6 [Performance Optimization](#36-performance-optimization)
3.7 [Model Management](#37-model-management)
3.8 [Error Handling และ Logging](#38-error-handling-และ-logging)

---

## 3.1 ภาพรวม Face Detection System

ระบบตรวจจับและจดจำใบหน้าที่ใช้ AI Models หลายตัวเพื่อความแม่นยำสูงสุด

### 🎯 ความสามารถหลัก
- **Face Detection**: YOLO v9c, v9e, v11m models
- **Face Recognition**: FaceNet, AdaFace, ArcFace 
- **Quality Assessment**: ประเมินคุณภาพใบหน้าแบบอัตโนมัติ
- **GPU Optimization**: การจัดการ VRAM อย่างมีประสิทธิภาพ

---

## 3.2 YOLO Models Implementation

### 3.2.1 YOLO Models สำหรับการตรวจจับใบหน้า

```python
"""
Enhanced Face Detection Service
ระบบตรวจจับใบหน้าอัจฉริยะที่รองรับโมเดล YOLOv9c, YOLOv9e และ YOLOv11m
"""

import time
import logging
import cv2
import numpy as np
import os
import asyncio
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class QualityCategory(Enum):
    """ระดับคุณภาพของใบหน้า"""
    EXCELLENT = "excellent"  # 80-100
    GOOD = "good"  # 70-79
    ACCEPTABLE = "acceptable"  # 40-69
    POOR = "poor"  # < 40

@dataclass
class BoundingBox:
    """กล่องครอบใบหน้า"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

@dataclass
class FaceDetection:
    """ผลการตรวจจับใบหน้า"""
    bbox: BoundingBox
    quality_score: Optional[float] = None
    landmarks: Optional[np.ndarray] = None
    pose_info: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": {
                "x1": float(self.bbox.x1),
                "y1": float(self.bbox.y1),
                "x2": float(self.bbox.x2),
                "y2": float(self.bbox.y2),
                "confidence": float(self.bbox.confidence)
            },
            "quality_score": float(self.quality_score) if self.quality_score else None,
            "landmarks": self.landmarks.tolist() if self.landmarks is not None else None,
            "pose_info": self.pose_info
        }

class FaceDetectionService:
    """บริการตรวจจับใบหน้าหลัก"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.models = {}
        self.current_model = None
        self.is_initialized = False
        
        # Initialize quality analyzer
        from .enhanced_quality_assessment import FaceQualityAnalyzer
        self.quality_analyzer = FaceQualityAnalyzer(self.config)

    def _get_default_config(self) -> Dict[str, Any]:
        """การตั้งค่าเริ่มต้น"""
        return {
            "models": {
                "yolov9c": {
                    "path": "model/face-detection/yolov9c-face-lindevs.onnx",
                    "input_size": (640, 640),
                    "conf_threshold": 0.5,
                    "iou_threshold": 0.4
                },
                "yolov9e": {
                    "path": "model/face-detection/yolov9e-face-lindevs.onnx",
                    "input_size": (640, 640),
                    "conf_threshold": 0.5,
                    "iou_threshold": 0.4
                },
                "yolov11m": {
                    "path": "model/face-detection/yolov11m-face.pt",
                    "input_size": (640, 640),
                    "conf_threshold": 0.5,
                    "iou_threshold": 0.45
                }
            },
            "fallback_order": ["yolov11m", "yolov9c", "yolov9e"],
            "max_faces": 50,
            "min_quality_threshold": 40.0,
            "use_gpu": True,
            "enable_quality_filtering": True
        }

    async def initialize(self) -> None:
        """เริ่มต้นระบบตรวจจับใบหน้า"""
        logger.info("🚀 Initializing Face Detection Service...")
        
        try:
            # Load available models
            await self._load_models()
            
            # Set primary model
            self._set_primary_model()
            
            self.is_initialized = True
            logger.info("✅ Face Detection Service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Face Detection Service: {e}")
            raise

    async def _load_models(self) -> None:
        """โหลดโมเดลทั้งหมด"""
        for model_name, model_config in self.config["models"].items():
            try:
                if model_name.startswith("yolov11"):
                    from .yolo_models import YOLOv11Detector
                    model = YOLOv11Detector(
                        model_path=model_config["path"],
                        input_size=model_config["input_size"],
                        conf_threshold=model_config["conf_threshold"],
                        iou_threshold=model_config["iou_threshold"],
                        use_gpu=self.config["use_gpu"]
                    )
                else:
                    from .yolo_models import YOLOv9ONNXDetector
                    model = YOLOv9ONNXDetector(
                        model_path=model_config["path"],
                        input_size=model_config["input_size"],
                        conf_threshold=model_config["conf_threshold"],
                        iou_threshold=model_config["iou_threshold"],
                        use_gpu=self.config["use_gpu"]
                    )
                
                await model.initialize()
                self.models[model_name] = model
                logger.info(f"✅ Loaded {model_name} model")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {model_name}: {e}")

    def _set_primary_model(self) -> None:
        """กำหนดโมเดลหลัก"""
        for model_name in self.config["fallback_order"]:
            if model_name in self.models:
                self.current_model = model_name
                logger.info(f"🎯 Primary model set to: {model_name}")
                return
        
        if self.models:
            self.current_model = list(self.models.keys())[0]
            logger.info(f"🎯 Primary model set to: {self.current_model}")
        else:
            raise ValueError("No models available")

    async def detect_faces(
        self, 
        image: np.ndarray, 
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """ตรวจจับใบหน้าในภาพ"""
        if not self.is_initialized:
            await self.initialize()

        start_time = time.time()
        
        # Merge config
        detection_config = {**self.config, **(config or {})}
        
        try:
            # Primary detection
            result = await self._detect_with_model(
                image, 
                self.current_model, 
                detection_config
            )
            
            if result["faces"]:
                logger.info(f"✅ Detection successful with {self.current_model}")
                result["model_used"] = self.current_model
                result["processing_time"] = time.time() - start_time
                return result
            
            # Fallback detection
            logger.info(f"🔄 Primary model failed, trying fallback models...")
            for model_name in self.config["fallback_order"]:
                if model_name != self.current_model and model_name in self.models:
                    try:
                        result = await self._detect_with_model(
                            image, 
                            model_name, 
                            detection_config
                        )
                        
                        if result["faces"]:
                            logger.info(f"✅ Fallback detection successful with {model_name}")
                            result["model_used"] = model_name
                            result["processing_time"] = time.time() - start_time
                            return result
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Fallback model {model_name} failed: {e}")
                        continue
            
            # OpenCV fallback
            logger.info("🔄 Trying OpenCV fallback...")
            result = await self._opencv_fallback_detection(image, detection_config)
            result["model_used"] = "opencv_fallback"
            result["processing_time"] = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"❌ All detection methods failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "faces": [],
                "model_used": "none",
                "processing_time": time.time() - start_time
            }

    async def _detect_with_model(
        self, 
        image: np.ndarray, 
        model_name: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """ตรวจจับด้วยโมเดลเฉพาะ"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")

        model = self.models[model_name]
        
        # Perform detection
        detections = await model.detect(image)
        
        # Process results
        processed_faces = []
        for detection in detections:
            face = FaceDetection(bbox=detection.bbox)
            
            # Calculate quality score
            if config.get("enable_quality_filtering", True):
                face.quality_score = self._calculate_face_quality(
                    image, detection.bbox
                )
                
                # Filter by quality
                if (face.quality_score < config.get("min_quality_threshold", 40.0)):
                    continue
            
            processed_faces.append(face)
        
        # Limit max faces
        max_faces = config.get("max_faces", 50)
        if len(processed_faces) > max_faces:
            # Sort by quality and take top N
            processed_faces.sort(
                key=lambda x: x.quality_score or 0, 
                reverse=True
            )
            processed_faces = processed_faces[:max_faces]
        
        return {
            "success": True,
            "faces": [face.to_dict() for face in processed_faces],
            "total_detections": len(detections),
            "filtered_count": len(processed_faces)
        }

    def _calculate_face_quality(
        self, 
        image: np.ndarray, 
        bbox: BoundingBox
    ) -> float:
        """คำนวณคุณภาพใบหน้า"""
        try:
            # Extract face region
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
            
            # Ensure bounds
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            face_region = image[y1:y2, x1:x2]
            
            # Calculate quality metrics
            size_score = self._calculate_size_score(bbox)
            confidence_score = bbox.confidence * 100
            aspect_score = self._calculate_aspect_score(bbox)
            sharpness_score = self._calculate_sharpness_score(face_region)
            
            # Weighted average
            quality_score = (
                size_score * 0.3 +
                confidence_score * 0.3 +
                aspect_score * 0.2 +
                sharpness_score * 0.2
            )
            
            return min(100.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.warning(f"Quality calculation failed: {e}")
            return 0.0

    def _calculate_size_score(self, bbox: BoundingBox) -> float:
        """คำนวณคะแนนขนาดใบหน้า"""
        width, height = bbox.width, bbox.height
        min_size = min(width, height)
        
        if min_size >= 80:
            return 100.0
        elif min_size >= 50:
            return 80.0
        elif min_size >= 24:
            return 60.0
        else:
            return max(0.0, (min_size / 24.0) * 40.0)

    def _calculate_aspect_score(self, bbox: BoundingBox) -> float:
        """คำนวณคะแนนอัตราส่วนใบหน้า"""
        aspect_ratio = bbox.width / bbox.height
        ideal_ratio = 0.75  # ใบหน้าปกติ
        
        ratio_diff = abs(aspect_ratio - ideal_ratio)
        if ratio_diff <= 0.1:
            return 100.0
        elif ratio_diff <= 0.3:
            return 80.0
        elif ratio_diff <= 0.5:
            return 60.0
        else:
            return 40.0

    def _calculate_sharpness_score(self, face_region: np.ndarray) -> float:
        """คำนวณคะแนนความคมชัดใบหน้า"""
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if laplacian_var >= 1000:
                return 100.0
            elif laplacian_var >= 500:
                return 80.0
            elif laplacian_var >= 100:
                return 60.0
            else:
                return max(0.0, (laplacian_var / 100.0) * 40.0)
                
        except Exception:
            return 50.0

    async def _opencv_fallback_detection(
        self, 
        image: np.ndarray, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """ตรวจจับด้วย OpenCV Haar Cascades (Fallback)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Load Haar cascade
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            detected_faces = []
            for (x, y, w, h) in faces:
                bbox = BoundingBox(
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + w),
                    y2=float(y + h),
                    confidence=0.7  # Default confidence for OpenCV
                )
                
                face = FaceDetection(bbox=bbox)
                face.quality_score = self._calculate_face_quality(image, bbox)
                
                if face.quality_score >= config.get("min_quality_threshold", 40.0):
                    detected_faces.append(face)
            
            return {
                "success": True,
                "faces": [face.to_dict() for face in detected_faces],
                "total_detections": len(faces),
                "filtered_count": len(detected_faces)
            }
            
        except Exception as e:
            logger.error(f"OpenCV fallback detection failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "faces": [],
                "total_detections": 0,
                "filtered_count": 0
            }
```

## 3.3 ชุดคำสั่งการจดจำใบหน้า (Face Recognition)

### 3.3.1 ชุดคำสั่ง Face Recognition Models และ Data Structures

```python
"""
Face Recognition Data Models
Enhanced models with better error handling and validation
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
from enum import Enum
import time

class RecognitionModel(Enum):
    """โมเดลสำหรับการจดจำใบหน้า"""
    ADAFACE = "adaface"
    ARCFACE = "arcface"
    FACENET = "facenet"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, model_string: str) -> Optional['RecognitionModel']:
        """Create RecognitionModel from string with error handling"""
        try:
            return cls(model_string.lower().strip())
        except ValueError:
            return None

    @classmethod
    def get_all_values(cls) -> List[str]:
        """Get all model values as strings"""
        return [model.value for model in cls]

class RecognitionQuality(Enum):
    """ระดับคุณภาพของใบหน้าสำหรับการจดจำ"""
    HIGH = "high"  # คุณภาพดี > 80%
    MEDIUM = "medium"  # คุณภาพปานกลาง 50-80%
    LOW = "low"  # คุณภาพต่ำ < 50%
    UNKNOWN = "unknown"  # ไม่สามารถประเมินได้

    @classmethod
    def from_score(cls, score: float) -> 'RecognitionQuality':
        """Get quality level from score"""
        if score >= 0.8:
            return cls.HIGH
        elif score >= 0.5:
            return cls.MEDIUM
        elif score > 0:
            return cls.LOW
        else:
            return cls.UNKNOWN

@dataclass
class FaceQuality:
    """คุณภาพของใบหน้าสำหรับการจดจำ"""
    score: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    sharpness: float = 0.0
    pose_quality: float = 0.0
    blur_level: float = 0.0
    lighting_quality: float = 0.0

    @property
    def overall_quality(self) -> RecognitionQuality:
        """ประเมินคุณภาพโดยรวม"""
        return RecognitionQuality.from_score(self.score / 100.0)

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary"""
        return {
            "score": float(self.score),
            "brightness": float(self.brightness),
            "contrast": float(self.contrast),
            "sharpness": float(self.sharpness),
            "pose_quality": float(self.pose_quality),
            "blur_level": float(self.blur_level),
            "lighting_quality": float(self.lighting_quality),
            "overall_quality": self.overall_quality.value,
        }

@dataclass
class FaceEmbedding:
    """Vector embedding ของใบหน้า"""
    vector: np.ndarray
    model_name: str
    quality: FaceQuality
    extraction_time: float
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        """Validate embedding after initialization"""
        if not isinstance(self.vector, np.ndarray):
            raise ValueError("Embedding vector must be numpy array")
        
        if len(self.vector.shape) != 1:
            raise ValueError("Embedding vector must be 1-dimensional")
        
        # Normalize vector if not already normalized
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            self.vector = self.vector / norm

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary สำหรับ serialization"""
        return {
            "vector": self.vector.tolist(),
            "model_name": self.model_name,
            "quality": self.quality.to_dict(),
            "extraction_time": self.extraction_time,
            "person_id": self.person_id,
            "person_name": self.person_name,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "vector_dimensions": len(self.vector)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FaceEmbedding':
        """สร้าง FaceEmbedding จาก dictionary"""
        quality_data = data.get("quality", {})
        quality = FaceQuality(
            score=quality_data.get("score", 0.0),
            brightness=quality_data.get("brightness", 0.0),
            contrast=quality_data.get("contrast", 0.0),
            sharpness=quality_data.get("sharpness", 0.0),
            pose_quality=quality_data.get("pose_quality", 0.0),
            blur_level=quality_data.get("blur_level", 0.0),
            lighting_quality=quality_data.get("lighting_quality", 0.0)
        )
        
        return cls(
            vector=np.array(data["vector"]),
            model_name=data["model_name"],
            quality=quality,
            extraction_time=data.get("extraction_time", 0.0),
            person_id=data.get("person_id"),
            person_name=data.get("person_name"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time())
        )

@dataclass
class RecognitionMatch:
    """ผลการจับคู่ใบหน้า"""
    person_id: str
    person_name: str
    confidence: float
    similarity: float
    embedding: FaceEmbedding
    match_quality: RecognitionQuality
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary"""
        return {
            "person_id": self.person_id,
            "person_name": self.person_name,
            "confidence": float(self.confidence),
            "similarity": float(self.similarity),
            "match_quality": self.match_quality.value,
            "embedding": self.embedding.to_dict(),
            "metadata": self.metadata
        }

@dataclass
class RecognitionResult:
    """ผลการจดจำใบหน้า"""
    success: bool
    matches: List[RecognitionMatch] = field(default_factory=list)
    best_match: Optional[RecognitionMatch] = None
    query_embedding: Optional[FaceEmbedding] = None
    processing_time: float = 0.0
    model_used: str = "unknown"
    error_message: Optional[str] = None
    total_comparisons: int = 0

    @property
    def has_matches(self) -> bool:
        """ตรวจสอบว่ามีการจับคู่หรือไม่"""
        return len(self.matches) > 0

    @property
    def best_confidence(self) -> float:
        """ความมั่นใจสูงสุด"""
        if self.best_match:
            return self.best_match.confidence
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary"""
        return {
            "success": self.success,
            "matches": [match.to_dict() for match in self.matches],
            "best_match": self.best_match.to_dict() if self.best_match else None,
            "query_embedding": self.query_embedding.to_dict() if self.query_embedding else None,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "error_message": self.error_message,
            "total_comparisons": self.total_comparisons,
            "has_matches": self.has_matches,
            "best_confidence": self.best_confidence
        }

# Type aliases for convenience
EmbeddingVector = np.ndarray
FaceGallery = Dict[str, Dict[str, Any]]
```

### 3.3.2 ชุดคำสั่ง Face Recognition Service

```python
"""
Face Recognition Service
บริการจดจำใบหน้าด้วยโมเดล FaceNet, AdaFace, และ ArcFace
"""

import time
import logging
import cv2
import numpy as np
import os
from typing import Any, Dict, List, Optional, Union, Tuple
import onnxruntime as ort
from scipy.spatial.distance import cosine, euclidean

from .models import (
    RecognitionModel, 
    FaceEmbedding, 
    FaceQuality,
    RecognitionMatch,
    RecognitionResult,
    RecognitionQuality
)

logger = logging.getLogger(__name__)

class FaceRecognitionService:
    """บริการจดจำใบหน้าหลัก"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.models = {}
        self.gallery = {}  # In-memory gallery
        self.is_initialized = False

    def _get_default_config(self) -> Dict[str, Any]:
        """การตั้งค่าเริ่มต้น"""
        return {
            "models": {
                "facenet": {
                    "path": "model/face-recognition/facenet_vggface2.onnx",
                    "input_size": (160, 160),
                    "normalize": True,
                    "similarity_threshold": 0.6
                },
                "adaface": {
                    "path": "model/face-recognition/adaface_ir101.onnx",
                    "input_size": (112, 112),
                    "normalize": True,
                    "similarity_threshold": 0.3
                },
                "arcface": {
                    "path": "model/face-recognition/arcface_r100.onnx",
                    "input_size": (112, 112),
                    "normalize": True,
                    "similarity_threshold": 0.4
                }
            },
            "default_model": "facenet",
            "use_gpu": True,
            "max_gallery_size": 10000,
            "similarity_metric": "cosine",  # cosine or euclidean
            "top_k_matches": 5
        }

    async def initialize(self) -> None:
        """เริ่มต้นบริการจดจำใบหน้า"""
        logger.info("🚀 Initializing Face Recognition Service...")
        
        try:
            # Load models
            await self._load_models()
            
            # Load existing gallery
            await self._load_gallery()
            
            self.is_initialized = True
            logger.info("✅ Face Recognition Service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Face Recognition Service: {e}")
            raise

    async def _load_models(self) -> None:
        """โหลดโมเดลทั้งหมด"""
        for model_name, model_config in self.config["models"].items():
            try:
                model_path = model_config["path"]
                
                if not os.path.exists(model_path):
                    logger.warning(f"⚠️ Model file not found: {model_path}")
                    continue
                
                # Create ONNX session
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.config["use_gpu"] else ['CPUExecutionProvider']
                session = ort.InferenceSession(model_path, providers=providers)
                
                self.models[model_name] = {
                    "session": session,
                    "config": model_config,
                    "input_name": session.get_inputs()[0].name,
                    "output_name": session.get_outputs()[0].name
                }
                
                logger.info(f"✅ Loaded {model_name} model")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {model_name}: {e}")

    async def _load_gallery(self) -> None:
        """โหลด gallery จากฐานข้อมูล"""
        # TODO: Load from persistent storage
        logger.info("📂 Gallery loaded (in-memory)")

    async def extract_embedding(
        self, 
        face_image: np.ndarray, 
        model_name: str = None
    ) -> FaceEmbedding:
        """สกัด embedding จากใบหน้า"""
        if not self.is_initialized:
            await self.initialize()
        
        model_name = model_name or self.config["default_model"]
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        start_time = time.time()
        
        try:
            # Preprocess face image
            processed_face = self._preprocess_face(face_image, model_name)
            
            # Extract embedding
            model_info = self.models[model_name]
            session = model_info["session"]
            
            embedding_raw = session.run(
                [model_info["output_name"]], 
                {model_info["input_name"]: processed_face}
            )[0]
            
            # Flatten and normalize
            embedding_vector = embedding_raw.flatten()
            if model_info["config"]["normalize"]:
                norm = np.linalg.norm(embedding_vector)
                if norm > 0:
                    embedding_vector = embedding_vector / norm
            
            # Calculate quality
            quality = self._assess_face_quality(face_image)
            
            # Create embedding object
            embedding = FaceEmbedding(
                vector=embedding_vector,
                model_name=model_name,
                quality=quality,
                extraction_time=time.time() - start_time
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            raise

    def _preprocess_face(self, face_image: np.ndarray, model_name: str) -> np.ndarray:
        """ประมวลผลใบหน้าสำหรับโมเดล"""
        model_config = self.config["models"][model_name]
        input_size = model_config["input_size"]
        
        # Resize
        face_resized = cv2.resize(face_image, input_size)
        
        # Convert to RGB if needed
        if len(face_resized.shape) == 3 and face_resized.shape[2] == 3:
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        # Standardize (mean=0.5, std=0.5)
        face_standardized = (face_normalized - 0.5) / 0.5
        
        # Add batch dimension and transpose for ONNX
        face_final = np.transpose(face_standardized, (2, 0, 1))
        face_final = np.expand_dims(face_final, axis=0)
        
        return face_final

    def _assess_face_quality(self, face_image: np.ndarray) -> FaceQuality:
        """ประเมินคุณภาพใบหน้า"""
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Brightness
            brightness = np.mean(gray)
            
            # Contrast (standard deviation)
            contrast = np.std(gray)
            
            # Sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Overall score
            brightness_score = min(100, (brightness / 128.0) * 100)
            contrast_score = min(100, (contrast / 64.0) * 100)
            sharpness_score = min(100, (sharpness / 1000.0) * 100)
            
            overall_score = (brightness_score + contrast_score + sharpness_score) / 3
            
            return FaceQuality(
                score=overall_score,
                brightness=brightness_score,
                contrast=contrast_score,
                sharpness=sharpness_score,
                pose_quality=75.0,  # Default value
                blur_level=100.0 - sharpness_score,
                lighting_quality=brightness_score
            )
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return FaceQuality(score=50.0)

    async def add_face_to_gallery(
        self, 
        face_image: np.ndarray,
        person_id: str,
        person_name: str,
        model_name: str = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """เพิ่มใบหน้าเข้า gallery"""
        try:
            # Extract embedding
            embedding = await self.extract_embedding(face_image, model_name)
            embedding.person_id = person_id
            embedding.person_name = person_name
            embedding.metadata = metadata or {}
            
            # Add to gallery
            if person_id not in self.gallery:
                self.gallery[person_id] = {
                    "person_name": person_name,
                    "embeddings": [],
                    "metadata": metadata or {}
                }
            
            self.gallery[person_id]["embeddings"].append(embedding)
            
            logger.info(f"✅ Added face for {person_name} ({person_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add face to gallery: {e}")
            return False

    async def recognize_face(
        self, 
        face_image: np.ndarray,
        model_name: str = None,
        top_k: int = None,
        similarity_threshold: float = None
    ) -> RecognitionResult:
        """จดจำใบหน้า"""
        start_time = time.time()
        model_name = model_name or self.config["default_model"]
        top_k = top_k or self.config["top_k_matches"]
        
        try:
            # Extract query embedding
            query_embedding = await self.extract_embedding(face_image, model_name)
            
            # Get similarity threshold
            threshold = similarity_threshold or self.config["models"][model_name]["similarity_threshold"]
            
            # Search gallery
            matches = []
            total_comparisons = 0
            
            for person_id, person_data in self.gallery.items():
                for stored_embedding in person_data["embeddings"]:
                    if stored_embedding.model_name == model_name:
                        # Calculate similarity
                        similarity = self._calculate_similarity(
                            query_embedding.vector, 
                            stored_embedding.vector
                        )
                        
                        total_comparisons += 1
                        
                        if similarity >= threshold:
                            confidence = similarity
                            match_quality = RecognitionQuality.from_score(similarity)
                            
                            match = RecognitionMatch(
                                person_id=person_id,
                                person_name=person_data["person_name"],
                                confidence=confidence,
                                similarity=similarity,
                                embedding=stored_embedding,
                                match_quality=match_quality,
                                metadata=person_data["metadata"]
                            )
                            
                            matches.append(match)
            
            # Sort by confidence and take top k
            matches.sort(key=lambda x: x.confidence, reverse=True)
            matches = matches[:top_k]
            
            # Determine best match
            best_match = matches[0] if matches else None
            
            return RecognitionResult(
                success=True,
                matches=matches,
                best_match=best_match,
                query_embedding=query_embedding,
                processing_time=time.time() - start_time,
                model_used=model_name,
                total_comparisons=total_comparisons
            )
            
        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            return RecognitionResult(
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time,
                model_used=model_name
            )

    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """คำนวณความคล้ายคลึงระหว่าง embedding"""
        if self.config["similarity_metric"] == "cosine":
            # Cosine similarity (1 - cosine distance)
            return 1.0 - cosine(embedding1, embedding2)
        elif self.config["similarity_metric"] == "euclidean":
            # Euclidean distance converted to similarity
            distance = euclidean(embedding1, embedding2)
            return 1.0 / (1.0 + distance)
        else:
            # Default: dot product (for normalized vectors)
            return np.dot(embedding1, embedding2)

    async def compare_faces(
        self, 
        face1: np.ndarray, 
        face2: np.ndarray,
        model_name: str = None
    ) -> Dict[str, Any]:
        """เปรียบเทียบใบหน้าสองภาพ"""
        try:
            # Extract embeddings
            embedding1 = await self.extract_embedding(face1, model_name)
            embedding2 = await self.extract_embedding(face2, model_name)
            
            # Calculate similarity
            similarity = self._calculate_similarity(
                embedding1.vector, 
                embedding2.vector
            )
            
            # Determine if same person
            threshold = self.config["models"][embedding1.model_name]["similarity_threshold"]
            is_same_person = similarity >= threshold
            
            return {
                "success": True,
                "is_same_person": is_same_person,
                "similarity": float(similarity),
                "confidence": float(similarity),
                "threshold": threshold,
                "model_used": embedding1.model_name,
                "embedding1": embedding1.to_dict(),
                "embedding2": embedding2.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Face comparison failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "is_same_person": False,
                "similarity": 0.0,
                "confidence": 0.0
            }

    def get_gallery_stats(self) -> Dict[str, Any]:
        """ข้อมูลสถิติ gallery"""
        total_people = len(self.gallery)
        total_embeddings = sum(
            len(person_data["embeddings"]) 
            for person_data in self.gallery.values()
        )
        
        model_counts = {}
        for person_data in self.gallery.values():
            for embedding in person_data["embeddings"]:
                model_name = embedding.model_name
                model_counts[model_name] = model_counts.get(model_name, 0) + 1
        
        return {
            "total_people": total_people,
            "total_embeddings": total_embeddings,
            "model_counts": model_counts,
            "gallery_size_limit": self.config["max_gallery_size"]
        }
```

---

*เอกสารนี้แสดงชุดคำสั่งหลักสำหรับการตรวจจับและจดจำใบหน้า รวมถึงการใช้งานโมเดล YOLO สำหรับการตรวจจับและโมเดล FaceNet, AdaFace, ArcFace สำหรับการจดจำใบหน้า*
