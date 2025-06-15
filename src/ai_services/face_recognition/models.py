"""
Face Recognition Data Models - Fixed Version
Enhanced models with better error handling and validation
แก้ไขปัญหา Enum และ backwards compatibility
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
from enum import Enum
import time


class RecognitionModel(Enum):
    """โมเดลสำหรับการจดจำใบหน้า - Fixed Enum"""

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


# Alias for backward compatibility
ModelType = RecognitionModel


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
    """คุณภาพของใบหน้าสำหรับการจดจำ - Enhanced"""

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


# Type aliases
EmbeddingVector = np.ndarray
FaceGallery = Dict[str, Dict[str, Any]]


@dataclass
class FaceEmbedding:
    """ผลลัพธ์การสกัด embedding vector จากรูปภาพใบหน้า - Enhanced & Fixed"""

    # Main fields for new system
    id: str  # Unique ID for this specific embedding instance
    person_id: str  # ID of the person this embedding belongs to
    person_name: Optional[str] = None  # Name of the person
    vector: Optional[np.ndarray] = None
    # Model type used for this embedding - Fixed handling
    model_type: Optional[RecognitionModel] = None
    timestamp: float = field(default_factory=time.time)
    quality_score: float = 0.0
    extraction_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Enhanced fields
    confidence: float = 0.0
    dimension: int = 0
    normalized: bool = False
    # Method used for extraction (e.g., 'onnx_facenet', 'framework_deepface')
    extraction_method: str = ""
    source_image_hash: Optional[str] = None  # Hash of the source image
    # Bounding box of the face in the source image [x1, y1, x2, y2]
    face_bbox: Optional[List[int]] = None
    # Landmark points of the face
    landmarks: Optional[List[List[int]]] = None
    # Face quality assessment
    face_quality: RecognitionQuality = RecognitionQuality.UNKNOWN

    def __post_init__(self) -> None:
        """Post-initialization processing with improved error handling"""
        # Handle vector dimension and normalization
        if self.vector is not None:
            self.dimension = self.vector.shape[0] if hasattr(self.vector, 'shape') else len(self.vector)
            # Check if vector is L2 normalized (sum of squares is close to 1)
            if self.dimension > 0:
                try:
                    norm_sq = np.sum(np.square(self.vector))
                    self.normalized = np.isclose(norm_sq, 1.0, atol=1e-5)
                except Exception:
                    self.normalized = False

        # Handle model_type conversion with improved error handling
        if self.model_type is not None:
            if isinstance(self.model_type, str):
                # Try to convert string to enum
                converted_model = RecognitionModel.from_string(self.model_type)
                if converted_model is not None:
                    self.model_type = converted_model
                else:
                    # Keep as string if conversion fails (for framework models)
                    pass
            # If it's already a RecognitionModel, keep it as is

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling numpy array serialization."""
        # Handle model_type conversion safely
        model_type_str = None
        if self.model_type is not None:
            if isinstance(self.model_type, RecognitionModel):
                model_type_str = self.model_type.value
            else:
                model_type_str = str(self.model_type)

        data = {
            "id": self.id,
            "person_id": self.person_id,
            "person_name": self.person_name,
            "vector": self.vector.tolist() if self.vector is not None else None,
            "model_type": model_type_str,
            "timestamp": self.timestamp,
            "quality_score": self.quality_score,
            "extraction_time": self.extraction_time,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "dimension": self.dimension,
            "normalized": self.normalized,
            "extraction_method": self.extraction_method,
            "source_image_hash": self.source_image_hash,
            "face_bbox": self.face_bbox,
            "landmarks": self.landmarks,
            "face_quality": self.face_quality.value,
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FaceEmbedding":
        """Create FaceEmbedding from dictionary with improved error handling."""
        # Handle vector conversion
        vector = None
        if data.get("vector") is not None:
            try:
                vector = np.array(data["vector"], dtype=np.float32)
            except Exception:
                vector = None

        # Handle model_type conversion
        model_type_data = data.get("model_type")
        model_type_enum: Optional[RecognitionModel] = None
        if model_type_data:
            if isinstance(model_type_data, str):
                model_type_enum = RecognitionModel.from_string(model_type_data)
            # Keep as None if conversion fails

        # Handle face_quality conversion
        face_quality_str = data.get("face_quality", "unknown")
        face_quality_enum = RecognitionQuality.UNKNOWN
        try:
            face_quality_enum = RecognitionQuality(face_quality_str)
        except (ValueError, TypeError):
            face_quality_enum = RecognitionQuality.UNKNOWN

        return cls(
            id=data.get("id", ""),
            person_id=data.get("person_id", ""),
            person_name=data.get("person_name"),
            vector=vector,
            model_type=model_type_enum,
            timestamp=data.get("timestamp", time.time()),
            quality_score=data.get("quality_score", 0.0),
            extraction_time=data.get("extraction_time", 0.0),
            metadata=data.get("metadata", {}),
            confidence=data.get("confidence", 0.0),
            normalized=data.get("normalized", False),
            extraction_method=data.get("extraction_method", ""),
            source_image_hash=data.get("source_image_hash"),
            face_bbox=data.get("face_bbox"),
            landmarks=data.get("landmarks"),
            face_quality=face_quality_enum,
        )


@dataclass
class FaceMatch:
    """ผลการจับคู่ใบหน้ากับคนในฐานข้อมูล - Enhanced & Fixed"""

    person_id: str
    confidence: float
    embedding: Optional[FaceEmbedding] = None

    # Enhanced fields
    similarity_score: float = 0.0
    distance: float = 0.0
    rank: int = 0
    match_quality: RecognitionQuality = RecognitionQuality.UNKNOWN
    comparison_method: str = "cosine_similarity"

    # Legacy fields for backward compatibility
    identity_id: str = field(init=False)
    similarity: float = field(init=False)
    is_match: bool = field(init=False)
    identity_name: Optional[str] = None
    person_name: Optional[str] = None

    def __post_init__(self) -> None:
        # Set legacy fields for backward compatibility
        self.identity_id = self.person_id
        self.similarity = self.confidence
        # Default is_match to True for any valid confidence
        # This will be properly set by the service using unknown_threshold
        self.is_match = self.confidence > 0.0

        # Set similarity_score if not provided
        if self.similarity_score == 0.0:
            self.similarity_score = self.confidence

        # Set match quality based on confidence (optimized thresholds)
        self.match_quality = RecognitionQuality.from_score(self.confidence)
    
    def set_match_status(self, unknown_threshold: float) -> None:
        """Set is_match based on unknown_threshold from config"""
        self.is_match = self.confidence >= unknown_threshold

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary สำหรับ JSON serialization"""
        return {
            "person_id": self.person_id,
            "person_name": self.person_name,
            "confidence": float(self.confidence),
            "similarity_score": float(self.similarity_score),
            "distance": float(self.distance),
            "rank": int(self.rank),
            "match_quality": self.match_quality.value,
            "comparison_method": self.comparison_method,
            "is_match": bool(self.is_match),
            # Legacy fields
            "identity_id": self.identity_id,
            "identity_name": self.identity_name,
            "similarity": float(self.similarity),
        }


@dataclass
class FaceComparisonResult:
    """ผลลัพธ์การเปรียบเทียบใบหน้าสองใบ - Enhanced"""

    similarity: float
    is_same_person: bool
    confidence: float
    processing_time: float
    model_used: Optional[RecognitionModel] = None
    error: Optional[str] = None

    # Enhanced fields
    distance: float = 0.0
    comparison_method: str = "cosine_similarity"
    threshold_used: float = 0.5  # Lowered from 0.6 to 0.5
    quality_assessment: Optional[FaceQuality] = None

    # Legacy fields for backward compatibility
    is_match: bool = field(init=False)
    success: bool = True

    def __post_init__(self) -> None:
        # Set legacy fields for backward compatibility
        self.is_match = self.is_same_person
        self.success = self.error is None

        # Calculate distance from similarity if not provided
        if self.distance == 0.0:
            self.distance = 1.0 - self.similarity

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary สำหรับ JSON serialization"""
        result = {
            "similarity": float(self.similarity),
            "is_same_person": bool(self.is_same_person),
            "confidence": float(self.confidence),
            "processing_time": float(self.processing_time),
            "model_used": self.model_used.value if self.model_used else None,
            "error": self.error,
            "distance": float(self.distance),
            "comparison_method": self.comparison_method,
            "threshold_used": float(self.threshold_used),
            "success": bool(self.success),
            # Legacy fields
            "is_match": bool(self.is_match),
        }

        if self.quality_assessment:
            result["quality_assessment"] = self.quality_assessment.to_dict()

        return result


@dataclass
class FaceRecognitionResult:
    """ผลลัพธ์การจดจำใบหน้า - Enhanced"""

    matches: List[FaceMatch]
    best_match: Optional[FaceMatch] = None
    confidence: float = 0.0
    processing_time: float = 0.0
    model_used: Optional[RecognitionModel] = None
    error: Optional[str] = None

    # Enhanced fields
    query_embedding: Optional[FaceEmbedding] = None
    total_candidates: int = 0
    search_time: float = 0.0
    embedding_time: float = 0.0

    # Legacy fields for backward compatibility
    embedding: Optional[np.ndarray] = None
    success: bool = True
    embedding_quality: RecognitionQuality = RecognitionQuality.UNKNOWN

    def __post_init__(self) -> None:
        self.success = self.error is None

        # Set total candidates
        self.total_candidates = len(self.matches)

        # Set legacy embedding field
        if self.query_embedding and self.query_embedding.vector is not None:
            self.embedding = self.query_embedding.vector
            self.embedding_quality = self.query_embedding.face_quality

    @property
    def has_match(self) -> bool:
        """ตรวจสอบว่ามีการจับคู่ที่ดีหรือไม่"""
        return self.best_match is not None and self.best_match.is_match

    @property
    def identity(self) -> Optional[str]:
        """ดึง identity ที่จับคู่ได้"""
        return self.best_match.person_id if self.has_match else None

    def get_top_matches(self, n: int = 5) -> List[FaceMatch]:
        """ดึง top N matches"""
        return sorted(self.matches, key=lambda m: m.confidence, reverse=True)[:n]

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary สำหรับ JSON serialization"""
        result = {
            "matches": [match.to_dict() for match in self.matches],
            "best_match": self.best_match.to_dict() if self.best_match else None,
            "confidence": float(self.confidence),
            "processing_time": float(self.processing_time),
            "model_used": self.model_used.value if self.model_used else None,
            "error": self.error,
            "total_candidates": int(self.total_candidates),
            "search_time": float(self.search_time),
            "embedding_time": float(self.embedding_time),
            "has_match": bool(self.has_match),
            "identity": self.identity,
            "success": bool(self.success),
            "embedding_quality": self.embedding_quality.value,
        }

        if self.query_embedding:
            result["query_embedding"] = self.query_embedding.to_dict()

        return result


@dataclass
class RecognitionConfig:
    """การตั้งค่าสำหรับ Face Recognition Service - Enhanced"""

    # Model settings
    preferred_model: RecognitionModel = RecognitionModel.FACENET
    similarity_threshold: float = 0.50  # Lowered from 0.60
    unknown_threshold: float = 0.40     # Lowered from 0.55
    embedding_dimension: int = 512

    # Processing settings
    max_faces: int = 10
    quality_threshold: float = 0.2
    batch_size: int = 8
    enable_gpu_optimization: bool = True

    # Performance settings
    cuda_memory_fraction: float = 0.8
    use_cuda_graphs: bool = False
    parallel_processing: bool = True

    # Quality assessment
    enable_quality_assessment: bool = True
    auto_model_selection: bool = True
    enable_unknown_detection: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary"""
        return {
            "preferred_model": self.preferred_model.value,
            "similarity_threshold": float(self.similarity_threshold),
            "unknown_threshold": float(self.unknown_threshold),
            "embedding_dimension": int(self.embedding_dimension),
            "max_faces": int(self.max_faces),
            "quality_threshold": float(self.quality_threshold),
            "batch_size": int(self.batch_size),
            "enable_gpu_optimization": bool(self.enable_gpu_optimization),
            "cuda_memory_fraction": float(self.cuda_memory_fraction),
            "use_cuda_graphs": bool(self.use_cuda_graphs),
            "parallel_processing": bool(self.parallel_processing),
            "enable_quality_assessment": bool(self.enable_quality_assessment),
            "auto_model_selection": bool(self.auto_model_selection),
            "enable_unknown_detection": bool(self.enable_unknown_detection),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecognitionConfig":
        """Create RecognitionConfig from dictionary"""
        # Handle preferred_model conversion
        preferred_model_str = data.get("preferred_model", "facenet")
        preferred_model = RecognitionModel.from_string(preferred_model_str)
        if preferred_model is None:
            preferred_model = RecognitionModel.FACENET

        return cls(
            preferred_model=preferred_model,
            similarity_threshold=data.get("similarity_threshold", 0.50),
            unknown_threshold=data.get("unknown_threshold", 0.40),
            embedding_dimension=data.get("embedding_dimension", 512),
            max_faces=data.get("max_faces", 10),
            quality_threshold=data.get("quality_threshold", 0.2),
            batch_size=data.get("batch_size", 8),
            enable_gpu_optimization=data.get("enable_gpu_optimization", True),
            cuda_memory_fraction=data.get("cuda_memory_fraction", 0.8),
            use_cuda_graphs=data.get("use_cuda_graphs", False),
            parallel_processing=data.get("parallel_processing", True),
            enable_quality_assessment=data.get("enable_quality_assessment", True),
            auto_model_selection=data.get("auto_model_selection", True),
            enable_unknown_detection=data.get("enable_unknown_detection", True),
        )


# Utility functions for backward compatibility
def get_model_from_string(model_string: str) -> Optional[RecognitionModel]:
    """Get RecognitionModel from string - utility function"""
    return RecognitionModel.from_string(model_string)


def get_all_model_names() -> List[str]:
    """Get all available model names"""
    return RecognitionModel.get_all_values()


# Export all classes and functions
__all__ = [
    "RecognitionModel", "ModelType", "RecognitionQuality", "FaceQuality",
    "FaceEmbedding", "FaceMatch", "FaceComparisonResult", "FaceRecognitionResult",
    "RecognitionConfig", "EmbeddingVector", "FaceGallery",
    "get_model_from_string", "get_all_model_names"
]