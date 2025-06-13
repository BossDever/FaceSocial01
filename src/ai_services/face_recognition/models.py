# cSpell:disable
# mypy: ignore-errors
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


# Alias for backward compatibility
ModelType = RecognitionModel


class RecognitionQuality(Enum):
    """ระดับคุณภาพของใบหน้าสำหรับการจดจำ"""

    HIGH = "high"  # คุณภาพดี > 80%
    MEDIUM = "medium"  # คุณภาพปานกลาง 50-80%
    LOW = "low"  # คุณภาพต่ำ < 50%
    UNKNOWN = "unknown"  # ไม่สามารถประเมินได้


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
        if self.score >= 80:
            return RecognitionQuality.HIGH
        elif self.score >= 50:
            return RecognitionQuality.MEDIUM
        elif self.score > 0:
            return RecognitionQuality.LOW
        else:
            return RecognitionQuality.UNKNOWN

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
    """ผลลัพธ์การสกัด embedding vector จากรูปภาพใบหน้า - Enhanced"""

    # Main fields for new system
    id: str  # Unique ID for this specific embedding instance
    person_id: str  # ID of the person this embedding belongs to
    person_name: Optional[str] = None  # Name of the person
    vector: Optional[np.ndarray] = None
    # Model type used for this embedding
    model_type: Optional[RecognitionModel] = None
    timestamp: float = field(default_factory=time.time)
    quality_score: float = 0.0
    extraction_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Enhanced fields
    confidence: float = 0.0
    dimension: int = 0
    normalized: bool = False
    # Method used for extraction (e.g., 'insightface_onnx_adaface')
    extraction_method: str = ""
    source_image_hash: Optional[str] = None  # Hash of the source image
    # Bounding box of the face in the source image [x1, y1, x2, y2]
    face_bbox: Optional[List[int]] = None
    # Landmark points of the face
    landmarks: Optional[List[List[int]]] = None
    # Face quality assessment
    face_quality: RecognitionQuality = RecognitionQuality.UNKNOWN

    def __post_init__(self) -> None:
        if self.vector is not None:
            self.dimension = self.vector.shape[0]
            # Example: Check if vector is L2 normalized (sum of squares is close to 1)
            # This is a common practice for many face recognition models.
            # Adjust tolerance as needed.
            if self.dimension > 0:
                norm_sq = np.sum(np.square(self.vector))
                self.normalized = np.isclose(norm_sq, 1.0, atol=1e-5)

        if self.model_type and isinstance(self.model_type, RecognitionModel):
            # Ensure model_type is stored as its string value if it's an Enum
            # This was a previous source of issues, ensuring it's handled here.
            pass # No change needed if it's already RecognitionModel type
        elif self.model_type and isinstance(self.model_type, str):
            try:
                # Attempt to convert string back to Enum for consistency internally
                self.model_type = RecognitionModel(self.model_type)
            except ValueError:
                # If string is not valid enum, keep as string or handle.
                # For now, log warning & keep string if not valid enum member.
                # This can happen if new model type string not in enum.
                # Consider error or default if strict enum adherence needed.
                # print(f"Warn: '{self.model_type}' not valid enum member.")
                pass # Keep as string if not a valid enum member

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling numpy array serialization."""
        data = {
            "id": self.id,
            "person_id": self.person_id,
            "person_name": self.person_name,
            "vector": self.vector.tolist() if self.vector is not None else None,
            # Ensure model_type is stored as string value for JSON serialization
            "model_type": str(self.model_type) if self.model_type else None,
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
        """Create FaceEmbedding from dictionary."""
        vector = np.array(data["vector"]) if data.get("vector") is not None else None
        model_type_str = data.get("model_type")
        model_type_enum: Optional[RecognitionModel] = None
        if model_type_str:
            try:
                model_type_enum = RecognitionModel(model_type_str)
            except ValueError:
                # Log or handle if model_type_str not valid enum member
                # print(f"Warn: '{model_type_str}' from dict not valid enum.")
                pass # Keep as None or use a default if desired

        face_quality_str = data.get("face_quality", "unknown")
        face_quality_enum = RecognitionQuality.UNKNOWN
        try:
            face_quality_enum = RecognitionQuality(face_quality_str)
        except ValueError:
            pass

        return cls(
            id=data["id"],
            person_id=data["person_id"],
            person_name=data.get("person_name"),
            vector=vector,
            model_type=model_type_enum, # Use the converted enum value
            timestamp=data.get("timestamp", time.time()),
            quality_score=data.get("quality_score", 0.0),
            extraction_time=data.get("extraction_time", 0.0),
            metadata=data.get("metadata", {}),
            confidence=data.get("confidence", 0.0),
            # dimension will be set in __post_init__ if vector is present
            normalized=data.get("normalized", False),
            extraction_method=data.get("extraction_method", ""),
            source_image_hash=data.get("source_image_hash"),
            face_bbox=data.get("face_bbox"),
            landmarks=data.get("landmarks"),
            face_quality=face_quality_enum,
        )


@dataclass
class FaceMatch:
    """ผลการจับคู่ใบหน้ากับคนในฐานข้อมูล - Enhanced"""

    person_id: str
    confidence: float
    embedding: Optional[FaceEmbedding] = None

    # Enhanced fields
    similarity_score: float = 0.0
    distance: float = 0.0
    rank: int = 0
    match_quality: RecognitionQuality = RecognitionQuality.UNKNOWN
    comparison_method: str = "cosine_similarity"    # Legacy fields for backward compatibility
    identity_id: str = field(init=False)
    similarity: float = field(init=False)
    is_match: bool = field(init=False)
    identity_name: Optional[str] = None
    person_name: Optional[str] = None

    def __post_init__(self) -> None:
        # Set legacy fields for backward compatibility
        self.identity_id = self.person_id
        self.similarity = self.confidence
        self.is_match = self.confidence > 0.5  # Lowered from 0.6 to 0.5

        # Set similarity_score if not provided
        if self.similarity_score == 0.0:
            self.similarity_score = self.confidence

        # Set match quality based on confidence (adjusted thresholds)
        if self.confidence >= 0.75:  # Lowered from 0.8 to 0.75
            self.match_quality = RecognitionQuality.HIGH
        elif self.confidence >= 0.5:  # Lowered from 0.6 to 0.5
            self.match_quality = RecognitionQuality.MEDIUM
        elif self.confidence > 0.0:
            self.match_quality = RecognitionQuality.LOW
        else:
            self.match_quality = RecognitionQuality.UNKNOWN

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
    threshold_used: float = 0.6
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
    similarity_threshold: float = 0.60
    unknown_threshold: float = 0.55
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