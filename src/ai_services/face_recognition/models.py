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


class RecognitionModel(Enum):
    """โมเดลสำหรับการจดจำใบหน้า"""

    ADAFACE = "adaface"
    ARCFACE = "arcface"
    FACENET = "facenet"


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

    vector: Optional[np.ndarray] = None
    model_type: Optional[RecognitionModel] = None
    quality_score: float = 0.0
    extraction_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    # Enhanced fields
    confidence: float = 0.0
    dimension: int = 0
    normalized: bool = False
    extraction_method: str = ""

    # Legacy fields for backward compatibility
    embedding: Optional[np.ndarray] = field(init=False)
    model_used: Optional[str] = field(init=False)
    success: bool = True
    error: Optional[str] = None
    processing_time: Optional[float] = field(init=False)
    face_quality: RecognitionQuality = RecognitionQuality.UNKNOWN

    def __post_init__(self):
        # Set legacy fields for backward compatibility
        self.embedding = self.vector
        self.model_used = self.model_type.value if self.model_type else None
        self.processing_time = self.extraction_time

        # Set dimension
        if self.vector is not None:
            self.dimension = len(self.vector.flatten())

        # Validate embedding
        if self.vector is not None:
            try:
                # Check for NaN or Inf values
                if np.any(np.isnan(self.vector)) or np.any(np.isinf(self.vector)):
                    self.success = False
                    self.error = "Invalid embedding values (NaN/Inf detected)"

                # Check if normalized
                norm = np.linalg.norm(self.vector)
                self.normalized = abs(norm - 1.0) < 1e-5

            except Exception as e:
                self.success = False
                self.error = f"Embedding validation failed: {e}"

    def normalize(self) -> "FaceEmbedding":
        """L2 normalize the embedding vector"""
        if self.vector is not None:
            try:
                norm = np.linalg.norm(self.vector)
                if norm > 1e-8:
                    self.vector = self.vector / norm
                    self.embedding = self.vector  # Update legacy field
                    self.normalized = True
            except Exception as e:
                self.error = f"Normalization failed: {e}"
        return self

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary สำหรับ JSON serialization"""
        result = {
            "model_type": self.model_type.value if self.model_type else None,
            "quality_score": float(self.quality_score),
            "extraction_time": float(self.extraction_time),
            "confidence": float(self.confidence),
            "dimension": int(self.dimension),
            "normalized": bool(self.normalized),
            "extraction_method": self.extraction_method,
            "success": bool(self.success),
            "error": self.error,
            "face_quality": self.face_quality.value,
        }

        # Add vector if requested (usually not for API responses due to size)
        if self.vector is not None:
            result["vector_shape"] = self.vector.shape
            result["vector_norm"] = float(np.linalg.norm(self.vector))

        return result


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
    comparison_method: str = "cosine_similarity"

    # Legacy fields for backward compatibility
    identity_id: str = field(init=False)
    similarity: float = field(init=False)
    is_match: bool = field(init=False)

    def __post_init__(self):
        # Set legacy fields for backward compatibility
        self.identity_id = self.person_id
        self.similarity = self.confidence
        self.is_match = self.confidence > 0.6  # Default threshold

        # Set similarity_score if not provided
        if self.similarity_score == 0.0:
            self.similarity_score = self.confidence

        # Set match quality based on confidence
        if self.confidence >= 0.8:
            self.match_quality = RecognitionQuality.HIGH
        elif self.confidence >= 0.6:
            self.match_quality = RecognitionQuality.MEDIUM
        elif self.confidence > 0.0:
            self.match_quality = RecognitionQuality.LOW
        else:
            self.match_quality = RecognitionQuality.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary สำหรับ JSON serialization"""
        return {
            "person_id": self.person_id,
            "confidence": float(self.confidence),
            "similarity_score": float(self.similarity_score),
            "distance": float(self.distance),
            "rank": int(self.rank),
            "match_quality": self.match_quality.value,
            "comparison_method": self.comparison_method,
            "is_match": bool(self.is_match),
            # Legacy fields
            "identity_id": self.identity_id,
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

    def __post_init__(self):
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

    def __post_init__(self):
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
class ModelPerformanceStats:
    """สถิติประสิทธิภาพของโมเดล Face Recognition - Enhanced"""

    total_embeddings_extracted: int = 0
    total_comparisons: int = 0
    total_extraction_time: float = 0.0
    total_comparison_time: float = 0.0
    average_extraction_time: float = 0.0
    average_comparison_time: float = 0.0

    # Enhanced metrics
    successful_extractions: int = 0
    failed_extractions: int = 0
    high_quality_embeddings: int = 0
    gpu_usage_time: float = 0.0
    cpu_usage_time: float = 0.0
    memory_peak_usage: float = 0.0

    def update_extraction_stats(
        self,
        time_taken: float,
        success: bool = True,
        quality: RecognitionQuality = RecognitionQuality.UNKNOWN,
    ) -> None:
        """อัปเดตสถิติการสกัด embedding"""
        self.total_embeddings_extracted += 1
        self.total_extraction_time += time_taken
        self.average_extraction_time = (
            self.total_extraction_time / self.total_embeddings_extracted
        )

        if success:
            self.successful_extractions += 1
        else:
            self.failed_extractions += 1

        if quality == RecognitionQuality.HIGH:
            self.high_quality_embeddings += 1

    def update_comparison_stats(self, time_taken: float) -> None:
        """อัปเดตสถิติการเปรียบเทียบใบหน้า"""
        self.total_comparisons += 1
        self.total_comparison_time += time_taken
        self.average_comparison_time = (
            self.total_comparison_time / self.total_comparisons
        )

    def update_device_usage(self, time_taken: float, device: str) -> None:
        """อัปเดตสถิติการใช้งานอุปกรณ์"""
        if device.lower() == "cuda" or device.lower() == "gpu":
            self.gpu_usage_time += time_taken
        else:
            self.cpu_usage_time += time_taken

    @property
    def success_rate(self) -> float:
        """อัตราความสำเร็จของการสกัด embedding"""
        total = self.total_embeddings_extracted
        return (self.successful_extractions / total) if total > 0 else 0.0

    @property
    def high_quality_rate(self) -> float:
        """อัตรา embedding คุณภาพสูง"""
        total = self.successful_extractions
        return (self.high_quality_embeddings / total) if total > 0 else 0.0

    @property
    def gpu_usage_ratio(self) -> float:
        """สัดส่วนการใช้ GPU"""
        total_time = self.gpu_usage_time + self.cpu_usage_time
        return (self.gpu_usage_time / total_time) if total_time > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary สำหรับ JSON serialization"""
        return {
            "total_embeddings_extracted": self.total_embeddings_extracted,
            "total_comparisons": self.total_comparisons,
            "total_extraction_time": float(self.total_extraction_time),
            "total_comparison_time": float(self.total_comparison_time),
            "average_extraction_time": float(self.average_extraction_time),
            "average_comparison_time": float(self.average_comparison_time),
            "successful_extractions": self.successful_extractions,
            "failed_extractions": self.failed_extractions,
            "high_quality_embeddings": self.high_quality_embeddings,
            "success_rate": float(self.success_rate),
            "high_quality_rate": float(self.high_quality_rate),
            "gpu_usage_time": float(self.gpu_usage_time),
            "cpu_usage_time": float(self.cpu_usage_time),
            "gpu_usage_ratio": float(self.gpu_usage_ratio),
            "memory_peak_usage": float(self.memory_peak_usage),
        }


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
