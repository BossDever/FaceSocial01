# cSpell:disable
# mypy: ignore-errors
"""
Face Analysis Data Models
โครงสร้างข้อมูลสำหรับระบบวิเคราะห์ใบหน้าแบบครบวงจร
Enhanced with better error handling and validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from enum import Enum
from pydantic import BaseModel  # Add pydantic BaseModel

# Import related models
try:
    from src.ai_services.face_detection import models as face_detection_models
    from src.ai_services.face_recognition import models as face_recognition_models
    DetectionConfig = face_detection_models.DetectionConfig
    DetectionEngine = face_detection_models.DetectionEngine
    RecognitionModel = face_recognition_models.RecognitionModel
    RecognitionQuality = face_recognition_models.RecognitionQuality
except ImportError:
    # Fallback definitions
    class DetectionEngine(Enum):
        YOLOV9C = "yolov9c"
        YOLOV9E = "yolov9e"
        YOLOV11M = "yolov11m"
        AUTO = "auto"

    class RecognitionModel(Enum):
        ADAFACE = "adaface"
        ARCFACE = "arcface"
        FACENET = "facenet"

    class RecognitionQuality(Enum):
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
        UNKNOWN = "unknown"

    @dataclass
    class DetectionConfig:
        engine: DetectionEngine = DetectionEngine.AUTO
        confidence_threshold: float = 0.5


class AnalysisMode(Enum):
    """โหมดการวิเคราะห์"""

    DETECTION_ONLY = "detection_only"  # ตรวจจับใบหน้าเท่านั้น
    RECOGNITION_ONLY = "recognition_only"  # จดจำใบหน้าเท่านั้น (ต้องมี face crops)
    FULL_ANALYSIS = "full_analysis"  # ตรวจจับ + จดจำ
    COMPREHENSIVE = "comprehensive"  # วิเคราะห์ครบวงจรทุกอย่าง
    VERIFICATION = "verification"  # เปรียบเทียบใบหน้า 2 ใบ


class QualityLevel(Enum):
    """ระดับคุณภาพของการวิเคราะห์"""

    HIGH = "high"  # คุณภาพสูง - ใช้เวลานาน แต่แม่นยำ
    BALANCED = "balanced"  # สมดุล - เหมาะสำหรับงานทั่วไป
    FAST = "fast"  # เร็ว - ลดคุณภาพเพื่อความเร็ว


@dataclass
class AnalysisConfig:
    """การตั้งค่าการวิเคราะห์แบบครบวงจร"""

    mode: AnalysisMode = AnalysisMode.FULL_ANALYSIS

    # Detection settings
    detection_config: Optional[DetectionConfig] = None
    detection_model: Optional[str] = None  # auto-select if None
    min_face_size: int = 32
    confidence_threshold: float = 0.5
    max_faces: int = 50

    # Recognition settings
    recognition_config: Optional[Dict[str, Any]] = None
    recognition_model: Optional[str] = None  # auto-select if None
    enable_embedding_extraction: bool = True
    enable_gallery_matching: bool = True
    gallery_top_k: int = 5

    # Performance settings
    batch_size: int = 8
    use_quality_based_selection: bool = True
    parallel_processing: bool = True
    quality_level: QualityLevel = QualityLevel.BALANCED

    # Output settings
    return_face_crops: bool = False
    return_embeddings: bool = False
    return_detailed_stats: bool = True

    def __post_init__(self):
        """Post-initialization validation and defaults"""
        # Set default detection config if not provided
        if self.detection_config is None:
            self.detection_config = DetectionConfig(
                engine=DetectionEngine.AUTO,
                confidence_threshold=self.confidence_threshold,
            )        # Set default recognition config if not provided
        if self.recognition_config is None and self.mode in [
            AnalysisMode.FULL_ANALYSIS,
            AnalysisMode.COMPREHENSIVE,
        ]:
            self.recognition_config = {
                "model": RecognitionModel.FACENET,
                "threshold": 0.6,
                "return_embeddings": self.return_embeddings,
            }

        # Convert string enums to proper Enum objects
        if isinstance(self.mode, str):
            try:
                self.mode = AnalysisMode(self.mode)
            except ValueError:
                # Keep as string if not valid enum
                pass

        if isinstance(self.quality_level, str):
            try:
                self.quality_level = QualityLevel(self.quality_level)
            except ValueError:
                # Keep as string if not valid enum
                pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "mode": self.mode.value if hasattr(self.mode, 'value') else str(self.mode),
            "detection_model": self.detection_model,
            "recognition_model": self.recognition_model,
            "min_face_size": self.min_face_size,
            "confidence_threshold": self.confidence_threshold,
            "max_faces": self.max_faces,
            "gallery_top_k": self.gallery_top_k,
            "batch_size": self.batch_size,
            "use_quality_based_selection": self.use_quality_based_selection,
            "parallel_processing": self.parallel_processing,
            "quality_level": self.quality_level.value if hasattr(self.quality_level, 'value') else str(self.quality_level),
            "return_face_crops": self.return_face_crops,
            "return_embeddings": self.return_embeddings,
            "return_detailed_stats": self.return_detailed_stats,
        }


# Define the missing FaceAnalysisJSONRequest model
class FaceAnalysisJSONRequest(BaseModel):
    image_base64: str
    mode: Optional[AnalysisMode] = AnalysisMode.FULL_ANALYSIS
    config: Optional[AnalysisConfig] = None
    gallery: Optional[Dict[str, Any]] = None

    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True


@dataclass
class FaceResult:
    """ผลลัพธ์การวิเคราะห์ใบหน้า 1 ใบ (Detection + Recognition) - Enhanced"""

    # Detection results (required)
    bbox: Any  # BoundingBox from face_detection.models
    confidence: float
    quality_score: float

    # Recognition results (optional)
    embedding: Optional[Any] = None  # FaceEmbedding from face_recognition.models
    matches: Optional[List[Any]] = None  # List[FaceMatch] from face_recognition.models
    best_match: Optional[Any] = None  # FaceMatch from face_recognition.models

    # Additional data
    face_crop: Optional[np.ndarray] = None
    face_id: Optional[str] = None
    landmarks: Optional[np.ndarray] = None

    # Enhanced fields
    processing_time: float = 0.0
    model_used: str = ""
    quality_assessment: Optional[Dict[str, Any]] = None
    analysis_metadata: Optional[Dict[str, Any]] = None

    @property
    def has_identity(self) -> bool:
        """ตรวจสอบว่าจดจำตัวตนได้หรือไม่"""
        return self.best_match is not None and getattr(
            self.best_match, "is_match", False
        )

    @property
    def identity(self) -> Optional[str]:
        """ดึงตัวตนที่จดจำได้"""
        if self.has_identity and self.best_match:
            return getattr(
                self.best_match,
                "person_id",
                getattr(self.best_match, "identity_id", None),
            )
        return None

    @property
    def identity_name(self) -> Optional[str]:
        """ดึงชื่อตัวตนที่จดจำได้"""
        if self.has_identity and self.best_match:
            return getattr(self.best_match, "identity_name", self.identity)
        return None

    @property
    def recognition_confidence(self) -> float:
        """ความมั่นใจในการจดจำ"""
        if self.best_match:
            return getattr(self.best_match, "confidence", 0.0)
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary สำหรับ JSON serialization"""
        result = {
            "bbox": self.bbox.to_dict()
            if hasattr(self.bbox, "to_dict")
            else {
                "x1": float(self.bbox.x1),
                "y1": float(self.bbox.y1),
                "x2": float(self.bbox.x2),
                "y2": float(self.bbox.y2),
                "confidence": float(self.bbox.confidence),
            },
            "detection_confidence": float(self.confidence),
            "quality_score": float(self.quality_score),
            "has_identity": self.has_identity,
            "identity": self.identity,
            "identity_name": self.identity_name,
            "recognition_confidence": float(self.recognition_confidence),
            "face_id": self.face_id,
            "processing_time": float(self.processing_time),
            "model_used": self.model_used,
        }

        if self.embedding and hasattr(self.embedding, "to_dict"):
            result["embedding"] = self.embedding.to_dict()

        if self.matches:
            result["matches"] = [
                match.to_dict() if hasattr(match, "to_dict") else match
                for match in self.matches
            ]

        if self.best_match and hasattr(self.best_match, "to_dict"):
            result["best_match"] = self.best_match.to_dict()

        if self.face_crop is not None:
            result["face_crop_shape"] = self.face_crop.shape

        if self.landmarks is not None:
            result["landmarks_count"] = len(self.landmarks)

        if self.quality_assessment:
            result["quality_assessment"] = self.quality_assessment

        if self.analysis_metadata:
            result["analysis_metadata"] = self.analysis_metadata

        return result


@dataclass
class FaceAnalysisResult:
    """ผลลัพธ์การวิเคราะห์ใบหน้าทั้งหมดในรูป - Enhanced"""

    # Input info
    image_shape: Tuple[int, int, int]
    config: AnalysisConfig

    # Results
    faces: List[FaceResult]

    # Performance metrics
    detection_time: float
    recognition_time: float
    total_time: float
    processing_time: float = 0.0  # Alias for total_time

    # Models used
    detection_model_used: Optional[str] = None
    recognition_model_used: Optional[str] = None

    # Statistics (computed automatically)
    total_faces: int = 0
    usable_faces: int = 0
    identified_faces: int = 0

    # Enhanced fields
    analysis_metadata: Optional[Dict[str, Any]] = None
    performance_breakdown: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    success: bool = True

    def __post_init__(self):
        """คำนวณ statistics หลังจากสร้าง object"""
        self.total_faces = len(self.faces)
        self.usable_faces = len([f for f in self.faces if f.quality_score >= 60])
        self.identified_faces = len([f for f in self.faces if f.has_identity])

        # Set processing_time alias
        self.processing_time = self.total_time

        # Set success status
        self.success = self.error is None

        # Create performance breakdown
        self.performance_breakdown = {
            "detection_time": float(self.detection_time),
            "recognition_time": float(self.recognition_time),
            "total_time": float(self.total_time),
            "average_face_processing_time": float(
                self.total_time / max(self.total_faces, 1)
            ),
            "faces_per_second": float(
                max(self.total_faces, 1) / max(self.total_time, 0.001)
            ),
        }

    @property
    def detection_success_rate(self) -> float:
        """อัตราความสำเร็จของการตรวจจับ"""
        return (self.usable_faces / self.total_faces) if self.total_faces > 0 else 0.0

    @property
    def recognition_success_rate(self) -> float:
        """อัตราความสำเร็จของการจดจำ"""
        return (
            (self.identified_faces / self.usable_faces)
            if self.usable_faces > 0
            else 0.0
        )

    @property
    def average_confidence(self) -> float:
        """ความเชื่อมั่นเฉลี่ย (detection)"""
        if not self.faces:
            return 0.0
        return sum(face.confidence for face in self.faces) / len(self.faces)

    @property
    def average_quality(self) -> float:
        """คุณภาพเฉลี่ย"""
        if not self.faces:
            return 0.0
        return sum(face.quality_score for face in self.faces) / len(self.faces)

    @property
    def average_recognition_confidence(self) -> float:
        """ความเชื่อมั่นเฉลี่ยในการจดจำ"""
        identified_faces = [f for f in self.faces if f.has_identity]
        if not identified_faces:
            return 0.0
        return sum(face.recognition_confidence for face in identified_faces) / len(
            identified_faces
        )

    def get_identified_faces(self) -> List[FaceResult]:
        """ดึงเฉพาะใบหน้าที่จดจำตัวตนได้"""
        return [face for face in self.faces if face.has_identity]

    def get_face_by_identity(self, identity_id: str) -> Optional[FaceResult]:
        """ค้นหาใบหน้าตาม identity"""
        for face in self.faces:
            if face.identity == identity_id:
                return face
        return None

    def get_faces_by_quality(self, min_quality: float = 60.0) -> List[FaceResult]:
        """ดึงใบหน้าที่มีคุณภาพเหนือเกณฑ์"""
        return [face for face in self.faces if face.quality_score >= min_quality]

    def get_faces_by_confidence(self, min_confidence: float = 0.8) -> List[FaceResult]:
        """ดึงใบหน้าที่มีความมั่นใจในการจดจำสูง"""
        return [
            face for face in self.faces if face.recognition_confidence >= min_confidence
        ]

    def get_unique_identities(self) -> List[str]:
        """ดึงรายการ identity ที่ไม่ซ้ำกัน"""
        identities = set()
        for face in self.faces:
            if face.identity:
                identities.add(face.identity)
        return list(identities)

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary สำหรับ JSON serialization"""
        return {
            "image_shape": {
                "height": int(self.image_shape[0]),
                "width": int(self.image_shape[1]),
                "channels": int(self.image_shape[2])
                if len(self.image_shape) > 2
                else 1,
            },
            "config": self.config.to_dict(),
            "faces": [face.to_dict() for face in self.faces],
            "performance": self.performance_breakdown,
            "models_used": {
                "detection_model": self.detection_model_used,
                "recognition_model": self.recognition_model_used,
            },
            "statistics": {
                "total_faces": self.total_faces,
                "usable_faces": self.usable_faces,
                "identified_faces": self.identified_faces,
                "unique_identities": len(self.get_unique_identities()),
                "detection_success_rate": float(self.detection_success_rate),
                "recognition_success_rate": float(self.recognition_success_rate),
                "average_detection_confidence": float(self.average_confidence),
                "average_quality": float(self.average_quality),
                "average_recognition_confidence": float(
                    self.average_recognition_confidence
                ),
            },
            "success": bool(self.success),
            "error": self.error,
            "analysis_metadata": self.analysis_metadata,
        }


@dataclass
class BatchAnalysisResult:
    """ผลลัพธ์การวิเคราะห์หลายรูปพร้อมกัน - Enhanced"""

    results: List[FaceAnalysisResult]
    total_images: int
    total_faces: int
    total_identities: int
    processing_time: float

    # Enhanced fields
    successful_analyses: int = 0
    failed_analyses: int = 0
    batch_metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Calculate additional statistics"""
        self.successful_analyses = len([r for r in self.results if r.success])
        self.failed_analyses = len([r for r in self.results if not r.success])

    @property
    def average_faces_per_image(self) -> float:
        """จำนวนใบหน้าเฉลี่ยต่อรูป"""
        return self.total_faces / self.total_images if self.total_images > 0 else 0.0

    @property
    def overall_success_rate(self) -> float:
        """อัตราความสำเร็จโดยรวม"""
        return (
            self.successful_analyses / self.total_images
            if self.total_images > 0
            else 0.0
        )

    @property
    def overall_recognition_rate(self) -> float:
        """อัตราการจดจำโดยรวม"""
        if not self.results:
            return 0.0

        total_usable = sum(r.usable_faces for r in self.results)
        total_identified = sum(r.identified_faces for r in self.results)

        return total_identified / total_usable if total_usable > 0 else 0.0

    @property
    def average_processing_time(self) -> float:
        """เวลาประมวลผลเฉลี่ยต่อรูป"""
        return (
            self.processing_time / self.total_images if self.total_images > 0 else 0.0
        )

    @property
    def throughput_fps(self) -> float:
        """อัตราการประมวลผล (รูป/วินาที)"""
        return (
            self.total_images / self.processing_time
            if self.processing_time > 0
            else 0.0
        )

    def get_summary_statistics(self) -> Dict[str, Any]:
        """ดึงสถิติสรุปแบบละเอียด"""
        if not self.results:
            return {}

        # Detection statistics
        detection_times = [r.detection_time for r in self.results if r.success]
        recognition_times = [r.recognition_time for r in self.results if r.success]
        quality_scores = []

        for result in self.results:
            if result.success:
                quality_scores.extend([f.quality_score for f in result.faces])

        return {
            "detection_stats": {
                "average_time": float(np.mean(detection_times))
                if detection_times
                else 0.0,
                "min_time": float(np.min(detection_times)) if detection_times else 0.0,
                "max_time": float(np.max(detection_times)) if detection_times else 0.0,
            },
            "recognition_stats": {
                "average_time": float(np.mean(recognition_times))
                if recognition_times
                else 0.0,
                "min_time": float(np.min(recognition_times))
                if recognition_times
                else 0.0,
                "max_time": float(np.max(recognition_times))
                if recognition_times
                else 0.0,
            },
            "quality_stats": {
                "average_quality": float(np.mean(quality_scores))
                if quality_scores
                else 0.0,
                "min_quality": float(np.min(quality_scores)) if quality_scores else 0.0,
                "max_quality": float(np.max(quality_scores)) if quality_scores else 0.0,
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary สำหรับ JSON serialization"""
        return {
            "results": [result.to_dict() for result in self.results],
            "summary": {
                "total_images": self.total_images,
                "total_faces": self.total_faces,
                "total_identities": self.total_identities,
                "processing_time": float(self.processing_time),
                "successful_analyses": self.successful_analyses,
                "failed_analyses": self.failed_analyses,
                "average_faces_per_image": float(self.average_faces_per_image),
                "overall_success_rate": float(self.overall_success_rate),
                "overall_recognition_rate": float(self.overall_recognition_rate),
                "average_processing_time": float(self.average_processing_time),
                "throughput_fps": float(self.throughput_fps),
            },
            "detailed_statistics": self.get_summary_statistics(),
            "batch_metadata": self.batch_metadata,
        }
