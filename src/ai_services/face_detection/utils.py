"""
ฟังก์ชันช่วยเหลือสำหรับระบบตรวจจับใบหน้า
Enhanced version with better error handling and performance
"""

import cv2
import numpy as np
import os
import logging
from typing import List, Tuple, Optional, Dict, Any, Union, cast
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_QUALITY_CONFIG: Dict[str, Union[int, float, Tuple[int, int]]] = {
    "size_weight": 30,
    "area_weight": 25,
    "confidence_weight": 30,
    "aspect_weight": 15,
    "excellent_size": (80, 80),
    "good_size": (50, 50),
    "acceptable_size": (25, 25),
    "minimum_size": (10, 10),
    "bonus_score_for_high_confidence": 5.0,
    "high_confidence_threshold": 0.7,
}


def _get_quality_config_value(
    config: Optional[Dict[str, Any]], key: str, expected_type: type
) -> Any:
    """Helper to get a typed value from the quality configuration."""
    value = (config or {}).get(key, DEFAULT_QUALITY_CONFIG[key])
    if not isinstance(value, expected_type):
        logger.warning(
            f"Config value for {key} has unexpected type {type(value)}, "
            f"expected {expected_type}. Using default: {DEFAULT_QUALITY_CONFIG[key]}"
        )
        return DEFAULT_QUALITY_CONFIG[key]
    return value


def _calculate_size_score(
    face_width: float, face_height: float, config: Optional[Dict[str, Any]]
) -> int:
    """Calculates the size score for face quality."""
    size_thresholds = {
        "excellent": _get_quality_config_value(config, "excellent_size", tuple),
        "good": _get_quality_config_value(config, "good_size", tuple),
        "acceptable": _get_quality_config_value(config, "acceptable_size", tuple),
        "minimum": _get_quality_config_value(config, "minimum_size", tuple),
    }

    excellent_min_w, excellent_min_h = cast(
        Tuple[int, int], size_thresholds["excellent"]
    )
    good_min_w, good_min_h = cast(Tuple[int, int], size_thresholds["good"])
    acceptable_min_w, acceptable_min_h = cast(
        Tuple[int, int], size_thresholds["acceptable"]
    )
    minimum_min_w, minimum_min_h = cast(Tuple[int, int], size_thresholds["minimum"])

    if face_width >= excellent_min_w and face_height >= excellent_min_h:
        return 100
    elif face_width >= good_min_w and face_height >= good_min_h:
        return 85
    elif face_width >= acceptable_min_w and face_height >= acceptable_min_h:
        return 65
    elif face_width >= minimum_min_w and face_height >= minimum_min_h:
        return 45
    return 25


@dataclass
class BoundingBox:
    """คลาสสำหรับเก็บข้อมูลกรอบรอบใบหน้า - Enhanced"""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: Optional[int] = None

    @property
    def width(self) -> float:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1e-6)

    def to_array(self) -> np.ndarray:
        """แปลงเป็น numpy array"""
        if self.class_id is not None:
            return np.array(
                [self.x1, self.y1, self.x2, self.y2, self.confidence, self.class_id]
            )
        return np.array([self.x1, self.y1, self.x2, self.y2, self.confidence])

    @classmethod
    def from_array(cls, arr: Union[np.ndarray, "BoundingBox"]) -> "BoundingBox":
        """สร้างจาก numpy array หรือ BoundingBox object"""
        if isinstance(arr, BoundingBox):
            return arr

        if isinstance(arr, np.ndarray):
            if len(arr) == 5:
                return cls(
                    x1=float(arr[0]),
                    y1=float(arr[1]),
                    x2=float(arr[2]),
                    y2=float(arr[3]),
                    confidence=float(arr[4]),
                )
            elif len(arr) == 6:
                return cls(
                    x1=float(arr[0]),
                    y1=float(arr[1]),
                    x2=float(arr[2]),
                    y2=float(arr[3]),
                    confidence=float(arr[4]),
                    class_id=int(arr[5]),
                )
            else:
                raise ValueError(f"Array must have 5 or 6 elements, got {len(arr)}")

        raise TypeError(f"Expected numpy array or BoundingBox, got {type(arr)}")

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary"""
        return {
            "x1": float(self.x1),
            "y1": float(self.y1),
            "x2": float(self.x2),
            "y2": float(self.y2),
            "width": float(self.width),
            "height": float(self.height),
            "center_x": float(self.center[0]),
            "center_y": float(self.center[1]),
            "area": float(self.area),
            "aspect_ratio": float(self.aspect_ratio),
            "confidence": float(self.confidence),
            "class_id": self.class_id,
        }


@dataclass
class FaceDetection:
    """คลาสสำหรับเก็บข้อมูลใบหน้าที่ตรวจพบ - Enhanced"""

    bbox: BoundingBox
    quality_score: Optional[float] = None
    model_used: str = ""
    processing_time: float = 0.0
    landmarks: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary สำหรับ JSON"""
        result = {
            "bbox": self.bbox.to_dict(),
            "quality_score": float(self.quality_score)
            if self.quality_score is not None
            else None,
            "model_used": self.model_used,
            "processing_time": float(self.processing_time),
        }

        if self.landmarks is not None:
            result["landmarks"] = self.landmarks.tolist()

        if self.meta:
            result["meta"] = self.meta

        return result


@dataclass
class DetectionResult:
    """คลาสสำหรับเก็บผลลัพธ์การตรวจจับใบหน้าทั้งหมด - Enhanced"""

    faces: List[FaceDetection]
    image_shape: Tuple[int, int, int]
    total_processing_time: float
    model_used: str
    fallback_used: bool = False
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def num_faces(self) -> int:
        """จำนวนใบหน้าที่พบ"""
        return len(self.faces)

    @property
    def success(self) -> bool:
        """ตรวจสอบว่าการตรวจจับสำเร็จหรือไม่"""
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary สำหรับ JSON"""
        return {
            "faces": [face.to_dict() for face in self.faces],
            "image_shape": {
                "height": int(self.image_shape[0]),
                "width": int(self.image_shape[1]),
                "channels": int(self.image_shape[2])
                if len(self.image_shape) > 2
                else 1,
            },
            "total_processing_time": float(self.total_processing_time),
            "face_count": self.num_faces,
            "model_used": self.model_used,
            "fallback_used": self.fallback_used,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }


def _calculate_area_score(
    detection: BoundingBox, image_shape: Tuple[int, int]
) -> int:
    """Calculates the area score for face quality."""
    if len(image_shape) < 2:
        logger.warning(f"Invalid image_shape: {image_shape}")
        image_area = 1
    else:
        image_area = image_shape[0] * image_shape[1]

    face_area = detection.area
    area_ratio = min(face_area / max(image_area, 1e-6) * 100, 100)

    if area_ratio > 20:
        return 100
    elif area_ratio > 10:
        return 90
    elif area_ratio > 3:
        return 80
    elif area_ratio > 0.5:
        return 60
    return 40


def _calculate_aspect_score(detection: BoundingBox) -> int:
    """Calculates the aspect ratio score for face quality."""
    aspect_ratio = detection.aspect_ratio
    aspect_diff = abs(aspect_ratio - 0.8)  # Ideal aspect ratio around 0.8 (e.g., 4:5)

    if aspect_diff < 0.15:
        return 100
    elif aspect_diff < 0.3:
        return 85
    elif aspect_diff < 0.5:
        return 70
    elif aspect_diff < 0.8:
        return 55
    return 35


def calculate_face_quality(
    detection: BoundingBox,
    image_shape: Tuple[int, int],
    config: Optional[Dict[str, Any]] = None,
) -> float:
    """
    คำนวณคุณภาพของใบหน้า - Enhanced version

    Args:
        detection: BoundingBox ของใบหน้า
        image_shape: รูปร่างของภาพ (height, width)
        config: การตั้งค่าสำหรับการคำนวณคุณภาพ

    Returns:
        คะแนนคุณภาพ 0-100
    """
    try:
        merged_config = DEFAULT_QUALITY_CONFIG.copy()
        if config:
            merged_config.update(config)

        size_weight = _get_quality_config_value(merged_config, "size_weight", int)
        area_weight = _get_quality_config_value(merged_config, "area_weight", int)
        confidence_weight = _get_quality_config_value(
            merged_config, "confidence_weight", int
        )
        aspect_weight = _get_quality_config_value(merged_config, "aspect_weight", int)
        bonus_score = _get_quality_config_value(
            merged_config, "bonus_score_for_high_confidence", float
        )
        high_conf_threshold = _get_quality_config_value(
            merged_config, "high_confidence_threshold", float
        )

        size_score = _calculate_size_score(
            detection.width, detection.height, merged_config
        )
        area_score = _calculate_area_score(detection, image_shape)
        confidence_score = detection.confidence * 100
        aspect_score = _calculate_aspect_score(detection)

        final_score = (
            size_score * size_weight / 100
            + area_score * area_weight / 100
            + confidence_score * confidence_weight / 100
            + aspect_score * aspect_weight / 100
        )

        if detection.confidence >= high_conf_threshold:
            final_score += bonus_score

        final_score = min(final_score, 100.0)
        return float(max(0.0, final_score))

    except Exception as e:
        logger.error(f"Error calculating face quality: {e}")
        return 50.0  # Default score on error


def validate_bounding_box(
    bbox: BoundingBox, image_shape: Tuple[int, int], relaxed_validation: bool = True
) -> bool:
    """
    ตรวจสอบความถูกต้องของ bounding box - Enhanced version

    Args:
        bbox: BoundingBox object
        image_shape: รูปร่างของภาพ (height, width)
        relaxed_validation: ใช้การตรวจสอบแบบหลวมหรือไม่

    Returns:
        True if valid, False otherwise
    """
    try:
        if not _validate_bbox_input(bbox, image_shape):
            return False

        img_height, img_width = image_shape[:2]

        if not _validate_bbox_dimensions(bbox, relaxed_validation):
            return False

        # Validate boundaries
        if not _validate_bbox_boundaries(
            bbox, img_width, img_height, relaxed_validation
        ):
            return False

        # Validate area
        if not _validate_bbox_area(bbox, img_width, img_height, relaxed_validation):
            return False

        # Validate aspect ratio
        if not _validate_bbox_aspect_ratio(bbox, relaxed_validation):
            return False

        return True

    except Exception as e:
        logger.error(f"Bounding box validation failed: {e}")
        return False


def _validate_bbox_input(bbox: BoundingBox, image_shape: Tuple[int, int]) -> bool:
    """Helper to validate input parameters for bbox validation."""
    if not isinstance(bbox, BoundingBox):
        logger.warning(f"Invalid bbox type: {type(bbox)}")
        return False
    if len(image_shape) < 2:
        logger.warning(f"Invalid image_shape: {image_shape}")
        return False
    return True


def _validate_bbox_dimensions(bbox: BoundingBox, relaxed_validation: bool) -> bool:
    """Helper to validate dimensions of the bounding box."""
    if bbox.x1 < 0 or bbox.y1 < 0 or bbox.x2 < 0 or bbox.y2 < 0:
        return False
    if bbox.x2 <= bbox.x1 or bbox.y2 <= bbox.y1:
        return False

    min_size = 8 if relaxed_validation else 16
    if bbox.width < min_size or bbox.height < min_size:
        return False
    return True


def _validate_bbox_boundaries(
    bbox: BoundingBox, img_width: int, img_height: int, relaxed_validation: bool
) -> bool:
    """Helper to validate boundaries of the bounding box."""
    margin = 10 if relaxed_validation else 5
    if bbox.x2 > img_width + margin or bbox.y2 > img_height + margin:
        return False
    return True


def _validate_bbox_area(
    bbox: BoundingBox, img_width: int, img_height: int, relaxed_validation: bool
) -> bool:
    """Helper to validate area of the bounding box."""
    image_area = img_width * img_height
    if image_area == 0:
        return False

    area_ratio = bbox.area / image_area
    max_area_ratio = 0.98 if relaxed_validation else 0.90
    if area_ratio > max_area_ratio:
        return False
    return True


def _validate_bbox_aspect_ratio(bbox: BoundingBox, relaxed_validation: bool) -> bool:
    """Helper to validate aspect ratio of the bounding box."""
    if bbox.height == 0:
        return False

    aspect_ratio = bbox.width / bbox.height
    min_aspect = 0.1 if relaxed_validation else 0.2
    max_aspect = 15.0 if relaxed_validation else 10.0
    if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
        return False
    return True


def filter_detection_results(
    faces: List[FaceDetection],
    image_shape: Tuple[int, int],
    min_quality: float = 30.0,
    relaxed_filtering: bool = True,
) -> List[FaceDetection]:
    """
    กรองผลลัพธ์การตรวจจับตามคุณภาพและความถูกต้อง - Enhanced version

    Args:
        faces: รายการใบหน้าที่ตรวจพบ
        image_shape: รูปร่างของภาพ (height, width)
        min_quality: คะแนนคุณภาพขั้นต่ำ
        relaxed_filtering: ใช้การกรองแบบหลวมหรือไม่

    Returns:
        รายการใบหน้าที่ผ่านการกรอง
    """
    if not faces:
        return faces

    filtered_faces = []
    relaxed_count = 0

    for face in faces:
        try:
            # ตรวจสอบ bounding box
            if not validate_bounding_box(face.bbox, image_shape, relaxed_filtering):
                logger.debug("Face filtered: invalid bbox")
                continue

            # คำนวณคุณภาพใหม่ถ้าจำเป็น
            if face.quality_score is None or face.quality_score > 100:
                face.quality_score = calculate_face_quality(face.bbox, image_shape)

            # กรองตามคุณภาพ
            if face.quality_score >= min_quality:
                filtered_faces.append(face)
            elif relaxed_filtering and face.bbox.confidence > 0.7:
                # ถ้าคุณภาพต่ำแต่ confidence สูง ให้ผ่านได้
                logger.debug(
                    f"Low quality but high confidence face accepted: "
                    f"quality={face.quality_score:.1f}, conf={face.bbox.confidence:.3f}"
                )
                filtered_faces.append(face)
                relaxed_count += 1
            else:
                logger.debug(
                    f"Face filtered: quality {face.quality_score:.1f} < {min_quality}"
                )

        except Exception as e:
            logger.error(f"Error filtering face: {e}")
            if relaxed_filtering:
                # ในกรณีเกิดข้อผิดพลาด ให้ใส่ใบหน้านี้ไปด้วย
                logger.debug("Adding face despite filtering error (relaxed mode)")
                filtered_faces.append(face)
            continue

    if relaxed_count > 0:
        logger.info(f"Relaxed filtering allowed {relaxed_count} additional faces")

    logger.info(f"Filtered faces: {len(faces)} -> {len(filtered_faces)}")
    return filtered_faces


def draw_detection_results(
    image: np.ndarray,
    detections: List[FaceDetection],
    show_quality: bool = True,
    show_confidence: bool = True,
    show_model: bool = False,
) -> np.ndarray:
    """
    วาดกรอบรอบใบหน้าที่ตรวจพบลงบนรูปภาพ - Enhanced version

    Args:
        image: รูปภาพต้นฉบับ
        detections: รายการใบหน้าที่ตรวจพบ
        show_quality: แสดงคะแนนคุณภาพหรือไม่
        show_confidence: แสดงความมั่นใจหรือไม่
        show_model: แสดงชื่อโมเดลหรือไม่

    Returns:
        รูปภาพที่วาดกรอบแล้ว
    """
    img_draw = image.copy()

    for i, face in enumerate(detections):
        # สีตามคะแนนคุณภาพ
        if show_quality and face.quality_score is not None:
            if face.quality_score >= 80:
                color = (0, 255, 0)  # เขียว = คุณภาพดีมาก
            elif face.quality_score >= 60:
                color = (0, 255, 255)  # เหลือง = คุณภาพดี
            elif face.quality_score >= 40:
                color = (0, 165, 255)  # ส้ม = คุณภาพปานกลาง
            else:
                color = (0, 0, 255)  # แดง = คุณภาพต่ำ
        else:
            color = (0, 255, 0)  # เขียว (default)

        # วาดกรอบ
        x1, y1, x2, y2 = (
            int(face.bbox.x1),
            int(face.bbox.y1),
            int(face.bbox.x2),
            int(face.bbox.y2),
        )
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

        # แสดงข้อมูล
        y_offset = y1 - 10

        if show_confidence:
            conf_text = f"Conf: {face.bbox.confidence:.2f}"
            cv2.putText(
                img_draw,
                conf_text,
                (x1, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
            y_offset -= 20

        if show_quality and face.quality_score is not None:
            quality_text = f"Q: {face.quality_score:.0f}"
            cv2.putText(
                img_draw,
                quality_text,
                (x1, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
            y_offset -= 20

        if show_model and face.model_used:
            model_text = f"Model: {face.model_used}"
            cv2.putText(
                img_draw,
                model_text,
                (x1, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        # Face ID
        face_id = f"Face {i + 1}"
        cv2.putText(
            img_draw, face_id, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    return img_draw


def save_detection_image(
    image: np.ndarray,
    detections: List[FaceDetection],
    output_dir: str,
    filename: str,
    show_details: bool = True,
) -> str:
    """
    บันทึกรูปภาพที่มีการวาดกรอบใบหน้าแล้ว - Enhanced version

    Args:
        image: รูปภาพต้นฉบับ
        detections: รายการใบหน้าที่ตรวจพบ
        output_dir: โฟลเดอร์สำหรับบันทึกไฟล์
        filename: ชื่อไฟล์
        show_details: แสดงรายละเอียดหรือไม่

    Returns:
        พาธของไฟล์ที่บันทึก
    """
    try:
        # สร้างโฟลเดอร์ถ้ายังไม่มี
        os.makedirs(output_dir, exist_ok=True)

        # วาดกรอบใบหน้า
        img_with_detections = draw_detection_results(
            image,
            detections,
            show_quality=show_details,
            show_confidence=show_details,
            show_model=show_details,
        )

        # สร้างชื่อไฟล์
        file_path = os.path.join(output_dir, filename)

        # บันทึกไฟล์
        success = cv2.imwrite(file_path, img_with_detections)

        if success:
            logger.info(f"Detection image saved: {file_path}")
            return file_path
        else:
            raise Exception("Failed to write image file")

    except Exception as e:
        logger.error(f"Error saving detection image: {e}")
        raise


def get_relaxed_face_detection_config() -> Dict[str, Any]:
    """
    ให้การตั้งค่าแบบ relaxed สำหรับการตรวจจับใบหน้า
    """
    return {
        # General service settings
        "use_enhanced_detector": False,
        # Model paths
        "yolov9c_model_path": "model/face-detection/yolov9c-face-lindevs.onnx",
        "yolov9e_model_path": "model/face-detection/yolov9e-face-lindevs.onnx",
        "yolov11m_model_path": "model/face-detection/yolov11m-face.pt",
        # Decision criteria for model selection (relaxed)
        "max_usable_faces_yolov9": 12,
        "min_agreement_ratio": 0.5,
        "min_quality_threshold": 40,
        "iou_threshold_agreement": 0.3,
        # Detection parameters (relaxed)
        "conf_threshold": 0.10,
        "iou_threshold_nms": 0.35,
        "img_size": 640,
        # Quality configuration (relaxed)
        "quality_config": {
            "min_quality_threshold": 40,
            "size_weight": 30,
            "area_weight": 25,
            "confidence_weight": 30,
            "aspect_weight": 15,
            "excellent_size": (80, 80),
            "good_size": (50, 50),
            "acceptable_size": (24, 24),
            "minimum_size": (8, 8),
            "bonus_score_for_high_confidence": 5.0,
            "high_confidence_threshold": 0.7,
        },
        # Fallback strategy configuration
        "fallback_config": {
            "enable_fallback_system": True,
            "max_fallback_attempts": 3,
            "fallback_models": [
                {
                    "model_name": "yolov11m",
                    "conf_threshold": 0.15,
                    "iou_threshold": 0.35,
                    "min_faces_to_accept": 1,
                },
                {
                    "model_name": "yolov9c",
                    "conf_threshold": 0.05,
                    "iou_threshold": 0.3,
                    "min_faces_to_accept": 1,
                },
                {
                    "model_name": "opencv_haar",
                    "scale_factor": 1.1,
                    "min_neighbors": 3,
                    "min_size": (20, 20),
                    "min_faces_to_accept": 1,
                },
            ],
            "min_detections_after_fallback": 1,
            "always_run_all_fallbacks_if_zero_initial": True,
        },
        # Filter settings
        "filter_min_quality": 30.0,
        "filter_min_quality_final": 40.0,
    }