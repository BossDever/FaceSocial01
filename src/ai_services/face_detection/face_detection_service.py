# cSpell:disable
# mypy: ignore-errors
"""
Enhanced Face Detection Service
ระบบตรวจจับใบหน้าอัจฉริยะที่รองรับโมเดล YOLOv9c, YOLOv9e และ YOLOv11m
พร้อมระบบ Fallback และการจัดการ GPU แบบอัจฉริยะ
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

from .yolo_models import YOLOv9ONNXDetector, YOLOv11Detector, fallback_opencv_detection
from .utils import (
    BoundingBox,
    FaceDetection,
    DetectionResult,
    calculate_face_quality,
    validate_bounding_box,
    get_relaxed_face_detection_config,
)

logger = logging.getLogger(__name__)


class QualityCategory(Enum):
    """ระดับคุณภาพของใบหน้า"""

    EXCELLENT = "excellent"  # 80-100
    GOOD = "good"  # 70-79
    ACCEPTABLE = "acceptable"  # 40-69
    POOR = "poor"  # < 40


@dataclass
class FaceQualityAnalyzer:
    """ระบบวิเคราะห์คุณภาพใบหน้า - Enhanced"""

    config: Dict[str, Any]

    def __post_init__(self):
        self.quality_weights = {
            "size_weight": self.config.get("size_weight", 30),
            "area_weight": self.config.get("area_weight", 25),
            "confidence_weight": self.config.get("confidence_weight", 30),
            "aspect_weight": self.config.get("aspect_weight", 15),
        }

        self.size_thresholds = {
            "excellent": self.config.get("excellent_size", (80, 80)),
            "good": self.config.get("good_size", (50, 50)),
            "acceptable": self.config.get("acceptable_size", (24, 24)),
            "minimum": self.config.get("minimum_size", (8, 8)),
        }

        self.min_quality_threshold = self.config.get("min_quality_threshold", 40)
        self.bonus_score = self.config.get("bonus_score_for_high_confidence", 5.0)
        self.high_confidence_threshold = self.config.get(
            "high_confidence_threshold", 0.7
        )

    def get_quality_category(self, score: float) -> QualityCategory:
        """ระบุระดับคุณภาพตามคะแนน"""
        if score >= 80:
            return QualityCategory.EXCELLENT
        elif score >= 70:
            return QualityCategory.GOOD
        elif score >= self.min_quality_threshold:
            return QualityCategory.ACCEPTABLE
        else:
            return QualityCategory.POOR

    def is_face_usable(self, face: FaceDetection) -> bool:
        """ตรวจสอบว่าใบหน้าใช้งานได้หรือไม่"""
        if face.quality_score is None:
            return False
        return face.quality_score >= self.min_quality_threshold

    def analyze_detection_quality(self, faces: List[FaceDetection]) -> Dict[str, Any]:
        """วิเคราะห์คุณภาพการตรวจจับทั้งหมด"""
        if not faces:
            return {
                "total_count": 0,
                "usable_count": 0,
                "quality_ratio": 0.0,
                "quality_categories": {cat.value: 0 for cat in QualityCategory},
                "avg_quality": 0.0,
            }

        quality_categories = {cat.value: 0 for cat in QualityCategory}
        usable_count = 0
        quality_scores = []

        for face in faces:
            if face.quality_score is not None:
                quality_scores.append(face.quality_score)
                category = self.get_quality_category(face.quality_score)
                quality_categories[category.value] += 1

                if self.is_face_usable(face):
                    usable_count += 1

        total_count = len(faces)
        quality_ratio = (usable_count / total_count) * 100 if total_count > 0 else 0
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        return {
            "total_count": total_count,
            "usable_count": usable_count,
            "quality_ratio": quality_ratio,
            "quality_categories": quality_categories,
            "avg_quality": avg_quality,
        }


@dataclass
class DetectionDecision:
    """ผลลัพธ์การตัดสินใจเลือกโมเดล"""

    def __init__(self):
        # Step 1: YOLOv9 testing results
        self.yolov9c_detections = []
        self.yolov9e_detections = []
        self.yolov9c_time = 0.0
        self.yolov9e_time = 0.0

        # Step 2: Agreement analysis
        self.agreement = False
        self.agreement_ratio = 0.0
        self.agreement_type = ""

        # Step 3: Decision
        self.use_yolov11m = False
        self.decision_reasons = []

        # Step 4: Final results
        self.final_detections = []
        self.final_model = ""
        self.final_time = 0.0

        # Quality info
        self.quality_info = {}

        # Total processing time
        self.total_time = 0.0

        # Fallback information
        self.fallback_attempts_info = []
        self.fallback_used = False

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary สำหรับ JSON"""
        return {
            "step1_results": {
                "yolov9c": {
                    "count": len(self.yolov9c_detections),
                    "time": self.yolov9c_time,
                },
                "yolov9e": {
                    "count": len(self.yolov9e_detections),
                    "time": self.yolov9e_time,
                },
            },
            "step2_agreement": {
                "agreement": self.agreement,
                "ratio": self.agreement_ratio,
                "type": self.agreement_type,
            },
            "step3_decision": {
                "use_yolov11m": self.use_yolov11m,
                "reasons": self.decision_reasons,
            },
            "step4_results": {
                "model_used": self.final_model,
                "count": len(self.final_detections),
                "time": self.final_time,
            },
            "quality_info": self.quality_info,
            "total_time": self.total_time,
            "fallback_info": {
                "fallback_used": self.fallback_used,
                "attempts": self.fallback_attempts_info,
            },
        }


class FaceDetectionService:
    """
    Enhanced Face Detection Service
    รองรับโมเดล YOLOv9c, YOLOv9e และ YOLOv11m พร้อมระบบอัจฉริยะ
    """

    def __init__(self, vram_manager, config: Optional[Dict[str, Any]] = None):
        self.vram_manager = vram_manager
        self.logger = logging.getLogger(__name__)

        # Use relaxed config if none provided
        self.config = (
            config if config is not None else get_relaxed_face_detection_config()
        )

        # Model management
        self.models: Dict[str, Union[YOLOv9ONNXDetector, YOLOv11Detector]] = {}
        self.model_stats: Dict[str, Dict[str, Union[float, int]]] = {}

        # Configuration
        self.use_enhanced_detector = self.config.get("use_enhanced_detector", False)

        # Decision criteria
        self.decision_criteria = {
            "max_usable_faces_yolov9": int(
                self.config.get("max_usable_faces_yolov9", 12)
            ),
            "min_agreement_ratio": float(self.config.get("min_agreement_ratio", 0.5)),
            "min_quality_threshold": int(self.config.get("min_quality_threshold", 40)),
            "iou_threshold": float(self.config.get("iou_threshold_agreement", 0.3)),
        }

        # Detection parameters
        self.detection_params = {
            "conf_threshold": self.config.get("conf_threshold", 0.10),
            "iou_threshold": self.config.get("iou_threshold_nms", 0.35),
            "img_size": self.config.get("img_size", 640),
        }

        # Quality analyzer
        quality_config = self.config.get("quality_config", {})
        quality_config.setdefault(
            "min_quality_threshold", self.decision_criteria["min_quality_threshold"]
        )
        self.quality_analyzer = FaceQualityAnalyzer(quality_config)

        # Model paths
        self.yolov9c_model_path = self.config.get(
            "yolov9c_model_path", "model/face-detection/yolov9c-face-lindevs.onnx"
        )
        self.yolov9e_model_path = self.config.get(
            "yolov9e_model_path", "model/face-detection/yolov9e-face-lindevs.onnx"
        )
        self.yolov11m_model_path = self.config.get(
            "yolov11m_model_path", "model/face-detection/yolov11m-face.pt"
        )

        # Fallback configuration
        self.fallback_config = self.config.get(
            "fallback_config", get_relaxed_face_detection_config()["fallback_config"]
        )

        # Statistics
        self.decision_log = []
        self.models_loaded = False

        # Performance tracking
        self.performance_stats = {
            "total_detections": 0,
            "successful_detections": 0,
            "fallback_used_count": 0,
            "average_processing_time": 0.0,
            "model_usage_count": {
                "yolov9c": 0,
                "yolov9e": 0,
                "yolov11m": 0,
                "opencv_haar": 0,
            },
        }

    async def initialize(self) -> bool:
        """โหลดโมเดลตรวจจับใบหน้าทั้งหมด"""
        try:
            logger.info("Loading face detection models (Enhanced/Relaxed Mode)...")

            # Request VRAM allocations
            yolov9c_allocation = await self.vram_manager.request_model_allocation(
                "yolov9c-face", "high", "face_detection_service"
            )
            yolov9e_allocation = await self.vram_manager.request_model_allocation(
                "yolov9e-face", "high", "face_detection_service"
            )
            yolov11m_allocation = await self.vram_manager.request_model_allocation(
                "yolov11m-face", "critical", "face_detection_service"
            )

            # Load YOLOv9c
            self.models["yolov9c"] = YOLOv9ONNXDetector(
                self.yolov9c_model_path, "YOLOv9c"
            )
            device_yolov9c = (
                "cuda" if yolov9c_allocation.location.value == "gpu" else "cpu"
            )
            if not self.models["yolov9c"].load_model(device_yolov9c):
                logger.error("Failed to load YOLOv9c model")

            # Load YOLOv9e
            self.models["yolov9e"] = YOLOv9ONNXDetector(
                self.yolov9e_model_path, "YOLOv9e"
            )
            device_yolov9e = (
                "cuda" if yolov9e_allocation.location.value == "gpu" else "cpu"
            )
            if not self.models["yolov9e"].load_model(device_yolov9e):
                logger.error("Failed to load YOLOv9e model")

            # Load YOLOv11m
            self.models["yolov11m"] = YOLOv11Detector(
                self.yolov11m_model_path, "YOLOv11m"
            )
            device_yolov11m = (
                "cuda" if yolov11m_allocation.location.value == "gpu" else "cpu"
            )
            if not self.models["yolov11m"].load_model(device_yolov11m):
                logger.error("Failed to load YOLOv11m model")

            # Check if at least one model loaded successfully
            loaded_models = [
                name for name, model in self.models.items() if model.model_loaded
            ]

            if not loaded_models:
                logger.error("No models loaded successfully")
                return False

            self.models_loaded = True
            logger.info(
                f"✅ Face Detection Service initialized with models: {loaded_models}"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading models: {e}", exc_info=True)
            return False

    async def detect_faces(
        self,
        image_input: Union[str, np.ndarray],
        model_name: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        min_face_size: Optional[int] = None,  # Not directly used
        max_faces: Optional[int] = None,
        return_landmarks: bool = False,  # Not directly used
        min_quality_threshold: Optional[float] = None,
        use_fallback: bool = True,
        fallback_strategy: Optional[List[Dict[str, Any]]] = None,
        force_cpu: bool = False,  # Not directly used
    ) -> DetectionResult:
        """
        Detect faces in an image, automatically selecting the best model.
        Includes an improved fallback system. (Refactored for C901)
        """
        start_time_total = time.time()

        # Step 1: Ensure models are loaded
        error_result = await self._ensure_models_loaded(start_time_total)
        if error_result:
            return error_result

        # Step 2: Validate image input
        # image_np is np.ndarray if validation_error is None
        image_np, validation_error = self._validate_image_input(
            image_input, start_time_total
        )
        if validation_error:
            return validation_error
        # image_np is now a valid np.ndarray

        # Step 3: Setup detection parameters
        primary_model_name, current_conf, current_iou, current_min_quality = (
            self._setup_detection_parameters(
                model_name, conf_threshold, iou_threshold, min_quality_threshold
            )
        )

        # Step 4: Attempt primary detection
        # _attempt_primary_detection returns:
        # (faces: List[FaceDetection], model_used: str, error: Optional[str])
        primary_faces, primary_model_used, primary_error = (
            await self._attempt_primary_detection(
                image_np,
                primary_model_name,
                current_conf,
                current_iou,
                current_min_quality,
            )
        )

        final_faces = primary_faces
        final_model_used = primary_model_used
        final_error = primary_error
        fallback_was_used = False

        # Step 5: Attempt fallback detection if necessary
        # Condition: fallback enabled AND (primary found no faces OR primary had error)
        primary_detection_failed_or_empty = not primary_faces or primary_error
        if use_fallback and primary_detection_failed_or_empty:
            self.logger.info(
                "Primary detection: no faces or error. Initiating fallback."
            )
            # _attempt_fallback_detection returns: (detected_faces, model_used,
            # fallback_triggered, error_message)
            fb_faces, fb_model_used, fb_triggered, fb_error = (
                await self._attempt_fallback_detection(
                    image_np,
                    primary_faces,  # Pass original primary faces
                    primary_error,  # Pass original primary error
                    fallback_strategy,
                    current_min_quality,
                )
            )

            if fb_triggered:  # Fallback mechanism was engaged
                fallback_was_used = True
                # Fallback results take precedence.
                # _attempt_fallback_detection returns best faces (new or primary).
                final_faces = fb_faces
                final_model_used = fb_model_used  # Model for fb_faces
                final_error = fb_error  # Error state after fallbacks

                # Clear error if fallback succeeded
                if fb_faces and not fb_error:
                    final_error = None

        # Step 6: Apply max_faces limit
        if max_faces is not None and len(final_faces) > max_faces:
            # Sort by quality score descending
            final_faces.sort(key=lambda f: f.quality_score or 0.0, reverse=True)
            final_faces = final_faces[:max_faces]

        # Step 7: Update performance statistics
        total_service_time = time.time() - start_time_total
        self._update_performance_stats(
            final_faces,
            final_model_used,  # Model that produced final_faces or last attempted
            total_service_time,
            fallback_was_used,
        )

        # Step 8: Return DetectionResult
        return DetectionResult(
            faces=final_faces,
            image_shape=image_np.shape,
            total_processing_time=total_service_time * 1000,  # ms
            model_used=final_model_used,
            fallback_used=fallback_was_used,
            error=final_error,
            metadata={
                "config_used": "relaxed",
                "quality_threshold_applied": current_min_quality,
                "conf_threshold_requested": current_conf,
                "iou_threshold_requested": current_iou,
            },
        )

    async def _ensure_models_loaded(
        self, start_time_total: float
    ) -> Optional[DetectionResult]:
        if not self.models_loaded:
            self.logger.warning("Models not loaded. Attempting to initialize...")
            initialized = await self.initialize()
            if not initialized:
                return DetectionResult(
                    faces=[],
                    image_shape=(0, 0, 0),
                    total_processing_time=time.time() - start_time_total,
                    model_used="N/A",
                    error="Models not loaded and initialization failed.",
                )
        return None

    def _validate_image_input(
        self, image_input: Union[str, np.ndarray], start_time_total: float
    ) -> Tuple[Optional[np.ndarray], Optional[DetectionResult]]:
        try:
            image = self._process_image_input(image_input)
            if image is None or image.size == 0:
                return None, DetectionResult(
                    faces=[],
                    image_shape=(0, 0, 0),
                    total_processing_time=time.time() - start_time_total,
                    model_used="N/A",
                    error="Invalid image input",
                )
            return image, None
        except Exception as e:
            return None, DetectionResult(
                faces=[],
                image_shape=(0, 0, 0),
                total_processing_time=time.time() - start_time_total,
                model_used="N/A",
                error=f"Error processing image: {e}",
            )

    def _setup_detection_parameters(
        self,
        model_name: Optional[str],
        conf_threshold: Optional[float],
        iou_threshold: Optional[float],
        min_quality_threshold: Optional[float],
    ) -> Tuple[str, float, float, float]:
        primary_model = model_name if model_name and model_name != "auto" else "yolov9c"
        current_conf = (
            conf_threshold
            if conf_threshold is not None
            else self.detection_params["conf_threshold"]
        )
        current_iou = (
            iou_threshold
            if iou_threshold is not None
            else self.detection_params["iou_threshold"]
        )
        current_min_quality = (
            min_quality_threshold
            if min_quality_threshold is not None
            else self.config.get("filter_min_quality_final", 40.0)
        )
        self.logger.info(
            f"Starting detection: model={primary_model}, "
            f"conf={current_conf}, iou={current_iou}"
        )
        return primary_model, current_conf, current_iou, current_min_quality

    async def _attempt_primary_detection(
        self,
        image: np.ndarray,
        primary_model: str,
        current_conf: float,
        current_iou: float,
        current_min_quality: float,
    ) -> Tuple[List[FaceDetection], str, Optional[str]]:  # Changed signature
        processed_faces: List[FaceDetection] = []
        model_used = "N/A"  # Default if model isn't found or loaded
        error_msg = None
        detection_time_ms = 0.0

        try:
            if primary_model not in self.models:
                error_msg = f"Primary model {primary_model} not found"
                self.logger.error(error_msg)
                return processed_faces, "N/A", error_msg

            detector = self.models[primary_model]
            if not detector.model_loaded:
                error_msg = f"Primary model {primary_model} not loaded"
                self.logger.error(error_msg)
                # Model used is primary_model, but it's not loaded.
                return processed_faces, primary_model, error_msg

            # Model found and loaded, this is the one used for this attempt.
            model_used = primary_model

            start_detect_time = time.time()
            raw_bboxes = await self._run_detection(
                detector, image, current_conf, current_iou
            )
            detection_time_ms = (time.time() - start_detect_time) * 1000

            processed_faces = self._process_raw_detections(
                raw_bboxes,
                image,
                primary_model,  # Model that performed detection
                detection_time_ms,
                current_min_quality,
            )
            self.logger.info(
                f"Primary detection ({primary_model}): "
                f"{len(processed_faces)} valid faces"
            )
            # Early exit logic removed; decision in detect_faces

        except Exception as e:
            error_msg = f"Error in primary detection ({primary_model}): {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            # Ensure model_used is correct if error occurred after model selection
            if primary_model in self.models:
                model_used = primary_model

        return processed_faces, model_used, error_msg

    async def _attempt_fallback_detection(
        self,
        image: np.ndarray,
        detected_faces_primary: List[FaceDetection],
        error_primary: Optional[str],
        fallback_strategy_override: Optional[List[Dict[str, Any]]],
        current_min_quality: float,
    ) -> Tuple[List[FaceDetection], str, bool, Optional[str]]:
        detected_faces_final = list(
            detected_faces_primary
        )  # Start with primary results
        model_used_for_detection = "N/A"
        # If primary detection yielded some results, assume that model was used
        if detected_faces_primary:
            model_used_for_detection = (
                detected_faces_primary[0].model_used
                if detected_faces_primary[0].model_used
                else "N/A"
            )

        fallback_actually_used = False
        final_error_message = error_primary

        if not detected_faces_final or error_primary:  # Condition to trigger fallback
            self.logger.info("Initiating fallback detection system")
            fallback_actually_used = True
            final_error_message = None  # Reset error message if fallback is attempted

            fallback_strategy = fallback_strategy_override or self.fallback_config.get(
                "fallback_models", []
            )

            for fb_attempt, fb_config in enumerate(fallback_strategy):
                # Stop if primary found faces and had no error
                if detected_faces_final and not error_primary:
                    break

                try:
                    fb_faces = await self._run_fallback_detection(
                        image, fb_config, current_min_quality, fb_attempt
                    )
                    if fb_faces:
                        detected_faces_final = fb_faces
                        model_used_for_detection = fb_config.get(
                            "model_name", "unknown"
                        )
                        final_error_message = None  # Clear error if fallback succeeds
                        log_msg = (
                            f"Fallback successful with {model_used_for_detection}"
                        )
                        self.logger.info(log_msg)
                        break  # Exit fallback loop on success
                except Exception as e:
                    self.logger.error(
                        f"Fallback attempt {fb_attempt + 1} "
                        f"({fb_config.get('model_name', 'unknown')}) failed: {e}"
                    )
                    # Keep the first error encountered during fallback
                    if not final_error_message:
                        final_error_message = f"Fallback failed: {e}"
                    continue

            if not detected_faces_final and not final_error_message:
                final_error_message = (
                    "All detection attempts (including fallbacks) "
                    "failed to find faces."
                )
            elif not detected_faces_final and final_error_message and error_primary:
                # If primary had an error and fallbacks also failed or had errors
                final_error_message = (
                    f"Primary error: {error_primary}. "
                    f"Fallback error: {final_error_message}"
                )

        return (
            detected_faces_final,
            model_used_for_detection,
            fallback_actually_used,
            final_error_message,
        )

    def _process_image_input(
        self, image_input: Union[str, np.ndarray]
    ) -> Optional[np.ndarray]:
        """ประมวลผล input ของรูปภาพ"""
        try:
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                image = cv2.imread(image_input)
                if image is None:
                    raise ValueError(f"Cannot read image file: {image_input}")
                return image
            elif isinstance(image_input, np.ndarray):
                return image_input
            else:
                raise ValueError("Invalid image input type")
        except Exception as e:
            logger.error(f"Error processing image input: {e}")
            return None

    async def _run_detection(
        self, detector, image: np.ndarray, conf_threshold: float, iou_threshold: float
    ) -> List[np.ndarray]:
        """รันการตรวจจับด้วยโมเดลที่ระบุ"""
        try:
            if hasattr(detector, "detect_faces_raw") and callable(
                detector.detect_faces_raw
            ):
                if asyncio.iscoroutinefunction(detector.detect_faces_raw):
                    return await detector.detect_faces_raw(
                        image,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                    )
                else:
                    return detector.detect_faces_raw(
                        image,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                    )
            elif hasattr(detector, "detect") and callable(detector.detect):
                if asyncio.iscoroutinefunction(detector.detect):
                    return await detector.detect(
                        image,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                    )
                else:
                    return detector.detect(
                        image,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                    )
            else:
                raise RuntimeError("Detector has no suitable detection method")
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def _process_raw_detections(
        self,
        raw_bboxes: List[np.ndarray],
        image: np.ndarray,
        model_name: str,
        detection_time: float,
        min_quality: float,
    ) -> List[FaceDetection]:
        """ประมวลผล raw detection results"""
        processed_faces = []

        for raw_bbox in raw_bboxes:
            try:
                # Convert to BoundingBox object
                if isinstance(raw_bbox, np.ndarray) and len(raw_bbox) >= 5:
                    bbox_obj = BoundingBox(
                        x1=float(raw_bbox[0]),
                        y1=float(raw_bbox[1]),
                        x2=float(raw_bbox[2]),
                        y2=float(raw_bbox[3]),
                        confidence=float(raw_bbox[4]),
                    )
                else:
                    continue

                # Validate bounding box
                if not validate_bounding_box(
                    bbox_obj, image.shape[:2], relaxed_validation=True
                ):
                    continue

                # Calculate quality score
                quality_score = calculate_face_quality(
                    bbox_obj, image.shape[:2], self.quality_analyzer.config
                )

                # Apply quality filter
                if quality_score >= min_quality:
                    face_detection = FaceDetection(
                        bbox=bbox_obj,
                        quality_score=quality_score,
                        model_used=model_name,
                        processing_time=detection_time / len(raw_bboxes)
                        if raw_bboxes
                        else detection_time,
                    )
                    processed_faces.append(face_detection)

            except Exception as e:
                logger.debug(f"Error processing detection: {e}")
                continue

        return processed_faces

    async def _run_fallback_detection(
        self,
        image: np.ndarray,
        fb_config: Dict[str, Any],
        min_quality: float,
        attempt_num: int,
    ) -> List[FaceDetection]:
        """รันการตรวจจับแบบ fallback"""
        fb_model_name = fb_config.get("model_name")
        fb_conf = fb_config.get("conf_threshold", 0.15)
        fb_iou = fb_config.get("iou_threshold", 0.35)
        fb_min_faces = fb_config.get("min_faces_to_accept", 1)

        logger.info(
            f"Fallback attempt {attempt_num + 1}: {fb_model_name} "
            f"(conf={fb_conf}, iou={fb_iou})"
        )

        detected_faces = []

        try:
            if fb_model_name == "opencv_haar":
                # OpenCV Haar Cascade fallback
                haar_scale = fb_config.get("scale_factor", 1.1)
                haar_min_neighbors = fb_config.get("min_neighbors", 5)
                haar_min_size = fb_config.get("min_size", (30, 30))

                start_time = time.time()
                raw_bboxes = fallback_opencv_detection(
                    image,
                    scale_factor=haar_scale,
                    min_neighbors=haar_min_neighbors,
                    min_size=haar_min_size,
                )
                detection_time = (time.time() - start_time) * 1000

                detected_faces = self._process_raw_detections(
                    raw_bboxes, image, "opencv_haar", detection_time, min_quality
                )

            elif fb_model_name in self.models:
                # YOLO model fallback
                detector = self.models[fb_model_name]
                if detector.model_loaded:
                    start_time = time.time()
                    raw_bboxes = await self._run_detection(
                        detector, image, fb_conf, fb_iou
                    )
                    detection_time = (time.time() - start_time) * 1000

                    detected_faces = self._process_raw_detections(
                        raw_bboxes, image, fb_model_name, detection_time, min_quality
                    )

            logger.info(f"Fallback {fb_model_name}: {len(detected_faces)} faces")

            # Check if meets minimum requirement
            if len(detected_faces) >= fb_min_faces:
                return detected_faces
            else:
                return []

        except Exception as e:
            logger.error(f"Fallback detection failed for {fb_model_name}: {e}")
            return []

    def _update_performance_stats(
        self,
        faces: List[FaceDetection],
        model_used: str,
        processing_time: float,
        fallback_used: bool,
    ):
        """อัปเดตสถิติประสิทธิภาพ"""
        self.performance_stats["total_detections"] += 1

        if faces:
            self.performance_stats["successful_detections"] += 1

        if fallback_used:
            self.performance_stats["fallback_used_count"] += 1

        # Update average processing time
        total_detections = self.performance_stats["total_detections"]
        current_avg = self.performance_stats["average_processing_time"]
        self.performance_stats["average_processing_time"] = (
            current_avg * (total_detections - 1) + processing_time
        ) / total_detections

        # Update model usage count
        if model_used in self.performance_stats["model_usage_count"]:
            self.performance_stats["model_usage_count"][model_used] += 1

    async def get_service_info(self) -> Dict[str, Any]:
        """ดึงข้อมูลสถานะของบริการ"""
        try:
            vram_status = await self.vram_manager.get_vram_status()

            model_info = {}
            for name, model in self.models.items():
                if hasattr(model, "get_performance_stats"):
                    model_info[name] = model.get_performance_stats()
                else:
                    model_info[name] = {
                        "model_name": name,
                        "model_loaded": getattr(model, "model_loaded", False),
                        "device": getattr(model, "device", "unknown"),
                    }

            return {
                "service_status": "online" if self.models_loaded else "offline",
                "configuration": "relaxed",
                "models_loaded": self.models_loaded,
                "model_info": model_info,
                "vram_status": vram_status,
                "performance_stats": self.performance_stats,
                "detection_config": {
                    "conf_threshold": self.detection_params["conf_threshold"],
                    "iou_threshold": self.detection_params["iou_threshold"],
                    "min_quality_threshold": self.decision_criteria[
                        "min_quality_threshold"
                    ],
                },
                "fallback_enabled": self.fallback_config.get(
                    "enable_fallback_system", False
                ),
            }

        except Exception as e:
            logger.error(f"Error getting service info: {e}")
            return {"service_status": "error", "error": str(e)}

    async def cleanup(self):
        """ทำความสะอาดทรัพยากร"""
        try:
            logger.info("🧹 Cleaning up Face Detection Service...")

            # Cleanup models
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, "cleanup") and callable(model.cleanup):
                        if asyncio.iscoroutinefunction(model.cleanup):
                            await model.cleanup()
                        else:
                            model.cleanup()
                    logger.info(f"✅ Cleaned up model: {model_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Error cleaning up model {model_name}: {e}")

            # Clear dictionaries
            self.models.clear()
            self.model_stats.clear()

            # Reset state
            self.models_loaded = False

            # Release VRAM allocations
            if self.vram_manager:
                try:
                    await self.vram_manager.release_model_allocation("yolov9c-face")
                    await self.vram_manager.release_model_allocation("yolov9e-face")
                    await self.vram_manager.release_model_allocation("yolov11m-face")
                except Exception as e:
                    logger.warning(f"⚠️ Error releasing VRAM allocations: {e}")

            logger.info("✅ Face Detection Service cleanup completed")

        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """ดึงสถิติประสิทธิภาพของบริการ"""
        return self.performance_stats.copy()
