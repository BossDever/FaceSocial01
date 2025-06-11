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
from typing import Dict, List, Tuple, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass

from .yolo_models import YOLOv9ONNXDetector, YOLOv11Detector, fallback_opencv_detection
from .utils import (
    BoundingBox, FaceDetection, DetectionResult, 
    calculate_face_quality, validate_bounding_box, 
    filter_detection_results, get_relaxed_face_detection_config
)

logger = logging.getLogger(__name__)

class QualityCategory(Enum):
    """ระดับคุณภาพของใบหน้า"""
    EXCELLENT = "excellent"  # 80-100
    GOOD = "good"           # 70-79
    ACCEPTABLE = "acceptable"  # 40-69
    POOR = "poor"           # < 40

@dataclass
class FaceQualityAnalyzer:
    """ระบบวิเคราะห์คุณภาพใบหน้า - Enhanced"""
    config: Dict[str, Any]
    
    def __post_init__(self):
        self.quality_weights = {
            'size_weight': self.config.get('size_weight', 30),
            'area_weight': self.config.get('area_weight', 25),
            'confidence_weight': self.config.get('confidence_weight', 30),
            'aspect_weight': self.config.get('aspect_weight', 15)
        }
        
        self.size_thresholds = {
            'excellent': self.config.get('excellent_size', (80, 80)),
            'good': self.config.get('good_size', (50, 50)),
            'acceptable': self.config.get('acceptable_size', (24, 24)),
            'minimum': self.config.get('minimum_size', (8, 8))
        }
        
        self.min_quality_threshold = self.config.get('min_quality_threshold', 40)
        self.bonus_score = self.config.get('bonus_score_for_high_confidence', 5.0)
        self.high_confidence_threshold = self.config.get('high_confidence_threshold', 0.7)
    
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
                'total_count': 0,
                'usable_count': 0,
                'quality_ratio': 0.0,
                'quality_categories': {cat.value: 0 for cat in QualityCategory},
                'avg_quality': 0.0
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
            'total_count': total_count,
            'usable_count': usable_count,
            'quality_ratio': quality_ratio,
            'quality_categories': quality_categories,
            'avg_quality': avg_quality
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
            'step1_results': {
                'yolov9c': {'count': len(self.yolov9c_detections), 'time': self.yolov9c_time},
                'yolov9e': {'count': len(self.yolov9e_detections), 'time': self.yolov9e_time}
            },
            'step2_agreement': {
                'agreement': self.agreement,
                'ratio': self.agreement_ratio,
                'type': self.agreement_type
            },
            'step3_decision': {
                'use_yolov11m': self.use_yolov11m,
                'reasons': self.decision_reasons
            },
            'step4_results': {
                'model_used': self.final_model,
                'count': len(self.final_detections),
                'time': self.final_time
            },
            'quality_info': self.quality_info,
            'total_time': self.total_time,
            'fallback_info': {
                'fallback_used': self.fallback_used,
                'attempts': self.fallback_attempts_info
            }
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
        self.config = config if config is not None else get_relaxed_face_detection_config()
        
        # Model management
        self.models: Dict[str, Union[YOLOv9ONNXDetector, YOLOv11Detector]] = {}
        self.model_stats: Dict[str, Dict[str, Union[float, int]]] = {}
        
        # Configuration
        self.use_enhanced_detector = self.config.get('use_enhanced_detector', False)
        
        # Decision criteria
        self.decision_criteria = {
            'max_usable_faces_yolov9': int(self.config.get('max_usable_faces_yolov9', 12)),
            'min_agreement_ratio': float(self.config.get('min_agreement_ratio', 0.5)),
            'min_quality_threshold': int(self.config.get('min_quality_threshold', 40)),
            'iou_threshold': float(self.config.get('iou_threshold_agreement', 0.3))
        }
        
        # Detection parameters
        self.detection_params = {
            'conf_threshold': self.config.get('conf_threshold', 0.10),
            'iou_threshold': self.config.get('iou_threshold_nms', 0.35),
            'img_size': self.config.get('img_size', 640)
        }
        
        # Quality analyzer
        quality_config = self.config.get('quality_config', {})
        quality_config.setdefault('min_quality_threshold', self.decision_criteria['min_quality_threshold'])
        self.quality_analyzer = FaceQualityAnalyzer(quality_config)
        
        # Model paths
        self.yolov9c_model_path = self.config.get('yolov9c_model_path', 'model/face-detection/yolov9c-face-lindevs.onnx')
        self.yolov9e_model_path = self.config.get('yolov9e_model_path', 'model/face-detection/yolov9e-face-lindevs.onnx')
        self.yolov11m_model_path = self.config.get('yolov11m_model_path', 'model/face-detection/yolov11m-face.pt')
        
        # Fallback configuration
        self.fallback_config = self.config.get('fallback_config', get_relaxed_face_detection_config()['fallback_config'])
        
        # Statistics
        self.decision_log = []
        self.models_loaded = False
        
        # Performance tracking
        self.performance_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'fallback_used_count': 0,
            'average_processing_time': 0.0,
            'model_usage_count': {'yolov9c': 0, 'yolov9e': 0, 'yolov11m': 0, 'opencv_haar': 0}
        }
    
    async def initialize(self) -> bool:
        """โหลดโมเดลตรวจจับใบหน้าทั้งหมด"""
        try:
            logger.info("Loading face detection models (Enhanced/Relaxed Mode)...")
            
            # Request VRAM allocations
            yolov9c_allocation = await self.vram_manager.request_model_allocation("yolov9c-face", "high", "face_detection_service")
            yolov9e_allocation = await self.vram_manager.request_model_allocation("yolov9e-face", "high", "face_detection_service")
            yolov11m_allocation = await self.vram_manager.request_model_allocation("yolov11m-face", "critical", "face_detection_service")
            
            # Load YOLOv9c
            self.models['yolov9c'] = YOLOv9ONNXDetector(self.yolov9c_model_path, "YOLOv9c")
            device_yolov9c = "cuda" if yolov9c_allocation.location.value == "gpu" else "cpu"
            if not self.models['yolov9c'].load_model(device_yolov9c):
                logger.error("Failed to load YOLOv9c model")
            
            # Load YOLOv9e
            self.models['yolov9e'] = YOLOv9ONNXDetector(self.yolov9e_model_path, "YOLOv9e")
            device_yolov9e = "cuda" if yolov9e_allocation.location.value == "gpu" else "cpu"
            if not self.models['yolov9e'].load_model(device_yolov9e):
                logger.error("Failed to load YOLOv9e model")
            
            # Load YOLOv11m
            self.models['yolov11m'] = YOLOv11Detector(self.yolov11m_model_path, "YOLOv11m")
            device_yolov11m = "cuda" if yolov11m_allocation.location.value == "gpu" else "cpu"
            if not self.models['yolov11m'].load_model(device_yolov11m):
                logger.error("Failed to load YOLOv11m model")
            
            # Check if at least one model loaded successfully
            loaded_models = [name for name, model in self.models.items() if model.model_loaded]
            
            if not loaded_models:
                logger.error("No models loaded successfully")
                return False
            
            self.models_loaded = True
            logger.info(f"✅ Face Detection Service initialized with models: {loaded_models}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}", exc_info=True)
            return False
    
    async def detect_faces(self, 
                         image_input: Union[str, np.ndarray],
                         model_name: Optional[str] = None,
                         conf_threshold: Optional[float] = None,
                         iou_threshold: Optional[float] = None,
                         min_face_size: Optional[int] = None,
                         max_faces: Optional[int] = None,
                         return_landmarks: bool = False,
                         min_quality_threshold: Optional[float] = None,
                         use_fallback: bool = True,
                         fallback_strategy: Optional[List[Dict[str, Any]]] = None,
                         force_cpu: bool = False) -> DetectionResult:
        """
        ตรวจจับใบหน้าในรูปภาพโดยเลือกโมเดลที่เหมาะสมที่สุดโดยอัตโนมัติ
        พร้อมระบบ Fallback ที่ปรับปรุงใหม่
        """
        start_time_total = time.time()
        
        if not self.models_loaded:
            logger.warning("Models not loaded. Attempting to initialize...")
            initialized = await self.initialize()
            if not initialized:
                return DetectionResult(
                    faces=[], 
                    image_shape=(0, 0, 0),
                    total_processing_time=time.time() - start_time_total,
                    model_used="N/A", 
                    error="Models not loaded and initialization failed."
                )
        
        # Process image input
        try:
            image = self._process_image_input(image_input)
            if image is None or image.size == 0:
                return DetectionResult(
                    faces=[], 
                    image_shape=(0, 0, 0),
                    total_processing_time=time.time() - start_time_total,
                    model_used="N/A", 
                    error="Invalid image input"
                )
        except Exception as e:
            return DetectionResult(
                faces=[], 
                image_shape=(0, 0, 0),
                total_processing_time=time.time() - start_time_total,
                model_used="N/A", 
                error=f"Error processing image: {e}"
            )
        
        # Set parameters
        primary_model = model_name if model_name and model_name != 'auto' else 'yolov9c'
        current_conf = conf_threshold if conf_threshold is not None else self.detection_params['conf_threshold']
        current_iou = iou_threshold if iou_threshold is not None else self.detection_params['iou_threshold']
        current_min_quality = min_quality_threshold if min_quality_threshold is not None else self.config.get('filter_min_quality_final', 40.0)
        
        logger.info(f"Starting detection: model={primary_model}, conf={current_conf}, iou={current_iou}")
        
        # Initialize variables
        detected_faces_final: List[FaceDetection] = []
        model_used_for_detection = "N/A"
        fallback_actually_used = False
        error_message = None
        
        # Primary detection attempt
        try:
            if primary_model in self.models:
                detector = self.models[primary_model]
                
                if detector.model_loaded:
                    start_detect_time = time.time()
                    raw_bboxes = await self._run_detection(detector, image, current_conf, current_iou)
                    detection_time_ms = (time.time() - start_detect_time) * 1000
                    
                    # Process raw detections
                    processed_faces = self._process_raw_detections(
                        raw_bboxes, image, primary_model, detection_time_ms, current_min_quality
                    )
                    
                    detected_faces_final.extend(processed_faces)
                    model_used_for_detection = primary_model
                    
                    logger.info(f"Primary detection ({primary_model}): {len(processed_faces)} valid faces")
                else:
                    error_message = f"Primary model {primary_model} not loaded"
                    logger.error(error_message)
            else:
                error_message = f"Primary model {primary_model} not found"
                logger.error(error_message)
                
        except Exception as e:
            error_message = f"Error in primary detection: {str(e)}"
            logger.error(error_message, exc_info=True)
        
        # Fallback system
        if use_fallback and (not detected_faces_final or error_message):
            logger.info("Initiating fallback detection system")
            fallback_actually_used = True
            
            fallback_strategy = fallback_strategy or self.fallback_config.get('fallback_models', [])
            
            for fb_attempt, fb_config in enumerate(fallback_strategy):
                if detected_faces_final:
                    break  # Stop if we found faces
                
                try:
                    fb_faces = await self._run_fallback_detection(
                        image, fb_config, current_min_quality, fb_attempt
                    )
                    
                    if fb_faces:
                        detected_faces_final = fb_faces
                        model_used_for_detection = fb_config.get('model_name', 'unknown')
                        error_message = None
                        logger.info(f"Fallback successful with {model_used_for_detection}")
                        break
                        
                except Exception as e:
                    logger.error(f"Fallback attempt {fb_attempt} failed: {e}")
                    continue
            
            if not detected_faces_final and not error_message:
                error_message = "All detection attempts failed"
        
        # Apply max_faces limit
        if max_faces and len(detected_faces_final) > max_faces:
            detected_faces_final.sort(key=lambda f: f.quality_score or 0, reverse=True)
            detected_faces_final = detected_faces_final[:max_faces]
        
        # Update performance statistics
        total_service_time = time.time() - start_time_total
        self._update_performance_stats(detected_faces_final, model_used_for_detection, total_service_time, fallback_actually_used)
        
        # Create final result
        return DetectionResult(
            faces=detected_faces_final,
            image_shape=image.shape,
            total_processing_time=total_service_time * 1000,  # Convert to ms
            model_used=model_used_for_detection,
            fallback_used=fallback_actually_used,
            error=error_message,
            metadata={
                'config_used': 'relaxed',
                'quality_threshold': current_min_quality,
                'conf_threshold': current_conf,
                'iou_threshold': current_iou
            }
        )
    
    def _process_image_input(self, image_input: Union[str, np.ndarray]) -> Optional[np.ndarray]:
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
    
    async def _run_detection(self, detector, image: np.ndarray, conf_threshold: float, iou_threshold: float) -> List[np.ndarray]:
        """รันการตรวจจับด้วยโมเดลที่ระบุ"""
        try:
            if hasattr(detector, 'detect_faces_raw') and callable(detector.detect_faces_raw):
                if asyncio.iscoroutinefunction(detector.detect_faces_raw):
                    return await detector.detect_faces_raw(image, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
                else:
                    return detector.detect_faces_raw(image, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
            elif hasattr(detector, 'detect') and callable(detector.detect):
                if asyncio.iscoroutinefunction(detector.detect):
                    return await detector.detect(image, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
                else:
                    return detector.detect(image, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
            else:
                raise RuntimeError(f"Detector has no suitable detection method")
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def _process_raw_detections(self, raw_bboxes: List[np.ndarray], image: np.ndarray, 
                              model_name: str, detection_time: float, min_quality: float) -> List[FaceDetection]:
        """ประมวลผล raw detection results"""
        processed_faces = []
        
        for raw_bbox in raw_bboxes:
            try:
                # Convert to BoundingBox object
                if isinstance(raw_bbox, np.ndarray) and len(raw_bbox) >= 5:
                    bbox_obj = BoundingBox(
                        x1=float(raw_bbox[0]), y1=float(raw_bbox[1]),
                        x2=float(raw_bbox[2]), y2=float(raw_bbox[3]),
                        confidence=float(raw_bbox[4])
                    )
                else:
                    continue
                
                # Validate bounding box
                if not validate_bounding_box(bbox_obj, image.shape[:2], relaxed_validation=True):
                    continue
                
                # Calculate quality score
                quality_score = calculate_face_quality(bbox_obj, image.shape[:2], self.quality_analyzer.config)
                
                # Apply quality filter
                if quality_score >= min_quality:
                    face_detection = FaceDetection(
                        bbox=bbox_obj,
                        quality_score=quality_score,
                        model_used=model_name,
                        processing_time=detection_time / len(raw_bboxes) if raw_bboxes else detection_time
                    )
                    processed_faces.append(face_detection)
                    
            except Exception as e:
                logger.debug(f"Error processing detection: {e}")
                continue
        
        return processed_faces
    
    async def _run_fallback_detection(self, image: np.ndarray, fb_config: Dict[str, Any], 
                                    min_quality: float, attempt_num: int) -> List[FaceDetection]:
        """รันการตรวจจับแบบ fallback"""
        fb_model_name = fb_config.get('model_name')
        fb_conf = fb_config.get('conf_threshold', 0.15)
        fb_iou = fb_config.get('iou_threshold', 0.35)
        fb_min_faces = fb_config.get('min_faces_to_accept', 1)
        
        logger.info(f"Fallback attempt {attempt_num + 1}: {fb_model_name} (conf={fb_conf}, iou={fb_iou})")
        
        detected_faces = []
        
        try:
            if fb_model_name == 'opencv_haar':
                # OpenCV Haar Cascade fallback
                haar_scale = fb_config.get('scale_factor', 1.1)
                haar_min_neighbors = fb_config.get('min_neighbors', 5)
                haar_min_size = fb_config.get('min_size', (30, 30))
                
                start_time = time.time()
                raw_bboxes = fallback_opencv_detection(
                    image, scale_factor=haar_scale, 
                    min_neighbors=haar_min_neighbors, min_size=haar_min_size
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
                    raw_bboxes = await self._run_detection(detector, image, fb_conf, fb_iou)
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
    
    def _update_performance_stats(self, faces: List[FaceDetection], model_used: str, 
                                processing_time: float, fallback_used: bool):
        """อัปเดตสถิติประสิทธิภาพ"""
        self.performance_stats['total_detections'] += 1
        
        if faces:
            self.performance_stats['successful_detections'] += 1
        
        if fallback_used:
            self.performance_stats['fallback_used_count'] += 1
        
        # Update average processing time
        total_detections = self.performance_stats['total_detections']
        current_avg = self.performance_stats['average_processing_time']
        self.performance_stats['average_processing_time'] = (
            (current_avg * (total_detections - 1) + processing_time) / total_detections
        )
        
        # Update model usage count
        if model_used in self.performance_stats['model_usage_count']:
            self.performance_stats['model_usage_count'][model_used] += 1
    
    async def get_service_info(self) -> Dict[str, Any]:
        """ดึงข้อมูลสถานะของบริการ"""
        try:
            vram_status = await self.vram_manager.get_vram_status()
            
            model_info = {}
            for name, model in self.models.items():
                if hasattr(model, 'get_performance_stats'):
                    model_info[name] = model.get_performance_stats()
                else:
                    model_info[name] = {
                        'model_name': name,
                        'model_loaded': getattr(model, 'model_loaded', False),
                        'device': getattr(model, 'device', 'unknown')
                    }
            
            return {
                'service_status': 'online' if self.models_loaded else 'offline',
                'configuration': 'relaxed',
                'models_loaded': self.models_loaded,
                'model_info': model_info,
                'vram_status': vram_status,
                'performance_stats': self.performance_stats,
                'detection_config': {
                    'conf_threshold': self.detection_params['conf_threshold'],
                    'iou_threshold': self.detection_params['iou_threshold'],
                    'min_quality_threshold': self.decision_criteria['min_quality_threshold']
                },
                'fallback_enabled': self.fallback_config.get('enable_fallback_system', False)
            }
            
        except Exception as e:
            logger.error(f"Error getting service info: {e}")
            return {
                'service_status': 'error',
                'error': str(e)
            }
    
    async def cleanup(self):
        """ทำความสะอาดทรัพยากร"""
        try:
            logger.info("🧹 Cleaning up Face Detection Service...")
            
            # Cleanup models
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'cleanup') and callable(model.cleanup):
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