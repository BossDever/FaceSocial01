# cSpell:disable
# mypy: ignore-errors
"""
Face Analysis Service
ระบบวิเคราะห์ใบหน้าแบบครบวงจร (Detection + Recognition)
Enhanced End-to-End Solution with better error handling and performance
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import asyncio
import time
import numpy as np
import cv2
import logging

# Use relative import and suppress missing import
from ...core.log_config import get_logger

from .models import (
    AnalysisConfig,
    FaceResult,
    FaceAnalysisResult,
    AnalysisMode,
    QualityLevel,
)

# Import detection utils
try:
    from ..face_detection.utils import BoundingBox
    DETECTION_UTILS_AVAILABLE = True
except ImportError:
    DETECTION_UTILS_AVAILABLE = False
    # Fallback BoundingBox definition
    class BoundingBox:
        def __init__(self, x1: float, y1: float, x2: float, y2: float, confidence: float):
            self.x1 = x1
            self.y1 = y1
            self.x2 = x2
            self.y2 = y2
            self.confidence = confidence

logger = get_logger(__name__)


class FaceAnalysisService:
    """
    Enhanced Face Analysis Service
    รวม Face Detection + Face Recognition ในระบบเดียว
    """

    def __init__(
        self,
        vram_manager: Any = None,
        config: Optional[Dict[str, Any]] = None,
        face_detection_service: Optional[Any] = None,
        face_recognition_service: Optional[Any] = None,
    ):
        self.vram_manager = vram_manager
        self.logger = logger

        # Parse configuration
        self.config = config or {}
        self.config.setdefault("detection", {})
        self.config.setdefault("recognition", {})

        # Sub-services - will be initialized later
        self.face_detection_service = face_detection_service
        self.face_recognition_service = face_recognition_service

        if self.face_detection_service is None:
            logger.info("FaceAnalysisService: FaceDetectionService not provided at init, creating new.")
        
        if self.face_recognition_service is None:
            logger.info("FaceAnalysisService: FaceRecognitionService not provided at init, creating new.")
        
        # Performance tracking
        self.stats: Dict[str, Any] = {
            "total_analyses": 0,
            "total_faces_detected": 0,
            "total_faces_recognized": 0,
            "processing_times": [],
            "success_rates": [],
            "detection_times": [],
            "recognition_times": [],
        }

        self.logger.info("Face Analysis Service initialized")

    def set_shared_services(
        self,
        face_detection_service: Any,
        face_recognition_service: Any,
    ) -> None:
        """Inject shared service instances."""
        self.logger.info("FaceAnalysisService: Setting shared FaceDetectionService and FaceRecognitionService.")
        self.face_detection_service = face_detection_service
        self.face_recognition_service = face_recognition_service
    
    async def initialize(self) -> bool:
        self.logger.info("🔧 Initializing Face Analysis Service...")
        # Initialization logic for FaceAnalysisService itself, if any.
        # Dependencies (detection/recognition services) should be initialized by the caller (main.py)
        # and injected via set_shared_services.

        if self.face_detection_service is None:
            self.logger.warning("⚠️ FaceDetectionService not set yet (will be injected later)")
        else:
            self.logger.info("✅ FaceDetectionService is set.")

        if self.face_recognition_service is None:
            self.logger.warning("⚠️ FaceRecognitionService not set yet (will be injected later)")
        else:
            self.logger.info("✅ FaceRecognitionService is set.")
            
        self.logger.info("✅ Face Analysis Service initialized (or checked for shared services).")
        return True

    async def _handle_detection(
        self, image: np.ndarray, config: AnalysisConfig
    ) -> Tuple[List[FaceResult], float, Optional[str]]:
        """Handles the face detection part of the analysis."""
        detection_start_time = time.time()
        if not self.face_detection_service:
            raise RuntimeError("Face detection service not available")

        try:
            detection_result = await self.face_detection_service.detect_faces(
                image,
                model_name=config.detection_model,
                conf_threshold=config.confidence_threshold,
                min_face_size=config.min_face_size,
                max_faces=config.max_faces,
                return_landmarks=True,
                min_quality_threshold=60.0 if config.use_quality_based_selection else 30.0,
            )
            detection_time = time.time() - detection_start_time
            detection_model_used = detection_result.model_used
            self.logger.info(
                f"Detection: {len(detection_result.faces)} faces "
                f"in {detection_time:.3f}s"
            )
            faces = await self._convert_detection_results(
                detection_result, config, image
            )
            return faces, detection_time, detection_model_used
        except Exception as det_error:
            self.logger.error(f"❌ Detection failed: {det_error}")
            detection_time = time.time() - detection_start_time
            return [], detection_time, None

    async def _handle_recognition(
        self,
        image: np.ndarray,
        faces: List[FaceResult],
        config: AnalysisConfig,
        gallery: Optional[Dict[str, Any]],
    ) -> Tuple[float, Optional[str]]:
        """Handles the face recognition part of the analysis."""
        recognition_start_time = time.time()
        recognition_model_used = None
        if not self.face_recognition_service:
            self.logger.warning("Face recognition service not available")
            return time.time() - recognition_start_time, recognition_model_used

        try:
            quality_threshold = (
                60.0 if config.use_quality_based_selection else 0.0
            )
            processable_faces = [
                f for f in faces if f.quality_score >= quality_threshold
            ]

            if processable_faces:
                await self._process_recognition_for_faces(
                    image, processable_faces, config, gallery
                )
                recognized_faces_count = sum(
                    1 for f in processable_faces if f.has_identity
                )
                recognition_model_used = config.recognition_model

                self.logger.info(
                    f"Recognition: {recognized_faces_count}/"
                    f"{len(processable_faces)} faces recognized"
                )

        except Exception as rec_error:
            self.logger.error(f"❌ Recognition failed: {rec_error}")

        return time.time() - recognition_start_time, recognition_model_used

    async def _handle_recognition_only(
        self, image: np.ndarray, config: AnalysisConfig, gallery: Optional[Dict[str, Any]]
    ) -> Tuple[List[FaceResult], float, Optional[str]]:
        """Handles the recognition-only mode."""
        recognition_start_time = time.time()
        faces: List[FaceResult] = []
        recognition_model_used = None

        if not self.face_recognition_service:
            raise RuntimeError("Face recognition service not available")

        try:
            # Convert np.ndarray to bytes for recognize_faces
            is_success, buffer = cv2.imencode(".jpg", image)  # Assuming image is BGR
            if not is_success:
                self.logger.error("❌ Failed to encode image for recognition-only mode")
                image_bytes_for_recognition = None
            else:
                image_bytes_for_recognition = buffer.tobytes()

            recognition_result_dict = {
                "success": False,
                "error": "Image encoding failed or not processed",
            }

            if image_bytes_for_recognition:
                if gallery and config.enable_gallery_matching:
                    recognition_result_dict = (
                        await self.face_recognition_service.recognize_faces_with_gallery(
                            image_bytes=image_bytes_for_recognition,
                            gallery=gallery,
                            model_name=config.recognition_model,
                        )
                    )
                else:
                    recognition_result_dict = (
                        await self.face_recognition_service.recognize_faces(
                            image_bytes=image_bytes_for_recognition,
                            model_name=config.recognition_model,
                        )
                    )

            if recognition_result_dict.get("success"):
                face_result = await self._create_recognition_only_result(
                    image, recognition_result_dict
                )
                if face_result:
                    if isinstance(face_result, list):
                        faces.extend(face_result)
                    else:
                        faces.append(face_result)

                recognition_model_used = (
                    config.recognition_model
                    if config.recognition_model
                    else None
                )
            else:
                self.logger.error(
                    f"❌ Recognition-only processing failed: "
                    f"{recognition_result_dict.get('error')}"
                )
        except Exception as rec_error:
            self.logger.error(
                f"❌ Recognition-only failed with exception: {rec_error}",
                exc_info=True,
            )

        return faces, time.time() - recognition_start_time, recognition_model_used

    async def analyze_faces(
        self,
        image: np.ndarray,
        config: Union[AnalysisConfig, Dict[str, Any]],  # รองรับทั้ง object และ dict
        gallery: Optional[Dict[str, Any]] = None,
    ) -> FaceAnalysisResult:
        """
        วิเคราะห์ใบหน้าครบวงจร

        Args:
            image: รูปภาพ (BGR format)
            config: การตั้งค่าการวิเคราะห์ (AnalysisConfig object หรือ dict)
            gallery: ฐานข้อมูลใบหน้าสำหรับจดจำ

        Returns:
            FaceAnalysisResult
        """
        start_time = time.time()
        detection_time = 0.0
        recognition_time = 0.0
        faces: List[FaceResult] = []
        detection_model_used = None
        recognition_model_used = None
        gallery_actually_used = False        # Declare current_config here to ensure it's always defined
        current_config: AnalysisConfig

        try:
            # แปลง config เป็น AnalysisConfig object ถ้าเป็น dict
            if isinstance(config, dict):
                try:
                    # สร้างสำเนาของ config เพื่อไม่ให้แก้ไข original
                    config_copy = config.copy()
                    
                    # แปลง string เป็น enum
                    if isinstance(config_copy.get("mode"), str):
                        try:
                            config_copy["mode"] = AnalysisMode(config_copy["mode"])
                        except ValueError:
                            config_copy["mode"] = AnalysisMode.FULL_ANALYSIS
                    
                    if isinstance(config_copy.get("quality_level"), str):
                        try:
                            config_copy["quality_level"] = QualityLevel(config_copy["quality_level"])
                        except ValueError:
                            config_copy["quality_level"] = QualityLevel.BALANCED
                    
                    current_config = AnalysisConfig(**config_copy)
                    
                except Exception as e:
                    self.logger.error(f"Failed to convert dict to AnalysisConfig: {e}")
                    # สร้าง default AnalysisConfig
                    current_config = AnalysisConfig()
            elif isinstance(config, AnalysisConfig):
                current_config = config
            else:
                self.logger.error(f"Invalid config type: {type(config)}. Using default.")
                current_config = AnalysisConfig()

            # Step 1: Face Detection
            if current_config.mode in [
                AnalysisMode.DETECTION_ONLY,
                AnalysisMode.FULL_ANALYSIS,
                AnalysisMode.COMPREHENSIVE,
            ]:
                faces, detection_time, detection_model_used = (
                    await self._handle_detection(image, current_config)
                )

            # Step 2: Face Recognition (modified to pass gallery)
            if (
                current_config.mode in [AnalysisMode.FULL_ANALYSIS, AnalysisMode.COMPREHENSIVE]
                and faces # Only proceed if faces were detected
            ):
                if gallery and current_config.enable_gallery_matching:
                    self.logger.info(
                        f"Starting recognition for {len(faces)} faces with provided gallery ({len(gallery)} people)."
                    )
                    gallery_actually_used = True
                    recognition_time_val, recognition_model_used_rec = await self._handle_recognition(
                        image, faces, current_config, gallery
                    )
                    recognition_time += recognition_time_val
                    if recognition_model_used_rec:
                        recognition_model_used = recognition_model_used_rec
                elif current_config.enable_database_matching:
                    self.logger.info(
                        f"Starting recognition for {len(faces)} faces with internal database."
                    )
                    recognition_time_val, recognition_model_used_rec = await self._handle_recognition(
                        image, faces, current_config, None
                    )
                    recognition_time += recognition_time_val
                    if recognition_model_used_rec:
                        recognition_model_used = recognition_model_used_rec
                else:
                    self.logger.info("Recognition skipped: Gallery not provided/enabled, and DB matching not enabled.")

            # Step 3: Handle recognition-only mode (modified to pass gallery)
            elif current_config.mode == AnalysisMode.RECOGNITION_ONLY:
                if gallery and current_config.enable_gallery_matching:
                    self.logger.info(f"Processing recognition-only mode with provided gallery ({len(gallery)} people).")
                    gallery_actually_used = True
                    faces, rec_time, rec_model = await self._handle_recognition_only(
                        image, current_config, gallery
                    )
                    recognition_time += rec_time
                    if rec_model:
                        recognition_model_used = rec_model
                elif current_config.enable_database_matching:
                    self.logger.info("Processing recognition-only mode with internal database.")
                    faces, rec_time, rec_model = await self._handle_recognition_only(
                        image, current_config, None
                    )
                    recognition_time += rec_time
                    if rec_model:
                        recognition_model_used = rec_model
                else:
                    self.logger.info("Recognition-only skipped: Gallery not provided/enabled, and DB matching not enabled.")

            total_time = time.time() - start_time

            # สร้างผลลัพธ์
            result = FaceAnalysisResult(
                image_shape=image.shape,
                config=current_config,
                faces=faces,
                detection_time=detection_time,
                recognition_time=recognition_time,
                total_time=total_time,
                detection_model_used=detection_model_used,
                recognition_model_used=recognition_model_used,
                analysis_metadata={
                    "quality_level": current_config.quality_level.value if hasattr(current_config.quality_level, 'value') else str(current_config.quality_level),
                    "parallel_processing": current_config.parallel_processing,
                    "gallery_size": len(gallery) if gallery else 0,
                    "gallery_provided": gallery is not None,
                    "gallery_used_for_matching": gallery_actually_used,
                    "database_used_for_matching": current_config.enable_database_matching and not gallery_actually_used,
                },
            )

            # อัปเดต statistics
            self._update_stats(result)

            self.logger.info(
                f"Analysis complete: {result.total_faces} faces, "
                f"{result.identified_faces} identified in {total_time:.3f}s"
            )

            return result

        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"❌ Face analysis failed: {e}")

            # Ensure config is an AnalysisConfig object for the error result
            error_config = current_config if 'current_config' in locals() and isinstance(current_config, AnalysisConfig) else AnalysisConfig()

            # Return error result
            return FaceAnalysisResult(
                image_shape=image.shape,
                config=error_config, # Use the ensured AnalysisConfig object
                faces=[],
                detection_time=detection_time,
                recognition_time=recognition_time,
                total_time=total_time,
                error=str(e),
                success=False,
            )

    async def _convert_detection_results(
        self, detection_result: Any, config: AnalysisConfig, image: np.ndarray
    ) -> List[FaceResult]:
        """แปลง detection results เป็น FaceResult objects"""
        faces = []

        for i, detected_face in enumerate(detection_result.faces):
            try:
                face_result = FaceResult(
                    bbox=detected_face.bbox,
                    confidence=detected_face.bbox.confidence,
                    quality_score=detected_face.quality_score or 0.0,
                    face_id=f"face_{i:03d}",
                    processing_time=detected_face.processing_time,
                    model_used=detected_face.model_used,
                    landmarks=getattr(detected_face, "landmarks", None),
                )

                # ตัดใบหน้าถ้าต้องการ
                if config.return_face_crops or config.mode in [
                    AnalysisMode.FULL_ANALYSIS,
                    AnalysisMode.COMPREHENSIVE,
                ]:
                    face_crop = self._extract_face_crop(image, detected_face.bbox)
                    if config.return_face_crops:
                        face_result.face_crop = face_crop

                faces.append(face_result)

            except Exception as e:
                self.logger.warning(f"⚠️ Failed to convert detection result {i}: {e}")
                continue

        return faces

    async def _execute_recognition_tasks(
        self, recognition_tasks: List[asyncio.Task], parallel_processing: bool
    ) -> None:
        """Executes recognition tasks either in parallel or sequentially."""
        if parallel_processing and len(recognition_tasks) > 1:
            try:
                await asyncio.gather(*recognition_tasks, return_exceptions=True)
            except Exception as e:
                self.logger.error(f"❌ Parallel recognition failed: {e}")
                # Fallback to sequential processing if gather fails
                for task in recognition_tasks:
                    try:
                        await task
                    except Exception as task_error:
                        self.logger.warning(
                            f"⚠️ Recognition task failed (fallback): {task_error}"
                        )
        else:
            # Sequential processing
            for task in recognition_tasks:
                try:
                    await task
                except Exception as task_error:
                    self.logger.warning(f"⚠️ Recognition task failed: {task_error}")

    async def _process_recognition_for_faces(
        self,
        image: np.ndarray,
        faces: List[FaceResult],
        config: AnalysisConfig,
        gallery: Optional[Dict[str, Any]],
    ) -> None:
        """ประมวลผล Face Recognition สำหรับหลายใบหน้า"""
        if not self.face_recognition_service:
            self.logger.warning("Face recognition service not available for batch processing.")
            return

        if not faces:
            self.logger.info("No faces provided for recognition processing.")
            return

        # Check if gallery matching is enabled and gallery is provided
        use_gallery = gallery and config.enable_gallery_matching
        
        if use_gallery:
            self.logger.info(
                f"Processing recognition for {len(faces)} faces with gallery ({len(gallery or {})} people)."
            )
        elif config.enable_database_matching:
            self.logger.info(
                f"Processing recognition for {len(faces)} faces with internal database."
            )
        else:
            self.logger.info("Recognition skipped: Neither gallery nor database matching is enabled.")
            return

        recognition_tasks = []
        for face_result in faces:
            if face_result.face_id is None:
                face_result.face_id = f"face_{time.time_ns()}"

            try:
                face_crop_bytes = face_result.get_face_crop_bytes(image, config.recognition_image_format)
                if face_crop_bytes is None:
                    face_crop_np = self._extract_face_crop(image, face_result.bbox)
                    if face_crop_np is not None:
                        _, buffer = cv2.imencode(f".{config.recognition_image_format}", face_crop_np)
                        face_crop_bytes = buffer.tobytes()

                if face_crop_bytes:
                    task = self._recognize_single_face(
                        face_result, face_crop_bytes, config, gallery if use_gallery else None
                    )
                    recognition_tasks.append(asyncio.create_task(task))
                else:
                    self.logger.warning(
                        f"⚠️ Failed to extract/encode face crop for recognition (ID: {face_result.face_id})."
                    )
            except Exception as e:
                self.logger.error(
                    f"⚠️ Error preparing face for recognition (ID: {face_result.face_id}): {e}",
                    exc_info=True
                )
                continue
        
        if recognition_tasks:
            await self._execute_recognition_tasks(recognition_tasks, config.parallel_processing)
            self.logger.info(f"Completed recognition tasks for {len(recognition_tasks)} faces.")
        else:
            self.logger.info("No recognition tasks were created.")

    async def _recognize_single_face(
        self,
        face_result: FaceResult,
        face_crop_bytes: bytes,
        config: AnalysisConfig,
        gallery: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Recognizes a single face, optionally using a provided gallery."""
        if not self.face_recognition_service:
            logger.warning("Face recognition service not available for _recognize_single_face")
            return

        if face_crop_bytes is None or len(face_crop_bytes) == 0:
            logger.warning(f"No face crop bytes for face_id {face_result.face_id}, skipping recognition.")
            return

        try:
            recognition_result_dict: Dict[str, Any] = {}
            if gallery and config.enable_gallery_matching:
                logger.debug(
                    f"Recognizing face {face_result.face_id} with gallery ({len(gallery)} people) "
                    f"using model {config.recognition_model}."
                )
                recognition_result_dict = await self.face_recognition_service.recognize_faces_with_gallery(
                    image_bytes=face_crop_bytes,
                    gallery=gallery,
                    model_name=config.recognition_model,
                )
            elif config.enable_database_matching:
                logger.debug(
                    f"Recognizing face {face_result.face_id} with internal database "
                    f"using model {config.recognition_model}."
                )
                recognition_result_dict = await self.face_recognition_service.recognize_faces(
                    image_bytes=face_crop_bytes, model_name=config.recognition_model
                )
            else:
                logger.debug(f"Recognition skipped for face {face_result.face_id} as no gallery/DB matching enabled.")
                face_result.recognition_model = config.recognition_model
                return

            if not isinstance(recognition_result_dict, dict):
                logger.error(
                    f"❌ Recognition service did not return a dict for face {face_result.face_id}. "
                    f"Got: {type(recognition_result_dict)}"
                )
                recognition_result_dict = {}            # Update FaceResult with recognition details
            face_result.query_embedding = recognition_result_dict.get("query_embedding")
            face_result.matches = recognition_result_dict.get("matches", [])
            face_result.best_match = recognition_result_dict.get("best_match")
            face_result.recognition_model = config.recognition_model
              # Apply unknown_threshold to best_match
            if face_result.best_match and hasattr(face_result.best_match, 'set_match_status'):
                logger.debug(f"Face {face_result.face_id}: Applying unknown_threshold {config.unknown_threshold} to FaceMatch object")
                face_result.best_match.set_match_status(config.unknown_threshold)
                logger.debug(f"Face {face_result.face_id}: is_match = {face_result.best_match.is_match}")
            elif face_result.best_match and isinstance(face_result.best_match, dict):
                # For dict-based best_match, create a proper FaceMatch object
                logger.debug(f"Face {face_result.face_id}: Converting dict best_match to FaceMatch object")
                from ..face_recognition.models import FaceMatch
                similarity = face_result.best_match.get('similarity', 0.0)
                face_match = FaceMatch(
                    person_id=face_result.best_match.get('person_id', ''),
                    confidence=similarity,
                    person_name=face_result.best_match.get('name', face_result.best_match.get('person_id', ''))
                )
                face_match.set_match_status(config.unknown_threshold)
                logger.debug(f"Face {face_result.face_id}: Created FaceMatch with similarity {similarity}, is_match = {face_match.is_match}")
                face_result.best_match = face_match
            else:
                logger.debug(f"Face {face_result.face_id}: No best_match to process")
            
            # Times from recognition service are for its own operations
            face_result.recognition_time = recognition_result_dict.get("processing_time", 0.0)
            face_result.embedding_time = recognition_result_dict.get("embedding_time", 0.0)
            face_result.search_time = recognition_result_dict.get("search_time", 0.0)
            
            # Log details of the best match if found
            if face_result.best_match and isinstance(face_result.best_match, dict):
                person_id = face_result.best_match.get('person_id')
                sim = face_result.best_match.get('similarity')
                name = face_result.best_match.get('person_name', person_id)
                logger.debug(
                    f"Face {face_result.face_id} best match: {name} (ID: {person_id}) with sim: {sim:.4f}"
                )
            elif face_result.matches:
                logger.debug(f"Face {face_result.face_id} has {len(face_result.matches)} matches, but no 'best_match' field.")
            else:
                logger.debug(f"No recognition match for face {face_result.face_id}.")

        except Exception as e:
            logger.error(
                f"❌ Error during single face recognition for face_id {face_result.face_id}: {e}",
                exc_info=True,
            )

    async def _create_recognition_only_result(
        self, image: np.ndarray, recognition_output: Dict[str, Any]
    ) -> Optional[Union[FaceResult, List[FaceResult]]]:
        """
        Creates FaceResult(s) from the output of recognize_faces in
        RECOGNITION_ONLY mode.
        """
        if not recognition_output or not recognition_output.get("success"):
            self.logger.warning(
                "Skipping result creation due to unsuccessful recognition output."
            )
            return None

        face_id = recognition_output.get("face_id", f"rec_face_{int(time.time())}")
        bbox_data = recognition_output.get("bbox")
        confidence = recognition_output.get("confidence", 1.0)
        quality_score = recognition_output.get("quality_score", 0.0)
        embedding_data = recognition_output.get("embedding")
        matches_data = recognition_output.get("matches", [])
        best_match_data = recognition_output.get("best_match")
        model_used = recognition_output.get("model_used")
        proc_time = recognition_output.get("processing_time", 0.0)
        emb_time = recognition_output.get("embedding_time", 0.0)
        search_time = recognition_output.get("search_time", 0.0)

        if bbox_data:
            if DETECTION_UTILS_AVAILABLE:
                bbox = BoundingBox(
                    x1=bbox_data[0],
                    y1=bbox_data[1],
                    x2=bbox_data[2],
                    y2=bbox_data[3],
                    confidence=float(
                        recognition_output.get("detection_confidence", confidence)
                    ),
                )
            else:
                bbox = BoundingBox(
                    bbox_data[0], bbox_data[1], bbox_data[2], bbox_data[3],
                    float(recognition_output.get("detection_confidence", confidence))
                )
        else:
            # Placeholder for the whole image if no specific bbox
            h, w = image.shape[:2]
            if DETECTION_UTILS_AVAILABLE:
                bbox = BoundingBox(x1=0, y1=0, x2=w, y2=h, confidence=confidence)
            else:
                bbox = BoundingBox(0, 0, w, h, confidence)

        face_result = FaceResult(
            face_id=face_id,
            bbox=bbox,
            confidence=float(confidence),
            quality_score=float(quality_score),
            embedding=embedding_data,
            matches=matches_data,
            best_match=best_match_data,
            model_used=model_used,
            processing_time=proc_time,
            analysis_metadata={
                "recognition_processing_time": proc_time,
                "embedding_time": emb_time,
                "search_time": search_time,
                "source": "recognition_only_mode",
            },
        )

        return face_result

    def _extract_face_crop(
        self, image: np.ndarray, bbox_data: Any, margin_pixels: int = 10
    ) -> Optional[np.ndarray]:
        """ตัดใบหน้าจากรูปภาพ"""
        try:
            # Ensure bbox_data is BoundingBox or has x1, y1, x2, y2
            if not hasattr(bbox_data, 'x1') or not hasattr(bbox_data, 'y1') or \
               not hasattr(bbox_data, 'x2') or not hasattr(bbox_data, 'y2'):
                self.logger.warning(f"Invalid bbox_data for crop: {bbox_data}")
                return None

            x1, y1, x2, y2 = (
                int(bbox_data.x1),
                int(bbox_data.y1),
                int(bbox_data.x2),
                int(bbox_data.y2),
            )
            img_h, img_w = image.shape[:2]

            # Add margin, ensuring bounds are within image dimensions
            x1_m = max(0, x1 - margin_pixels)
            y1_m = max(0, y1 - margin_pixels)
            x2_m = min(img_w, x2 + margin_pixels)
            y2_m = min(img_h, y2 + margin_pixels)

            if x1_m >= x2_m or y1_m >= y2_m:
                self.logger.warning(
                    f"Invalid crop dimensions after margin: "
                    f"({x1_m},{y1_m}) to ({x2_m},{y2_m})"
                )
                # Fallback to original bbox if margin makes it invalid
                x1_m, y1_m, x2_m, y2_m = x1, y1, x2, y2
                if x1_m >= x2_m or y1_m >= y2_m:  # Still invalid
                    return None

            face_crop = image[y1_m:y2_m, x1_m:x2_m]

            if face_crop.size == 0:
                self.logger.warning(
                    f"Empty face crop extracted for bbox: {bbox_data} "
                    f"from image shape {image.shape}"
                )
                return None
            return face_crop
        except Exception as e:
            self.logger.error(f"❌ Error extracting face crop: {e}", exc_info=True)
            return None

    def _update_stats(self, result: FaceAnalysisResult) -> None:
        """อัปเดตสถิติการทำงาน"""
        self.stats["total_analyses"] += 1
        self.stats["total_faces_detected"] += result.total_faces
        self.stats["total_faces_recognized"] += result.identified_faces
        self.stats["processing_times"].append(result.total_time)
        self.stats["success_rates"].append(1 if result.success else 0)
        if result.detection_time > 0:
            self.stats["detection_times"].append(result.detection_time)
        if result.recognition_time > 0:
            self.stats["recognition_times"].append(result.recognition_time)

    def get_service_info(self) -> Dict[str, Any]:
        """ดึงข้อมูลเกี่ยวกับ service และ model ที่ใช้"""
        detection_info = {}
        if self.face_detection_service:
            if hasattr(self.face_detection_service, 'get_service_info'):
                try:
                    # ตรวจสอบว่าเป็น async method หรือไม่
                    import inspect
                    if inspect.iscoroutinefunction(self.face_detection_service.get_service_info):
                        detection_info = {"async_method": "get_service_info available but async - call with await"}
                    else:
                        detection_info = self.face_detection_service.get_service_info()
                except Exception as e:
                    detection_info = {"error": str(e)}

        recognition_info = {}
        if self.face_recognition_service:
            if hasattr(self.face_recognition_service, 'get_service_info'):
                try:
                    # get_service_info ใน FaceRecognitionService ไม่ได้เป็น async
                    recognition_info = self.face_recognition_service.get_service_info()
                except Exception as e:
                    recognition_info = {"error": str(e)}

        return {
            "service_name": "FaceAnalysisService",
            "version": "1.1.0",
            "description": "Comprehensive face detection and recognition service.",
            "available_modes": [mode.value for mode in AnalysisMode],
            "available_quality_levels": [ql.value for ql in QualityLevel],
            "detection_service": {
                "available": self.face_detection_service is not None,
                "info": detection_info,
            },
            "recognition_service": {
                "available": self.face_recognition_service is not None,
                "info": recognition_info,
            },
            "current_config_defaults": {
                "detection_config": self.config.get("detection", {}),
                "recognition_config": self.config.get("recognition", {}),
                "analysis_config": "AnalysisConfig object required for detailed settings",
            },
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """ดึงสถิติการทำงานของ service"""
        # Calculate average times, handling potential division by zero
        avg_processing_time = (
            sum(self.stats["processing_times"]) / len(self.stats["processing_times"])
            if self.stats["processing_times"]
            else 0
        )
        avg_detection_time = (
            sum(self.stats["detection_times"]) / len(self.stats["detection_times"])
            if self.stats["detection_times"]
            else 0
        )
        avg_recognition_time = (
            sum(self.stats["recognition_times"]) / len(self.stats["recognition_times"])
            if self.stats["recognition_times"]
            else 0
        )
        success_rate = (
            sum(self.stats["success_rates"]) / len(self.stats["success_rates"])
            if self.stats["success_rates"]
            else 0
        )

        detection_model_stats = {}
        if self.face_detection_service and hasattr(self.face_detection_service, 'get_performance_stats'):
            try:
                detection_model_stats = self.face_detection_service.get_performance_stats()
            except Exception as e:
                detection_model_stats = {"error": str(e)}

        recognition_model_stats = {}
        if self.face_recognition_service and hasattr(self.face_recognition_service, 'get_performance_stats'):
            try:
                recognition_model_stats = self.face_recognition_service.get_performance_stats()
            except Exception as e:
                recognition_model_stats = {"error": str(e)}

        return {
            "total_analyses": self.stats["total_analyses"],
            "total_faces_detected": self.stats["total_faces_detected"],
            "total_faces_recognized": self.stats["total_faces_recognized"],
            "average_processing_time_ms": avg_processing_time * 1000,
            "average_detection_time_ms": avg_detection_time * 1000,
            "average_recognition_time_ms": avg_recognition_time * 1000,
            "success_rate_percent": success_rate * 100,
            "detection_model_stats": detection_model_stats,
            "recognition_model_stats": recognition_model_stats,
        }

    async def shutdown(self) -> None:
        """Shutdown the service and release resources."""
        self.logger.info("Shutting down Face Analysis Service...")
        if self.face_detection_service and hasattr(self.face_detection_service, 'shutdown'):
            try:
                await self.face_detection_service.shutdown()
                self.logger.info("Face Detection Service shut down.")
            except Exception as e:
                self.logger.error(f"Error shutting down face detection service: {e}")
        
        if self.face_recognition_service and hasattr(self.face_recognition_service, 'shutdown'):
            try:
                await self.face_recognition_service.shutdown()
                self.logger.info("Face Recognition Service shut down.")
            except Exception as e:
                self.logger.error(f"Error shutting down face recognition service: {e}")
        
        self.logger.info("Face Analysis Service shut down successfully.")