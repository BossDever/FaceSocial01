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

# ruff: noqa: E501, C901, F821, W293  # Disable line length, complexity, undefined name, blank whitespace warnings

import logging

# ruff: noqa: F821
# Use relative import and suppress missing import
from ...core.log_config import get_logger  # type: ignore[reportMissingImports]

from .models import (
    AnalysisConfig,
    FaceResult,
    FaceAnalysisResult,
    AnalysisMode,
    QualityLevel,
)
from ..face_detection.face_detection_service import FaceDetectionService
from ..face_recognition.face_recognition_service import FaceRecognitionService
from ..face_detection.utils import BoundingBox

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
        face_detection_service: Optional[FaceDetectionService] = None,  # Add for DI
        face_recognition_service: Optional[FaceRecognitionService] = None,  # Add for DI
    ):
        self.vram_manager = vram_manager
        self.logger = logger  # Use the imported logger

        # Parse configuration
        self.config = config or {}
        self.config.setdefault("detection", {})
        self.config.setdefault("recognition", {})

        # Sub-services - will be initialized later
        self.face_detection_service = face_detection_service
        self.face_recognition_service = face_recognition_service

        if self.face_detection_service is None:
            logger.info("FaceAnalysisService: FaceDetectionService not provided at init, creating new.")
            # This path should ideally not be taken if using shared services
            # self.face_detection_service = FaceDetectionService(vram_manager, self.config.detection_config_dict)
        
        if self.face_recognition_service is None:
            logger.info("FaceAnalysisService: FaceRecognitionService not provided at init, creating new.")
            # This path should ideally not be taken if using shared services
            # self.face_recognition_service = FaceRecognitionService(vram_manager, self.config.recognition_config_dict)
        
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

        self.logger.info("Face Analysis Service initialized")    def set_shared_services(
        self,
        face_detection_service: FaceDetectionService,
        face_recognition_service: FaceRecognitionService,
    ):
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
            # return False # Optionally prevent startup if services are critical and not set
        else:
            self.logger.info("✅ FaceDetectionService is set.")

        if self.face_recognition_service is None:
            self.logger.warning("⚠️ FaceRecognitionService not set yet (will be injected later)")
            # return False
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
                min_quality_threshold=60.0
                if config.use_quality_based_selection
                else 30.0,
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
                recognition_result_dict = (
                    await self.face_recognition_service.recognize_faces(
                        image_bytes=image_bytes_for_recognition,
                        model_name=config.recognition_model,
                    )
                )

            if recognition_result_dict.get("success"):
                # _create_recognition_only_result expects the image
                # and the dict from recognize_faces
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
            # recognition_model_used remains None

        return faces, time.time() - recognition_start_time, recognition_model_used

    async def analyze_faces(
        self,
        image: np.ndarray,
        config: AnalysisConfig,
        gallery: Optional[Dict[str, Any]] = None, # Add gallery parameter
    ) -> FaceAnalysisResult:
        """
        วิเคราะห์ใบหน้าครบวงจร

        Args:
            image: รูปภาพ (BGR format)
            config: การตั้งค่าการวิเคราะห์
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
        gallery_actually_used = False # Initialize gallery_actually_used

        try:
            # Step 1: Face Detection
            if config.mode in [
                AnalysisMode.DETECTION_ONLY,
                AnalysisMode.FULL_ANALYSIS,
                AnalysisMode.COMPREHENSIVE,
            ]:
                faces, detection_time, detection_model_used = (
                    await self._handle_detection(image, config)
                )

            # Step 2: Face Recognition (modified to pass gallery)
            if (
                config.mode in [AnalysisMode.FULL_ANALYSIS, AnalysisMode.COMPREHENSIVE]
                and faces # Only proceed if faces were detected
            ):
                if gallery and config.enable_gallery_matching:
                    self.logger.info(
                        f"Starting recognition for {len(faces)} faces with provided gallery ({len(gallery)} people)."
                    )
                    gallery_actually_used = True # Set to True when gallery is used
                    # Pass gallery to _handle_recognition
                    recognition_time_val, recognition_model_used_rec = await self._handle_recognition(
                        image, faces, config, gallery
                    )
                    recognition_time += recognition_time_val
                    if recognition_model_used_rec:
                        recognition_model_used = recognition_model_used_rec
                elif config.enable_database_matching:
                    self.logger.info(
                        f"Starting recognition for {len(faces)} faces with internal database."
                    )
                    # Call _handle_recognition without gallery to use internal DB
                    recognition_time_val, recognition_model_used_rec = await self._handle_recognition(
                        image, faces, config, None # Explicitly None for gallery
                    )
                    recognition_time += recognition_time_val
                    if recognition_model_used_rec:
                        recognition_model_used = recognition_model_used_rec
                else:
                    self.logger.info("Recognition skipped: Gallery not provided/enabled, and DB matching not enabled.")

            # Step 3: Handle recognition-only mode (modified to pass gallery)
            elif config.mode == AnalysisMode.RECOGNITION_ONLY:
                if gallery and config.enable_gallery_matching:
                    self.logger.info(f"Processing recognition-only mode with provided gallery ({len(gallery)} people).")
                    gallery_actually_used = True # Set to True when gallery is used
                    # Pass gallery to _handle_recognition_only
                    faces, rec_time, rec_model = await self._handle_recognition_only(
                        image, config, gallery
                    )
                    recognition_time += rec_time
                    if rec_model:
                        recognition_model_used = rec_model
                elif config.enable_database_matching:
                    self.logger.info("Processing recognition-only mode with internal database.")
                     # Call _handle_recognition_only without gallery
                    faces, rec_time, rec_model = await self._handle_recognition_only(
                        image, config, None # Explicitly None for gallery
                    )
                    recognition_time += rec_time
                    if rec_model:
                        recognition_model_used = rec_model
                else:
                    self.logger.info("Recognition-only skipped: Gallery not provided/enabled, and DB matching not enabled.")
                    # Result will have no faces if no recognition is done.

            total_time = time.time() - start_time

            # สร้างผลลัพธ์
            result = FaceAnalysisResult(
                image_shape=image.shape,
                config=config,
                faces=faces,
                detection_time=detection_time,
                recognition_time=recognition_time,
                total_time=total_time,
                detection_model_used=detection_model_used,                recognition_model_used=recognition_model_used,
                analysis_metadata={
                    "quality_level": config.quality_level.value if hasattr(config.quality_level, 'value') else str(config.quality_level),
                    "parallel_processing": config.parallel_processing,
                    "gallery_size": len(gallery) if gallery else 0,
                    "gallery_provided": gallery is not None, # Whether it was passed in
                    "gallery_used_for_matching": gallery_actually_used, # Whether it was used
                    "database_used_for_matching": config.enable_database_matching and not gallery_actually_used,
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

            # Return error result
            return FaceAnalysisResult(
                image_shape=image.shape,
                config=config,
                faces=[],
                detection_time=detection_time,
                recognition_time=recognition_time,
                total_time=total_time,
                error=str(e),
                success=False,
            )

    async def _convert_detection_results(
        self, detection_result, config: AnalysisConfig, image: np.ndarray
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
    ):
        """Executes recognition tasks either in parallel or sequentially."""
        if parallel_processing and len(recognition_tasks) > 1:
            try:
                await asyncio.gather(*recognition_tasks, return_exceptions=True)
            except Exception as e:
                self.logger.error(f"❌ Parallel recognition failed: {e}")
                # Fallback to sequential processing if gather fails
                # (though return_exceptions=True should prevent this)
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
        gallery: Optional[Dict[str, Any]], # Add gallery parameter
    ):
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
            if face_result.face_id is None: # Should not happen if IDs are assigned
                face_result.face_id = f"face_{time.time_ns()}"

            try:
                face_crop_bytes = face_result.get_face_crop_bytes(image, config.recognition_image_format)
                if face_crop_bytes is None: # Try extracting again if not already set
                     face_crop_np = self._extract_face_crop(image, face_result.bbox)
                     if face_crop_np is not None:
                         _, buffer = cv2.imencode(f".{config.recognition_image_format}", face_crop_np)
                         face_crop_bytes = buffer.tobytes()

                if face_crop_bytes:
                    # Pass gallery to _recognize_single_face
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
        gallery: Optional[Dict[str, Any]] = None, # Add gallery parameter
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
                # Call the new method that accepts a gallery
                recognition_result_dict = await self.face_recognition_service.recognize_faces_with_gallery(
                    image_bytes=face_crop_bytes,
                    gallery=gallery,
                    model_name=config.recognition_model,
                )
            elif config.enable_database_matching: # Fallback to internal DB if no gallery or not enabled for gallery
                logger.debug(
                    f"Recognizing face {face_result.face_id} with internal database "
                    f"using model {config.recognition_model}."
                )
                recognition_result_dict = await self.face_recognition_service.recognize_faces(
                    image_bytes=face_crop_bytes, model_name=config.recognition_model
                )
            else:
                logger.debug(f"Recognition skipped for face {face_result.face_id} as no gallery/DB matching enabled.")
                # Populate with minimal data if no recognition was performed but was expected
                face_result.recognition_model = config.recognition_model
                return


            if not isinstance(recognition_result_dict, dict):
                logger.error(
                    f"❌ Recognition service did not return a dict for face {face_result.face_id}. "
                    f"Got: {type(recognition_result_dict)}"
                )
                recognition_result_dict = {} # Fallback

            # Update FaceResult with recognition details
            face_result.query_embedding = recognition_result_dict.get("query_embedding")
            face_result.matches = recognition_result_dict.get("matches", [])
            face_result.best_match = recognition_result_dict.get("best_match")
            face_result.recognition_model = config.recognition_model
            
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
            elif face_result.matches: # Log if matches exist but no single best_match field (e.g. from older service version)
                 logger.debug(f"Face {face_result.face_id} has {len(face_result.matches)} matches, but no 'best_match' field.")
            else:
                logger.debug(f"No recognition match for face {face_result.face_id}.")


        except Exception as e:
            logger.error(
                f"❌ Error during single face recognition for face_id {face_result.face_id}: {e}",
                exc_info=True,
            )
            # Optionally set error state on face_result
            # face_result.error = str(e)

    async def _create_recognition_only_result(
        self, image: np.ndarray, recognition_output: Dict[str, Any]
    ) -> Optional[Union[FaceResult, List[FaceResult]]]:
        """
        Creates FaceResult(s) from the output of recognize_faces in
        RECOGNITION_ONLY mode.
        This method now expects the direct dictionary output from recognize_faces.
        It might return a single FaceResult or a list if multiple faces were
        processed by recognize_faces.
        """
        if not recognition_output or not recognition_output.get("success"):
            self.logger.warning(
                "Skipping result creation due to unsuccessful recognition output."
            )
            return None

        # If recognize_faces processed multiple sub-images/faces and returns a
        # list of results (e.g., if it internally handled multiple detected
        # faces from a single input image bytes)
        # For now, assume recognize_faces returns a single primary result for
        # the given image_bytes. If it can return multiple, this logic needs to adapt.

        face_id = recognition_output.get("face_id", f"rec_face_{int(time.time())}")
        bbox_data = recognition_output.get("bbox") # Might be None
        # Assume high confidence if not provided
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
            # Placeholder for the whole image if no specific bbox
            h, w = image.shape[:2]
            bbox = BoundingBox(x1=0, y1=0, x2=w, y2=h, confidence=confidence)

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
            has_identity=bool(matches_data)
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
            y2_m = min(img_h, y2 + margin_pixels)            if x1_m >= x2_m or y1_m >= y2_m:
                self.logger.warning(
                    f"Invalid crop dimensions after margin: "
                    f"({x1_m},{y1_m}) to ({x2_m},{y2_m})"
                )
                # Fallback to original bbox if margin makes it invalid
                # (though return_exceptions=True should prevent this)
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
            detection_info = self.face_detection_service.get_service_info()

        recognition_info = {}
        if self.face_recognition_service:
            recognition_info = self.face_recognition_service.get_service_info()

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
            },            "current_config_defaults": {
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

        return {
            "total_analyses": self.stats["total_analyses"],
            "total_faces_detected": self.stats["total_faces_detected"],
            "total_faces_recognized": self.stats["total_faces_recognized"],
            "average_processing_time_ms": avg_processing_time * 1000,
            "average_detection_time_ms": avg_detection_time * 1000,
            "average_recognition_time_ms": avg_recognition_time * 1000,
            "success_rate_percent": success_rate * 100,
            "detection_model_stats": self.face_detection_service.get_performance_stats()
            if self.face_detection_service
            else {},
            "recognition_model_stats":
                self.face_recognition_service.get_performance_stats()
                if self.face_recognition_service
                else {},
        }

    async def shutdown(self) -> None:
        """Shutdown the service and release resources."""
        self.logger.info("Shutting down Face Analysis Service...")
        if self.face_detection_service:
            await self.face_detection_service.shutdown()
            self.logger.info("Face Detection Service shut down.")
        if self.face_recognition_service:
            await self.face_recognition_service.shutdown()
            self.logger.info("Face Recognition Service shut down.")
        self.logger.info("Face Analysis Service shut down successfully.")


async def main_test():
    # This is a placeholder for testing the service directly.
    # In a real application, this would be part of a larger system.
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting FaceAnalysisService test...")

    # Mock VRAMManager and config for testing
    class MockVRAMManager:
        def request_vram(self, amount: int, task_id: str) -> bool:
            logger.info(f"VRAM request: {amount}MB for {task_id}")
            return True

        def release_vram(self, amount: int, task_id: str) -> None:
            logger.info(f"VRAM release: {amount}MB for {task_id}")

    mock_vram_manager = MockVRAMManager()
    mock_config = {
        "detection": {
            "preferred_model": "yolov8n_face",
            "confidence_threshold": 0.5,
            "iou_threshold": 0.4,
        },
        "recognition": {
            "preferred_model": "facenet",
            "similarity_threshold": 0.6,
        },
    }

    service = FaceAnalysisService(mock_vram_manager, mock_config)
    initialized = await service.initialize()

    if not initialized:
        logger.error("Service initialization failed. Exiting test.")
        return

    # Example: Load an image (replace with actual image path)
    try:
        # Create a dummy image for testing if no image is available
        test_image_np = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(
            test_image_np,
            "Test Image",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        # Simulate a face for detection (simple rectangle)
        cv2.rectangle(test_image_np, (100, 100), (200, 200), (0, 255, 0), 2)

        logger.info(f"Test image shape: {test_image_np.shape}")

        # Test with DETECTION_ONLY mode
        analysis_config_detection = AnalysisConfig(
            mode=AnalysisMode.DETECTION_ONLY,
            detection_model="yolov8n_face", # Example model
            confidence_threshold=0.5,
        )
        logger.info(
            f"Analyzing with config (detection): {analysis_config_detection.to_dict()}"
        )
        result_detection = await service.analyze_faces(
            test_image_np, analysis_config_detection
        )
        logger.info(f"Detection Result: {result_detection.to_dict()}")

        # Test with FULL_ANALYSIS mode (requires a gallery, mock for now)
        mock_gallery = {
            "person1_embedding_id": {
                "person_id": "person1",
                "name": "Person One",
                # Mock embedding vector (ensure correct dimension for your model)
                "vector": list(np.random.rand(512)),
                "model_type": "facenet", # Match the model used for gallery creation
            }
        }
        analysis_config_full = AnalysisConfig(
            mode=AnalysisMode.FULL_ANALYSIS,
            detection_model="yolov8n_face",
            recognition_model="facenet", # Example model
            enable_gallery_matching=True,
        )
        logger.info(
            f"Analyzing with config (full): {analysis_config_full.to_dict()}"
        )
        result_full = await service.analyze_faces(
            test_image_np, analysis_config_full, gallery=mock_gallery
        )
        logger.info(f"Full Analysis Result: {result_full.to_dict()}")

    except FileNotFoundError:
        logger.error("Test image not found. Please provide a valid image path.")
    except Exception as e:
        logger.error(f"An error occurred during testing: {e}", exc_info=True)
    finally:
        await service.shutdown()
        logger.info("FaceAnalysisService test finished.")


if __name__ == "__main__":
    asyncio.run(main_test())
