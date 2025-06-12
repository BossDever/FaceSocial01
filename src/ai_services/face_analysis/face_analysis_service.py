# cSpell:disable
# mypy: ignore-errors
"""
Face Analysis Service
ระบบวิเคราะห์ใบหน้าแบบครบวงจร (Detection + Recognition)
Enhanced End-to-End Solution with better error handling and performance
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Any, Tuple, Union
import logging
import time
import asyncio

from .models import (
    FaceAnalysisResult,
    FaceResult,
    AnalysisConfig,
    AnalysisMode,
    QualityLevel,
    BoundingBox,
)

logger = logging.getLogger(__name__)


class FaceAnalysisService:
    """
    Enhanced Face Analysis Service
    รวม Face Detection + Face Recognition ในระบบเดียว
    """

    def __init__(self, vram_manager: Any, config: Dict[str, Any]):
        self.vram_manager = vram_manager
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Sub-services - will be initialized later
        self.face_detection_service = None
        self.face_recognition_service = None

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

    async def initialize(self) -> bool:
        """เริ่มต้นระบบ Face Analysis"""
        try:
            self.logger.info("🔧 Initializing Face Analysis Service...")

            # Initialize face detection service
            try:
                from ..face_detection.face_detection_service import FaceDetectionService

                detection_config = self.config.get("detection", {})
                self.face_detection_service = FaceDetectionService(
                    self.vram_manager, detection_config
                )

                detection_init = await self.face_detection_service.initialize()
                if not detection_init:
                    self.logger.error("❌ Failed to initialize face detection")
                    return False
                else:
                    self.logger.info("✅ Face detection service initialized")

            except ImportError as e:
                self.logger.error(f"❌ Face detection service not available: {e}")
                return False

            # Initialize face recognition service
            try:
                from ..face_recognition.face_recognition_service import (
                    FaceRecognitionService,
                )

                recognition_config = self.config.get("recognition", {})
                self.face_recognition_service = FaceRecognitionService(
                    self.vram_manager, recognition_config
                )

                recognition_init = await self.face_recognition_service.initialize()
                if not recognition_init:
                    self.logger.error("❌ Failed to initialize face recognition")
                    return False
                else:
                    self.logger.info("✅ Face recognition service initialized")

            except ImportError as e:
                self.logger.error(f"❌ Face recognition service not available: {e}")
                return False

            self.logger.info("✅ Face Analysis Service ready")
            return True

        except Exception as e:
            self.logger.error(f"❌ Face Analysis Service initialization failed: {e}")
            return False

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
        self, image: np.ndarray, config: AnalysisConfig, gallery: Dict[str, Any]
    ) -> Tuple[List[FaceResult], float, Optional[str]]:
        """Handles the recognition-only mode."""
        recognition_start_time = time.time()
        faces: List[FaceResult] = []
        recognition_model_used = None

        if not self.face_recognition_service:
            raise RuntimeError("Face recognition service not available")

        try:
            # Convert np.ndarray to bytes for recognize_faces
            is_success, buffer = cv2.imencode(".jpg", image) # Assuming image is BGR
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
                    config.recognition_model.value
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
        gallery: Optional[Dict[str, Any]] = None,
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

            # Step 2: Face Recognition
            if (
                config.mode in [AnalysisMode.FULL_ANALYSIS, AnalysisMode.COMPREHENSIVE]
                and gallery
                and config.enable_gallery_matching
                and faces
            ):
                recognition_time, recognition_model_used_rec = (
                    await self._handle_recognition(image, faces, config, gallery)
                )
                # Prioritize recognition model if available
                if recognition_model_used_rec:
                    recognition_model_used = recognition_model_used_rec

            # Step 3: Handle recognition-only mode
            elif config.mode == AnalysisMode.RECOGNITION_ONLY and gallery:
                faces, recognition_time, recognition_model_used = (
                    await self._handle_recognition_only(image, config, gallery)
                )

            total_time = time.time() - start_time

            # สร้างผลลัพธ์
            result = FaceAnalysisResult(
                image_shape=image.shape,
                config=config,
                faces=faces,
                detection_time=detection_time,
                recognition_time=recognition_time,
                total_time=total_time,
                detection_model_used=detection_model_used,
                recognition_model_used=recognition_model_used,
                analysis_metadata={
                    "quality_level": config.quality_level.value,
                    "parallel_processing": config.parallel_processing,
                    "gallery_size": len(gallery) if gallery else 0,
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
        gallery: Dict[str, Any],
    ):
        """ประมวลผล Face Recognition สำหรับหลายใบหน้า"""
        if not self.face_recognition_service:
            return

        recognition_tasks = []
        for face_result in faces:
            try:
                face_crop = face_result.face_crop
                if face_crop is None:
                    face_crop = self._extract_face_crop(image, face_result.bbox)

                if face_crop is not None:
                    task = self._recognize_single_face(
                        face_result, face_crop, config, gallery
                    )
                    # Ensure it's a task
                    recognition_tasks.append(asyncio.create_task(task))
                else:
                    self.logger.warning("⚠️ Failed to extract face crop for recognition")
            except Exception as e:
                self.logger.warning(f"⚠️ Error preparing face for recognition: {e}")
                continue

        if recognition_tasks:
            await self._execute_recognition_tasks(
                recognition_tasks, config.parallel_processing
            )

    async def _recognize_single_face(
        self,
        face_result: FaceResult,
        face_crop: np.ndarray,
        config: AnalysisConfig,
        gallery: Dict[str, Any],
    ):
        """จดจำใบหน้าเดี่ยว"""
        try:
            # Convert face_crop (np.ndarray) to bytes for recognize_faces
            is_success, buffer = cv2.imencode(".jpg", face_crop)
            if not is_success:
                self.logger.error("Failed to encode face_crop for recognition")
                return
            face_crop_bytes = buffer.tobytes()

            recognition_result_dict = await (
                self.face_recognition_service.recognize_faces(
                    image_bytes=face_crop_bytes, model_name=config.recognition_model
                )
            )

            # Ensure recognition_result_dict is a dictionary
            if not isinstance(recognition_result_dict, dict):
                logger.error(
                    f"❌ Recognition service did not return a dict. "
                    f"Got: {type(recognition_result_dict)}"
                )
                # Fallback to a default dict structure or handle error appropriately
                recognition_result_dict = {}


            # Update FaceResult with recognition details
            face_result.query_embedding = recognition_result_dict.get("query_embedding")
            face_result.matches = recognition_result_dict.get("matches", [])
            face_result.best_match = recognition_result_dict.get("best_match")
            face_result.recognition_model = config.recognition_model
            face_result.recognition_time = recognition_result_dict.get(
                "processing_time", 0.0
            )
            face_result.embedding_time = recognition_result_dict.get(
                "embedding_time", 0.0
            )

        except Exception as e:
            self.logger.error(
                f"❌ Error recognizing single face ({face_result.face_id}): {e}",
                exc_info=True,
            )
            # Ensure metadata is initialized even on error
            if not face_result.analysis_metadata:
                face_result.analysis_metadata = {}
            face_result.analysis_metadata["recognition_error"] = str(e)

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
            y2_m = min(img_h, y2 + margin_pixels)

            if x1_m >= x2_m or y1_m >= y2_m:
                self.logger.warning(
                    f"Invalid crop dimensions after margin: "
                    f"({x1_m},{y1_m}) to ({x2_m},{y2_m})"
                )
                # Fallback to original bbox if margin makes it invalid
                # (though return_exceptions=True should prevent this)
                x1_m, y1_m, x2_m, y2_m = x1, y1, x2, y2
                if x1_m >= x2_m or y1_m >= y2_m: # Still invalid
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
        detection_models = []
        if self.face_detection_service:
            detection_models = (
                self.face_detection_service.get_available_models()
            ) # Use the correct method

        recognition_models = []
        if self.face_recognition_service:
            recognition_models = (
                self.face_recognition_service.get_available_models()
            ) # Use the correct method

        return {
            "service_name": "FaceAnalysisService",
            "version": "1.1.0", # Example version
            "description": "Comprehensive face detection and recognition service.",
            "available_modes": [mode.value for mode in AnalysisMode],
            "available_quality_levels": [ql.value for ql in QualityLevel],
            "detection_service": {
                "available": self.face_detection_service is not None,
                "models": detection_models,
            },
            "recognition_service": {
                "available": self.face_recognition_service is not None,
                "models": recognition_models,
            },
            "current_config_defaults": self.config, # Assuming config is serializable
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
