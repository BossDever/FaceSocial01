# cSpell:disable
# mypy: ignore-errors
"""
Face Analysis Service
ระบบวิเคราะห์ใบหน้าแบบครบวงจร (Detection + Recognition)
Enhanced End-to-End Solution with better error handling and performance
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Any
import logging
import time
import asyncio

from .models import (
    FaceAnalysisResult, FaceResult, AnalysisConfig, 
    AnalysisMode, BatchAnalysisResult, QualityLevel
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
            'total_analyses': 0,
            'total_faces_detected': 0,
            'total_faces_recognized': 0,
            'processing_times': [],
            'success_rates': [],
            'detection_times': [],
            'recognition_times': []
        }
        
        self.logger.info("Face Analysis Service initialized")
    
    async def initialize(self) -> bool:
        """เริ่มต้นระบบ Face Analysis"""
        try:
            self.logger.info("🔧 Initializing Face Analysis Service...")
            
            # Initialize face detection service
            try:
                from ..face_detection.face_detection_service import FaceDetectionService
                detection_config = self.config.get('detection', {})
                self.face_detection_service = FaceDetectionService(self.vram_manager, detection_config)
                
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
                from ..face_recognition.face_recognition_service import FaceRecognitionService
                recognition_config = self.config.get('recognition', {})
                self.face_recognition_service = FaceRecognitionService(self.vram_manager, recognition_config)
                
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
    
    async def analyze_faces(self,
                           image: np.ndarray,
                           config: AnalysisConfig,
                           gallery: Optional[Dict[str, Any]] = None) -> FaceAnalysisResult:
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
        
        try:
            faces = []
            detection_model_used = None
            recognition_model_used = None
            
            # Step 1: Face Detection (ถ้าต้องการ)
            if config.mode in [AnalysisMode.DETECTION_ONLY, AnalysisMode.FULL_ANALYSIS, AnalysisMode.COMPREHENSIVE]:
                if not self.face_detection_service:
                    raise RuntimeError("Face detection service not available")
                    
                detection_start = time.time()
                
                try:
                    detection_result = await self.face_detection_service.detect_faces(
                        image,
                        model_name=config.detection_model,
                        conf_threshold=config.confidence_threshold,
                        min_face_size=config.min_face_size,
                        max_faces=config.max_faces,
                        return_landmarks=True,
                        min_quality_threshold=60.0 if config.use_quality_based_selection else 30.0
                    )
                    
                    detection_time = time.time() - detection_start
                    detection_model_used = detection_result.model_used
                    
                    self.logger.info(f"Detection: {len(detection_result.faces)} faces in {detection_time:.3f}s")
                    
                    # แปลง detection results เป็น FaceResult
                    faces = await self._convert_detection_results(detection_result, config, image)
                    
                except Exception as det_error:
                    self.logger.error(f"❌ Detection failed: {det_error}")
                    detection_time = time.time() - detection_start
                    # Continue with empty faces list
            
            # Step 2: Face Recognition (ถ้าต้องการ)
            if (config.mode in [AnalysisMode.FULL_ANALYSIS, AnalysisMode.COMPREHENSIVE] and 
                gallery and config.enable_gallery_matching and faces):
                
                if not self.face_recognition_service:
                    self.logger.warning("Face recognition service not available")
                else:
                    recognition_start = time.time()
                    
                    try:
                        # ประมวลผล recognition สำหรับใบหน้าที่มีคุณภาพดี
                        quality_threshold = 60.0 if config.use_quality_based_selection else 0.0
                        processable_faces = [f for f in faces if f.quality_score >= quality_threshold]
                        
                        if processable_faces:
                            await self._process_recognition_for_faces(
                                image, processable_faces, config, gallery
                            )
                            
                            # Count successful recognitions
                            recognized_faces = [f for f in processable_faces if f.has_identity]
                            recognition_model_used = config.recognition_model
                            
                            self.logger.info(f"Recognition: {len(recognized_faces)}/{len(processable_faces)} faces recognized")
                        
                    except Exception as rec_error:
                        self.logger.error(f"❌ Recognition failed: {rec_error}")
                    
                    recognition_time = time.time() - recognition_start
            
            # Step 3: Handle recognition-only mode
            elif config.mode == AnalysisMode.RECOGNITION_ONLY:
                if not self.face_recognition_service:
                    raise RuntimeError("Face recognition service not available")
                
                # For recognition-only, assume the entire image is a face
                recognition_start = time.time()
                
                try:
                    if gallery:
                        recognition_result = await self.face_recognition_service.recognize_face(
                            image, gallery, config.recognition_model, config.gallery_top_k
                        )
                        
                        # Create a single face result for the entire image
                        face_result = await self._create_recognition_only_result(image, recognition_result)
                        faces = [face_result] if face_result else []
                        recognition_model_used = recognition_result.model_used.value if recognition_result.model_used else None
                        
                except Exception as rec_error:
                    self.logger.error(f"❌ Recognition-only failed: {rec_error}")
                
                recognition_time = time.time() - recognition_start
            
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
                    'quality_level': config.quality_level.value,
                    'parallel_processing': config.parallel_processing,
                    'gallery_size': len(gallery) if gallery else 0
                }
            )
            
            # อัปเดต statistics
            self._update_stats(result)
            
            self.logger.info(f"Analysis complete: {result.total_faces} faces, "
                           f"{result.identified_faces} identified in {total_time:.3f}s")
            
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
                success=False
            )
    
    async def _convert_detection_results(self, detection_result, config: AnalysisConfig, image: np.ndarray) -> List[FaceResult]:
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
                    landmarks=getattr(detected_face, 'landmarks', None)
                )
                
                # ตัดใบหน้าถ้าต้องการ
                if config.return_face_crops or config.mode in [AnalysisMode.FULL_ANALYSIS, AnalysisMode.COMPREHENSIVE]:
                    face_crop = self._extract_face_crop(image, detected_face.bbox)
                    if config.return_face_crops:
                        face_result.face_crop = face_crop
                
                faces.append(face_result)
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to convert detection result {i}: {e}")
                continue
        
        return faces
    
    async def _process_recognition_for_faces(self, image: np.ndarray, faces: List[FaceResult], 
                                           config: AnalysisConfig, gallery: Dict[str, Any]):
        """ประมวลผล Face Recognition สำหรับหลายใบหน้า"""
        if not self.face_recognition_service:
            return
        
        recognition_tasks = []
        
        for face_result in faces:
            try:
                # ตัดใบหน้า
                face_crop = face_result.face_crop
                if face_crop is None:
                    face_crop = self._extract_face_crop(image, face_result.bbox)
                
                if face_crop is not None:
                    # สร้าง task สำหรับ recognition
                    task = self._recognize_single_face(face_result, face_crop, config, gallery)
                    recognition_tasks.append(task)
                else:
                    self.logger.warning("⚠️ Failed to extract face crop for recognition")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Error preparing face for recognition: {e}")
                continue
        
        # ประมวลผลแบบ parallel ถ้าเปิดใช้งาน
        if config.parallel_processing and len(recognition_tasks) > 1:
            try:
                await asyncio.gather(*recognition_tasks, return_exceptions=True)
            except Exception as e:
                self.logger.error(f"❌ Parallel recognition failed: {e}")
                # Fallback to sequential processing
                for task in recognition_tasks:
                    try:
                        await task
                    except Exception as task_error:
                        self.logger.warning(f"⚠️ Recognition task failed: {task_error}")
        else:
            # Sequential processing
            for task in recognition_tasks:
                try:
                    await task
                except Exception as task_error:
                    self.logger.warning(f"⚠️ Recognition task failed: {task_error}")
    
    async def _recognize_single_face(self, face_result: FaceResult, face_crop: np.ndarray,
                                   config: AnalysisConfig, gallery: Dict[str, Any]):
        """จดจำใบหน้าเดี่ยว"""
        try:
            recognition_result = await self.face_recognition_service.recognize_face(
                face_crop, gallery, config.recognition_model, config.gallery_top_k
            )
            
            # อัปเดต face_result ด้วยผลลัพธ์ recognition
            if recognition_result.query_embedding:
                face_result.embedding = recognition_result.query_embedding
            
            if recognition_result.matches:
                face_result.matches = recognition_result.matches
            
            if recognition_result.best_match:
                face_result.best_match = recognition_result.best_match
            
            # เพิ่ม metadata
            if not face_result.analysis_metadata:
                face_result.analysis_metadata = {}
            
            face_result.analysis_metadata.update({
                'recognition_processing_time': recognition_result.processing_time,
                'embedding_time': recognition_result.embedding_time,
                'search_time': recognition_result.search_time,
                'total_candidates': recognition_result.total_candidates
            })
            
        except Exception as e:
            self.logger.error(f"❌ Single face recognition failed: {e}")
    
    async def _create_recognition_only_result(self, image: np.ndarray, recognition_result) -> Optional[FaceResult]:
        """สร้าง FaceResult สำหรับ recognition-only mode"""
        try:
            if not recognition_result:
                return None
            
            # Create a dummy bbox for the entire image
            h, w = image.shape[:2]
            
            # Import BoundingBox if needed
            try:
                from ..face_detection.utils import BoundingBox
                dummy_bbox = BoundingBox(x1=0, y1=0, x2=w, y2=h, confidence=1.0)
            except ImportError:
                # Create a simple bbox-like object
                class DummyBBox:
                    def __init__(self):
                        self.x1, self.y1, self.x2, self.y2 = 0, 0, w, h
                        self.confidence = 1.0
                dummy_bbox = DummyBBox()
            
            face_result = FaceResult(
                bbox=dummy_bbox,
                confidence=1.0,
                quality_score=recognition_result.query_embedding.quality_score if recognition_result.query_embedding else 50.0,
                face_id="face_001",
                embedding=recognition_result.query_embedding,
                matches=recognition_result.matches,
                best_match=recognition_result.best_match,
                processing_time=recognition_result.processing_time,
                model_used=recognition_result.model_used.value if recognition_result.model_used else "unknown"
            )
            
            return face_result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create recognition-only result: {e}")
            return None
    
    def _extract_face_crop(self, image: np.ndarray, bbox) -> Optional[np.ndarray]:
        """ตัดใบหน้าจากรูปภาพ"""
        try:
            # ขยาย bbox เล็กน้อยเพื่อให้ได้ context
            h, w = image.shape[:2]
            
            # คำนวณการขยาย (15% ของขนาดหน้า)
            face_w = bbox.x2 - bbox.x1
            face_h = bbox.y2 - bbox.y1
            
            expand_w = int(face_w * 0.15)
            expand_h = int(face_h * 0.15)
            
            # ขยาย bbox
            x1 = max(0, int(bbox.x1 - expand_w))
            y1 = max(0, int(bbox.y1 - expand_h))
            x2 = min(w, int(bbox.x2 + expand_w))
            y2 = min(h, int(bbox.y2 + expand_h))
            
            # ตัดใบหน้า
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return None
            
            # แปลงจาก BGR เป็น RGB สำหรับ face recognition
            if len(face_crop.shape) == 3:
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_crop
            
            return face_rgb
            
        except Exception as e:
            self.logger.error(f"❌ Failed to extract face crop: {e}")
            return None
    
    async def compare_faces(self,
                           face_image1: np.ndarray,
                           face_image2: np.ndarray,
                           model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        เปรียบเทียบใบหน้า 2 ใบ
        """
        start_time = time.time()
        
        try:
            if not self.face_recognition_service:
                raise RuntimeError("Face recognition service not available")
            
            # สกัด embeddings
            embedding1 = await self.face_recognition_service.extract_embedding(face_image1, model_name)
            embedding2 = await self.face_recognition_service.extract_embedding(face_image2, model_name)
            
            if not embedding1 or not embedding2:
                return {
                    'success': False,
                    'error': 'Failed to extract embeddings',
                    'processing_time': time.time() - start_time
                }
            
            # เปรียบเทียบ
            comparison_result = self.face_recognition_service.compare_faces(
                embedding1.vector,
                embedding2.vector,
                embedding1.model_used
            )
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'comparison': comparison_result.to_dict(),
                'embedding1': embedding1.to_dict(),
                'embedding2': embedding2.to_dict(),
                'processing_time': processing_time
            }
            
        except Exception as e:
            self.logger.error(f"❌ Face comparison failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    async def batch_analyze(self,
                           images: List[np.ndarray],
                           config: AnalysisConfig,
                           gallery: Optional[Dict[str, Any]] = None) -> BatchAnalysisResult:
        """
        วิเคราะห์หลายรูปพร้อมกัน
        """
        start_time = time.time()
        
        try:
            # สร้าง tasks สำหรับแต่ละรูป
            analysis_tasks = []
            for i, image in enumerate(images):
                task = self.analyze_faces(image, config, gallery)
                analysis_tasks.append(task)
            
            # ประมวลผลแบบ parallel
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # กรองผลลัพธ์ที่สำเร็จ
            valid_results: List[FaceAnalysisResult] = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.warning(f"⚠️ Analysis failed for image {i}: {result}")
                else:
                    valid_results.append(result)
            
            # สรุปผลลัพธ์
            total_faces = sum(len(result.faces) for result in valid_results)
            
            # นับ unique identities
            all_identities = set()
            for result in valid_results:
                for face in result.faces:
                    if face.has_identity:
                        all_identities.add(face.identity)
            
            processing_time = time.time() - start_time
            
            return BatchAnalysisResult(
                results=valid_results,
                total_images=len(images),
                total_faces=total_faces,
                total_identities=len(all_identities),
                processing_time=processing_time,
                batch_metadata={
                    'config_used': config.to_dict(),
                    'gallery_size': len(gallery) if gallery else 0,
                    'parallel_processing': config.parallel_processing
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Batch analysis failed: {e}")
            return BatchAnalysisResult(
                results=[],
                total_images=len(images),
                total_faces=0,
                total_identities=0,
                processing_time=time.time() - start_time,
                batch_metadata={'error': str(e)}
            )
    
    async def detect_and_recognize(self,
                                  image: np.ndarray,
                                  known_faces: Dict[str, np.ndarray],
                                  config: Optional[AnalysisConfig] = None) -> FaceAnalysisResult:
        """
        ตรวจจับและจดจำใบหน้าในขั้นตอนเดียว (Simplified API)
        """
        if config is None:
            config = AnalysisConfig(
                mode=AnalysisMode.FULL_ANALYSIS,
                enable_gallery_matching=True,
                use_quality_based_selection=True,
                quality_level=QualityLevel.BALANCED
            )
        
        # แปลง known_faces เป็น gallery format
        gallery = {}
        for identity_id, embedding in known_faces.items():
            gallery[identity_id] = {
                'name': identity_id,
                'embeddings': [embedding]
            }
        
        return await self.analyze_faces(image, config, gallery)
    
    def _update_stats(self, result: FaceAnalysisResult):
        """อัปเดต statistics"""
        self.stats['total_analyses'] += 1
        self.stats['total_faces_detected'] += result.total_faces
        self.stats['total_faces_recognized'] += result.identified_faces
        self.stats['processing_times'].append(result.total_time)
        self.stats['detection_times'].append(result.detection_time)
        self.stats['recognition_times'].append(result.recognition_time)
        
        if result.usable_faces > 0:
            success_rate = result.identified_faces / result.usable_faces
            self.stats['success_rates'].append(success_rate)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """ดึงสถิติประสิทธิภาพ"""
        stats = self.stats.copy()
        
        if self.stats['processing_times']:
            stats['average_processing_time'] = np.mean(self.stats['processing_times'])
            stats['total_processing_time'] = sum(self.stats['processing_times'])
            
        if self.stats['detection_times']:
            stats['average_detection_time'] = np.mean(self.stats['detection_times'])
            
        if self.stats['recognition_times']:
            stats['average_recognition_time'] = np.mean(self.stats['recognition_times'])
            
        if self.stats['success_rates']:
            stats['average_success_rate'] = np.mean(self.stats['success_rates'])
        
        # Add sub-service stats
        if self.face_detection_service:
            stats['detection_service_stats'] = self.face_detection_service.get_performance_stats()
        
        if self.face_recognition_service:
            stats['recognition_service_stats'] = self.face_recognition_service.get_performance_stats().to_dict()
        
        return stats
    
    async def get_available_models(self) -> Dict[str, Any]:
        """ดึงรายการโมเดลที่มีอยู่"""
        try:
            available_vram = await self.vram_manager.get_available_memory() if self.vram_manager else 0
            
            result = {
                'available_vram_mb': available_vram / (1024 * 1024) if available_vram else 0,
                'detection_models': {},
                'recognition_models': {},
                'recommendations': {}
            }
            
            # Detection models
            if self.face_detection_service:
                service_info = await self.face_detection_service.get_service_info()
                result['detection_models'] = service_info.get('model_info', {})
            
            # Recognition models
            if self.face_recognition_service:
                # Add model information
                result['recognition_models'] = {
                    'facenet': {'embedding_size': 512, 'input_size': [160, 160]},
                    'adaface': {'embedding_size': 512, 'input_size': [112, 112]},
                    'arcface': {'embedding_size': 512, 'input_size': [112, 112]}
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error getting available models: {e}")
            return {'error': str(e)}
    
    async def switch_models(self,
                           detection_model: Optional[str] = None,
                           recognition_model: Optional[str] = None) -> Dict[str, bool]:
        """เปลี่ยนโมเดลที่ใช้งาน"""
        results = {}
        
        try:
            if detection_model and self.face_detection_service:
                # Detection models are handled differently - they're selected dynamically
                results['detection'] = True  # Placeholder
                self.logger.info(f"Detection model preference set to: {detection_model}")
            elif detection_model:
                results['detection'] = False
            
            if recognition_model and self.face_recognition_service:
                success = await self.face_recognition_service.switch_model(recognition_model)
                results['recognition'] = success
            elif recognition_model:
                results['recognition'] = False
                
        except Exception as e:
            self.logger.error(f"❌ Error switching models: {e}")
            if detection_model:
                results['detection'] = False
            if recognition_model:
                results['recognition'] = False
        
        return results
    
    async def cleanup(self):
        """ทำความสะอาดทรัพยากร"""
        try:
            self.logger.info("🧹 Cleaning up Face Analysis Service...")
            
            if self.face_detection_service:
                await self.face_detection_service.cleanup()
                self.logger.info("✅ Face detection service cleaned up")
            
            if self.face_recognition_service:
                await self.face_recognition_service.cleanup()
                self.logger.info("✅ Face recognition service cleaned up")
            
            # Clear statistics
            self.stats = {
                'total_analyses': 0,
                'total_faces_detected': 0,
                'total_faces_recognized': 0,
                'processing_times': [],
                'success_rates': [],
                'detection_times': [],
                'recognition_times': []
            }
            
            self.logger.info("✅ Face Analysis Service cleaned up")
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")
    
    def create_gallery_from_embeddings(self, 
                                     embeddings_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        สร้าง Gallery จาก embeddings dictionary
        """
        gallery = {}
        for identity_id, data in embeddings_dict.items():
            gallery[identity_id] = {
                'name': data.get('name', identity_id),
                'embeddings': data.get('embeddings', [])
            }
        return gallery