"""
Enhanced Face Recognition Service with GPU Optimization
ระบบจดจำใบหน้าที่ปรับปรุงแล้วพร้อม GPU optimization และ Multi-model support
"""

import logging
import numpy as np
import cv2
import os
import time
from typing import Optional, Dict, Any, List, Union, cast

# Conditional imports
try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None  # type: ignore

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

from .models import (
    FaceEmbedding,
    FaceMatch,
    FaceRecognitionResult,
    FaceComparisonResult,
    RecognitionModel,
    ModelPerformanceStats,
    RecognitionConfig,
    FaceGallery,
)

logger = logging.getLogger(__name__)


class FaceRecognitionService:
    """Enhanced Face Recognition Service with Multi-model Support"""

    def __init__(
        self, vram_manager: Any = None, config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.vram_manager = vram_manager
        self.logger = logging.getLogger(__name__)

        # Parse configuration
        if config is None:
            config = {}

        self.config = RecognitionConfig(
            preferred_model=RecognitionModel(config.get("preferred_model", "facenet")),
            similarity_threshold=config.get("similarity_threshold", 0.60),
            unknown_threshold=config.get("unknown_threshold", 0.55),
            embedding_dimension=config.get("embedding_dimension", 512),
            enable_gpu_optimization=config.get("enable_gpu_optimization", True),
            batch_size=config.get("batch_size", 8),
            quality_threshold=config.get("quality_threshold", 0.2),
        )

        # Model management
        self.current_model = None
        self.current_model_type = self.config.preferred_model
        self.models_cache: Dict[str, Any] = {}

        # Face database - compatible with existing system
        self.face_database: Dict[str, List[FaceEmbedding]] = {}

        # Performance tracking
        self.stats = ModelPerformanceStats()

        # Model configurations with explicit types
        self.model_configs: Dict[
            RecognitionModel, Dict[str, Union[str, tuple, list, int]]
        ] = {
            RecognitionModel.FACENET: {
                "model_path": "model/face-recognition/facenet_vggface2.onnx",
                "input_size": (160, 160),
                "mean": [127.5, 127.5, 127.5],
                "std": [128.0, 128.0, 128.0],
                "embedding_size": 512,
            },
            RecognitionModel.ADAFACE: {
                "model_path": "model/face-recognition/adaface_ir101.onnx",
                "input_size": (112, 112),
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
                "embedding_size": 512,
            },
            RecognitionModel.ARCFACE: {
                "model_path": "model/face-recognition/arcface_r100.onnx",
                "input_size": (112, 112),
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
                "embedding_size": 512,
            },
        }

        self.logger.info("🚀 Enhanced Face Recognition Service initialized")

    async def initialize(self) -> bool:
        """Initialize the face recognition service"""
        try:
            self.logger.info("🔧 Initializing Face Recognition Service...")

            if not ONNX_AVAILABLE:
                self.logger.error("❌ ONNX Runtime not available")
                return False

            # Load default model
            success = await self.load_model(self.current_model_type)
            if not success:
                self.logger.error("❌ Failed to load default model")
                return False

            # Warm up model
            await self._warmup_model()

            self.logger.info("✅ Face Recognition Service initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error initializing service: {e}")
            return False

    async def load_model(self, model_type: RecognitionModel) -> bool:
        """Load face recognition model with GPU optimization"""
        try:
            # Check if model is already loaded
            if self.current_model_type == model_type and self.current_model is not None:
                return True

            # Clean up previous model
            if self.current_model is not None:
                del self.current_model
                self.current_model = None
                if TORCH_AVAILABLE and torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Get model configuration
            model_config = self.model_configs.get(model_type)
            if not model_config:
                self.logger.error(f"❌ Unknown model type: {model_type}")
                return False

            # Type-safe access to model path
            model_path = cast(str, model_config["model_path"])
            if not os.path.exists(model_path):
                self.logger.error(f"❌ Model file not found: {model_path}")
                return False

            self.logger.info(f"🔄 Loading {model_type.value} model from: {model_path}")

            # Configure ONNX session options
            if not ort:
                self.logger.error("❌ ONNX Runtime not available")
                return False

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            )
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            # Configure providers
            providers: List[Union[str, tuple]] = []
            device_used = "CPU"            if (
                self.config.enable_gpu_optimization
                and TORCH_AVAILABLE
                and torch
                and torch.cuda.is_available()
            ):
                try:
                    # GPU memory configuration
                    gpu_mem_limit = int(2 * 1024 * 1024 * 1024)  # 2GB

                    cuda_options = {
                        "device_id": 0,
                        "arena_extend_strategy": "kSameAsRequested",
                        "gpu_mem_limit": gpu_mem_limit,
                        "cudnn_conv_algo_search": "HEURISTIC",
                    }

                    providers.append(("CUDAExecutionProvider", cuda_options))
                    device_used = "GPU"
                    self.logger.info(
                        f"🔥 Configured {model_type.value} for GPU with "
                        f"{gpu_mem_limit / 1024 / 1024:.0f}MB limit"
                    )

                except Exception as cuda_error:
                    self.logger.warning(f"⚠️ CUDA setup failed: {cuda_error}")

            providers.append("CPUExecutionProvider")

            # Create inference session
            self.current_model = ort.InferenceSession(
                model_path, providers=providers, sess_options=session_options
            )

            self.current_model_type = model_type

            # Log success
            if self.current_model:
                actual_providers = self.current_model.get_providers()
                if "CUDAExecutionProvider" in actual_providers:
                    device_used = "GPU"
                else:
                    device_used = "CPU"

                self.logger.info(
                    f"✅ {model_type.value} loaded successfully on {device_used}"
                )
                self.logger.info(f"   Input: {self.current_model.get_inputs()[0].name}")
                self.logger.info(
                    f"   Output: {self.current_model.get_outputs()[0].name}"
                )

            # Update VRAM allocation if needed
            if self.vram_manager:
                await self.vram_manager.request_model_allocation(
                    f"{model_type.value}-face-recognition",
                    "high",
                    "face_recognition_service",
                )

            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to load {model_type.value}: {e}")
            return False

    async def _warmup_model(self) -> None:
        """Warm up model for optimal performance"""
        try:
            if self.current_model is None:
                return

            self.logger.info(f"🔥 Warming up {self.current_model_type.value} model...")

            # Get model configuration
            model_config = self.model_configs[self.current_model_type]
            input_size = cast(tuple, model_config["input_size"])

            # Create dummy input
            dummy_input = np.random.randn(1, 3, input_size[1], input_size[0]).astype(
                np.float32
            )
            input_name = self.current_model.get_inputs()[0].name

            # Run warmup iterations
            warmup_start = time.time()
            for i in range(5):
                try:
                    _ = self.current_model.run(None, {input_name: dummy_input})
                    self.logger.debug(f"🔥 Warmup iteration {i + 1} successful")
                except Exception as warmup_error:
                    self.logger.warning(
                        f"⚠️ Warmup iteration {i + 1} failed: {warmup_error}"
                    )

            warmup_time = time.time() - warmup_start
            self.logger.info(f"🔥 Model warmed up in {warmup_time:.3f}s")

        except Exception as e:
            self.logger.warning(f"⚠️ Model warmup failed: {e}")

    def _preprocess_image(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess face image for model inference"""
        try:
            if face_image is None or face_image.size == 0:
                self.logger.error("❌ Invalid face image")
                return None

            # Get model configuration
            model_config = self.model_configs[self.current_model_type]
            target_size = cast(tuple, model_config["input_size"])
            mean = np.array(cast(list, model_config["mean"]))
            std = np.array(cast(list, model_config["std"]))

            # Ensure image is in correct format
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                # Convert BGR to RGB if needed
                if face_image.max() > 1.1:  # Likely BGR format
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            elif len(face_image.shape) == 2:
                # Convert grayscale to RGB
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            else:
                self.logger.error(f"❌ Unsupported image format: {face_image.shape}")
                return None

            # Resize image with proper type conversion
            face_resized = cv2.resize(
                face_image, target_size, interpolation=cv2.INTER_LANCZOS4
            )

            # Normalize
            face_normalized = face_resized.astype(np.float32)

            if self.current_model_type == RecognitionModel.FACENET:
                # FaceNet uses different normalization
                face_normalized = (face_normalized - 127.5) / 128.0
            else:
                # AdaFace and ArcFace
                face_normalized = face_normalized / 255.0
                face_normalized = (face_normalized - mean) / std

            # Convert to NCHW format
            face_normalized = np.transpose(face_normalized, (2, 0, 1))
            face_normalized = np.expand_dims(face_normalized, axis=0)

            # แก้ไขปัญหา: ให้ return เป็น np.ndarray แทนที่จะเป็น Any
            return np.asarray(face_normalized, dtype=np.float32)

        except Exception as e:
            self.logger.error(f"❌ Image preprocessing failed: {e}")
            return None

    def _calculate_embedding_quality(self, embedding: np.ndarray) -> float:
        """Calculate embedding quality score"""
        try:
            # Vectorized operations for speed
            magnitude = np.linalg.norm(embedding)
            variance = np.var(embedding)
            sparsity = np.sum(np.abs(embedding) < 0.01) / len(embedding)

            # Quality metrics
            magnitude_score = min(float(magnitude), 1.0)
            variance_score = min(float(variance) * 10, 1.0)
            sparsity_score = max(0.0, 1.0 - float(sparsity))

            # Combine scores
            quality = (
                magnitude_score * 0.4 + variance_score * 0.3 + sparsity_score * 0.3
            )
            return float(np.clip(quality * 100, 0.0, 100.0))

        except Exception:
            return 50.0

    async def extract_embedding(
        self, face_image: np.ndarray, model_name: Optional[str] = None
    ) -> Optional[FaceEmbedding]:
        """Extract face embedding with enhanced error handling"""
        start_time = time.time()

        try:
            # Switch model if requested
            if model_name and model_name != self.current_model_type.value:
                try:
                    target_model = RecognitionModel(model_name.lower())
                    await self.load_model(target_model)
                except ValueError:
                    self.logger.warning(
                        f"⚠️ Unknown model: {model_name}, using current model"
                    )

            # Ensure model is loaded
            if self.current_model is None:
                if not await self.initialize():
                    return None

            # Preprocess image
            input_tensor = self._preprocess_image(face_image)
            if input_tensor is None:
                self.logger.error("❌ Image preprocessing failed")
                return None

            # Run inference
            try:
                if self.current_model is None:
                    self.logger.error("❌ Model is not loaded")
                    return None

                input_name = self.current_model.get_inputs()[0].name
                outputs = self.current_model.run(None, {input_name: input_tensor})

                if not outputs or len(outputs) == 0:
                    self.logger.error("❌ No outputs from model")
                    return None

                # Extract embedding
                embedding_vector = outputs[0]
                if len(embedding_vector.shape) > 1:
                    embedding_vector = embedding_vector[0]

                # Normalize embedding
                norm = np.linalg.norm(embedding_vector)
                if norm > 1e-8:
                    embedding_vector = embedding_vector / norm
                else:
                    self.logger.warning("⚠️ Very small embedding norm")

            except Exception as inference_error:
                self.logger.error(f"❌ Inference failed: {inference_error}")
                return None

            # Calculate quality
            quality_score = self._calculate_embedding_quality(embedding_vector)
            processing_time = time.time() - start_time

            # Create embedding object
            embedding = FaceEmbedding(
                vector=embedding_vector.astype(np.float32),
                model_type=self.current_model_type,
                quality_score=quality_score,
                extraction_time=processing_time,
                confidence=quality_score / 100.0,
                dimension=len(embedding_vector),
                normalized=True,
                extraction_method="onnx_inference",
            )

            # Update statistics
            self.stats.update_extraction_stats(
                processing_time, success=True, quality=embedding.face_quality
            )

            if self.current_model:
                device_used = (
                    "cuda"
                    if "CUDAExecutionProvider" in self.current_model.get_providers()
                    else "cpu"
                )
                self.stats.update_device_usage(processing_time, device_used)

            self.logger.debug(
                f"✅ Embedding extracted: {embedding_vector.shape}, "
                f"quality={quality_score:.1f}, time={processing_time * 1000:.1f}ms"
            )

            return embedding

        except Exception as e:
            processing_time = time.time() - start_time
            self.stats.update_extraction_stats(processing_time, success=False)
            self.logger.error(f"❌ Embedding extraction failed: {e}")
            return None

    # [ส่วนที่เหลือของโค้ดยังคงเหมือนเดิม...]

    def compare_faces(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        model_used: Optional[str] = None,
    ) -> FaceComparisonResult:
        """Compare two face embeddings"""
        start_time = time.time()

        try:
            # Ensure embeddings are valid
            if embedding1 is None or embedding2 is None:
                return FaceComparisonResult(
                    similarity=0.0,
                    is_same_person=False,
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    error="Invalid embeddings",
                )

            # Normalize embeddings if needed
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 > 1e-8:
                embedding1 = embedding1 / norm1
            if norm2 > 1e-8:
                embedding2 = embedding2 / norm2

            # Calculate cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            similarity = float(np.clip(dot_product, -1.0, 1.0))

            # Convert to 0-1 range
            similarity = (similarity + 1.0) / 2.0

            # Determine if same person
            is_same_person = similarity >= self.config.similarity_threshold

            processing_time = time.time() - start_time

            # Update statistics
            self.stats.update_comparison_stats(processing_time)

            return FaceComparisonResult(
                similarity=similarity,
                is_same_person=is_same_person,
                confidence=similarity,
                processing_time=processing_time,
                model_used=RecognitionModel(model_used)
                if model_used
                else self.current_model_type,
                distance=1.0 - similarity,
                comparison_method="cosine_similarity",
                threshold_used=self.config.similarity_threshold,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Face comparison failed: {e}")
            return FaceComparisonResult(
                similarity=0.0,
                is_same_person=False,
                confidence=0.0,
                processing_time=processing_time,
                error=str(e),
            )

    async def recognize_face(
        self,
        face_image: np.ndarray,
        gallery: FaceGallery,
        model_name: Optional[str] = None,
        top_k: int = 5,
    ) -> FaceRecognitionResult:
        """Recognize face against gallery"""
        start_time = time.time()

        try:
            # Extract query embedding
            embedding_start = time.time()
            query_embedding = await self.extract_embedding(face_image, model_name)
            embedding_time = time.time() - embedding_start

            if query_embedding is None or query_embedding.vector is None:
                return FaceRecognitionResult(
                    matches=[],
                    best_match=None,
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    model_used=self.current_model_type,
                    error="Failed to extract query embedding",
                    query_embedding=query_embedding,
                    embedding_time=embedding_time,
                )

            # Search in gallery
            search_start = time.time()
            matches: List[FaceMatch] = []

            for person_id, person_data in gallery.items():
                max_similarity = 0.0

                # Get embeddings for this person
                person_embeddings = person_data.get("embeddings", [])

                for stored_embedding in person_embeddings:
                    if isinstance(stored_embedding, np.ndarray):
                        # Compare embeddings
                        comparison = self.compare_faces(
                            query_embedding.vector,
                            stored_embedding,
                            self.current_model_type.value,
                        )

                        if comparison.similarity > max_similarity:
                            max_similarity = comparison.similarity

                # Create match if above threshold
                if max_similarity > 0:
                    match = FaceMatch(
                        person_id=person_id,
                        confidence=max_similarity,
                        similarity_score=max_similarity,
                        distance=1.0 - max_similarity,
                        comparison_method="cosine_similarity",
                    )
                    matches.append(match)

            search_time = time.time() - search_start

            # Sort matches by confidence
            matches.sort(key=lambda x: x.confidence, reverse=True)

            # Take top K matches
            if len(matches) > top_k:
                matches = matches[:top_k]

            # Add rank information
            for i, match in enumerate(matches):
                match.rank = i + 1

            # Determine best match
            best_match = None
            final_confidence = 0.0

            if matches and matches[0].confidence >= self.config.similarity_threshold:
                best_match = matches[0]
                final_confidence = matches[0].confidence

            total_time = time.time() - start_time

            return FaceRecognitionResult(
                matches=matches,
                best_match=best_match,
                confidence=final_confidence,
                processing_time=total_time,
                model_used=self.current_model_type,
                query_embedding=query_embedding,
                total_candidates=len(gallery),
                search_time=search_time,
                embedding_time=embedding_time,
            )

        except Exception as e:
            self.logger.error(f"❌ Face recognition failed: {e}")
            return FaceRecognitionResult(
                matches=[],
                best_match=None,
                confidence=0.0,
                processing_time=time.time() - start_time,
                model_used=self.current_model_type,
                error=str(e),
            )

    async def add_face_to_database(
        self,
        person_id: str,
        face_image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add face to internal database"""
        try:
            embedding = await self.extract_embedding(face_image)

            if embedding is None:
                self.logger.error(f"❌ Failed to extract embedding for {person_id}")
                return False

            if metadata:
                embedding.metadata = metadata

            if person_id not in self.face_database:
                self.face_database[person_id] = []

            self.face_database[person_id].append(embedding)

            total_embeddings = len(self.face_database[person_id])
            self.logger.info(
                f"✅ Added face for {person_id} to database (total: {total_embeddings})"
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ Error adding face to database: {e}")
            return False

    def get_performance_stats(self) -> ModelPerformanceStats:
        """Get performance statistics"""
        return self.stats

    async def switch_model(self, model_name: str) -> bool:
        """Switch to different recognition model"""
        try:
            target_model = RecognitionModel(model_name.lower())
            success = await self.load_model(target_model)

            if success:
                self.logger.info(f"✅ Switched to {model_name} model")
            else:
                self.logger.error(f"❌ Failed to switch to {model_name} model")

            return success

        except ValueError:
            self.logger.error(f"❌ Unknown model: {model_name}")
            return False

    async def cleanup(self) -> None:  # เพิ่ม return type annotation
        """Clean up resources"""
        try:
            if self.current_model is not None:
                del self.current_model
                self.current_model = None

            # Clear model cache
            self.models_cache.clear()

            # Clear GPU cache if available
            if TORCH_AVAILABLE and torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.logger.info("🧹 GPU cache cleared")

            # Release VRAM allocations
            if self.vram_manager:
                await self.vram_manager.release_model_allocation(
                    f"{self.current_model_type.value}-face-recognition"
                )

            self.logger.info("✅ Face Recognition Service cleaned up")

        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")

    def create_gallery_from_database(self) -> FaceGallery:
        """Create gallery from internal database"""
        gallery = {}

        for person_id, embeddings in self.face_database.items():
            gallery[person_id] = {
                "name": person_id,
                "embeddings": [
                    emb.vector for emb in embeddings if emb.vector is not None
                ],
            }

        return gallery
