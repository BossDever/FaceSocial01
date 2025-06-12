"""
Enhanced Face Recognition Service with GPU Optimization
ระบบจดจำใบหน้าที่ปรับปรุงแล้วพร้อม GPU optimization และ Multi-model support
"""

import logging
import numpy as np
import cv2
import os
import time
from typing import Optional, Dict, Any, List, Union, cast, Tuple

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

    current_model: Optional[Any]  # ort.InferenceSession if available
    current_model_type: Optional[RecognitionModel]
    models_cache: Dict[str, Any]
    face_database: Dict[str, List[FaceEmbedding]]
    stats: ModelPerformanceStats
    model_configs: Dict[RecognitionModel, Dict[str, Union[str, tuple, list, int]]]
    config: RecognitionConfig
    vram_manager: Any
    logger: logging.Logger

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
        self.models_cache = {}

        # Face database - compatible with existing system
        self.face_database = {}

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
            if self.current_model_type is None:
                self.logger.error("❌ Default model type is not set.")
                return False
            # self.current_model_type is confirmed to be RecognitionModel here
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

    def _cleanup_previous_model(self) -> None:
        """Clean up previous model and free memory, also reset type."""
        if self.current_model is not None:
            model_type_val = (
                self.current_model_type.value
                if self.current_model_type
                else "Unknown type"
            )
            self.logger.debug(f"Cleaning up previous model: {model_type_val}")
            del self.current_model
            self.current_model = None
            if TORCH_AVAILABLE and torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.debug("Previous model internal resources released.")

        # Always reset current_model_type to ensure clean state for next load
        if self.current_model_type is not None:
            model_type_val = (
                self.current_model_type.value
                if self.current_model_type
                else "None"
            )
            self.logger.debug(f"Resetting current model type from {model_type_val}.")
        self.current_model_type = None
        self.logger.debug("Model state (current_model and current_model_type) reset.")

    def _validate_model_config(
        self, model_type: RecognitionModel
    ) -> Optional[Dict[str, Union[str, tuple, list, int]]]:
        """Validate and get model configuration"""
        model_config = self.model_configs.get(model_type)
        if not model_config:
            self.logger.error(f"❌ Unknown model type: {model_type}")
            return None

        model_path = cast(str, model_config["model_path"])
        if not os.path.exists(model_path):
            self.logger.error(f"❌ Model file not found: {model_path}")
            return None

        return model_config

    def _configure_session_options(self) -> Optional[Any]:
        """Configure ONNX session options"""
        if not ort:
            self.logger.error("❌ ONNX Runtime not available")
            return None

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return session_options

    def _try_configure_cuda_provider(
        self, model_type: RecognitionModel
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Attempts to configure the CUDAExecutionProvider."""
        if not (
            self.config.enable_gpu_optimization
            and TORCH_AVAILABLE
            and torch is not None  # Check torch object itself
            and torch.cuda.is_available()
        ):
            self.logger.debug(
                f"CUDA prerequisites not met for {model_type.value} "
                f"(enable_gpu_optimization: {self.config.enable_gpu_optimization}, "
                f"TORCH_AVAILABLE: {TORCH_AVAILABLE}, "
                f"torch is not None: {torch is not None}, "
                f"cuda.is_available: {torch.cuda.is_available() if torch else 'N/A'})."
            )
            return None

        try:
            # Consider making these configurable if they vary per model or setup
            gpu_mem_limit = int(2 * 1024 * 1024 * 1024)  # 2GB
            cuda_options: Dict[str, Any] = {
                "device_id": 0,  # Make configurable if multi-GPU
                "arena_extend_strategy": "kSameAsRequested",
                "gpu_mem_limit": gpu_mem_limit,
                "cudnn_conv_algo_search": "HEURISTIC",
            }
            self.logger.info(
                f"🔥 Attempting to configure CUDAExecutionProvider for "
                f"{model_type.value} with {gpu_mem_limit / (1024**2):.0f}MB limit."
            )
            return ("CUDAExecutionProvider", cuda_options)
        except Exception as cuda_error:
            self.logger.warning(
                f"⚠️ CUDAExecutionProvider configuration failed for "
                f"{model_type.value}: {cuda_error}. Will attempt to use CPU."
            )
            return None

    def _configure_providers(
        self, model_type: RecognitionModel
    ) -> List[Union[str, Tuple[str, Dict[str, Any]]]]:
        """Configure execution providers for ONNX"""
        providers: List[Union[str, Tuple[str, Dict[str, Any]]]] = []

        cuda_provider_config = self._try_configure_cuda_provider(model_type)
        if cuda_provider_config:
            providers.append(cuda_provider_config)
            self.logger.info(
                f"✅ CUDAExecutionProvider successfully configured and added for "
                f"{model_type.value}."
            )
        else:
            self.logger.info(
                f"ℹ️ CUDAExecutionProvider not added for {model_type.value} "
                f"(not available, disabled, or config error)."
            )

        providers.append("CPUExecutionProvider")
        self.logger.debug(f"Final providers for {model_type.value}: {providers}")
        return providers

    def _log_model_success(self, model_type: RecognitionModel) -> None:
        """Log successful model loading"""
        if self.current_model:
            actual_providers = self.current_model.get_providers()
            device_used = (
                "GPU" if "CUDAExecutionProvider" in actual_providers else "CPU"
            )

            self.logger.info(
                f"✅ {model_type.value} loaded successfully on {device_used}"
            )
            self.logger.info(f"   Input: {self.current_model.get_inputs()[0].name}")
            self.logger.info(f"   Output: {self.current_model.get_outputs()[0].name}")

    def _create_onnx_session(
        self,
        model_path: str,
        providers: List[Union[str, Tuple[str, Dict[str, Any]]]],
        session_options: Any,  # ort.SessionOptions
    ) -> Optional[Any]:  # ort.InferenceSession
        """Creates an ONNX inference session."""
        try:
            self.logger.info(
                f"🔄 Creating ONNX inference session for model from: {model_path} "
                f"with providers: {providers}"
            )
            # Ensure ort is available before using it
            if not ort:
                self.logger.error(
                    "❌ ONNX Runtime is not available for session creation."
                )
                return None
            return ort.InferenceSession(
                model_path, providers=providers, sess_options=session_options
            )
        except Exception as e:
            self.logger.error(
                f"❌ Failed to create ONNX session from {model_path}: {e}",
                exc_info=True,
            )
            return None

    async def _ensure_model_ready(self) -> bool:
        """Ensure model is loaded and ready for inference"""
        if self.current_model is None or self.current_model_type is None:
            self.logger.error("❌ No model loaded")
            return False

        # Optional: Check if model is still valid
        try:
            # Test if model session is still valid
            if hasattr(self.current_model, 'get_inputs'):
                _ = self.current_model.get_inputs()
            return True
        except Exception as e:
            self.logger.error(f"❌ Model validation failed: {e}")
            return False

    def _get_actual_embedding_size(self) -> int:
        """Run a dummy input through the model to get the actual embedding size."""
        if not self.current_model or not self.current_model_type:
            self.logger.error("❌ Model not loaded, cannot determine embedding size.")
            return 0

        model_config = self.model_configs[self.current_model_type]
        input_size = cast(Tuple[int, int], model_config["input_size"])

        # Create a dummy input tensor
        dummy_input = np.random.rand(
            1, 3, input_size[0], input_size[1]
        ).astype(np.float32)

        input_name = self.current_model.get_inputs()[0].name

        try:
            output = self.current_model.run(None, {input_name: dummy_input})
            embedding_size = cast(int, output[0].shape[-1]) # Explicitly cast to int
            self.logger.info(
                f"✅ Actual embedding size for {self.current_model_type.value} "
                f"determined: {embedding_size}"
            )
            return embedding_size
        except Exception as e:
            self.logger.error(
                f"❌ Error running dummy input for {self.current_model_type.value}: {e}"
            )
            return 0

    def _validate_model_output_shape(self, model_type: RecognitionModel) -> int:
        """Validate the model output shape and determine embedding size."""
        if not self.current_model:
            self.logger.error("❌ Model not loaded, cannot validate output shape.")
            return 0

        outputs = self.current_model.get_outputs()
        if not outputs:
            self.logger.error(f"❌ No outputs found for model {model_type.value}")
            return 0

        output_shape = outputs[0].shape
        self.logger.debug(f"Output shape for {model_type.value}: {output_shape}")

        # Check for dynamic dimensions (None or string)
        if any(isinstance(dim, str) or dim is None for dim in output_shape):
            self.logger.info(
                f"⚠️ Dynamic output dimension detected for {model_type.value}. "
                f"Attempting to determine actual size."
            )
            # If the model is FaceNet and has dynamic output, get actual size
            if model_type == RecognitionModel.FACENET:
                return self._get_actual_embedding_size()
            else:
                # For other models with dynamic output, we might need specific handling
                # or rely on a default/configured value if runtime check is not
                # feasible. For now, log a warning and use configured size.
                configured_size = cast(
                    int, self.model_configs[model_type].get("embedding_size", 0)
                )
                self.logger.warning(
                    f"⚠️ Dynamic output for {model_type.value} but not FaceNet. "
                    f"Using configured embedding size: {configured_size}. "
                    f"Consider implementing specific runtime checks if needed."
                )
                return configured_size

        # Assuming the last dimension is the embedding size for fixed shapes
        if len(output_shape) > 1 and isinstance(output_shape[-1], int):
            return output_shape[-1]

        self.logger.warning(
            f"⚠️ Could not determine embedding size from output shape {output_shape} "
            f"for {model_type.value}. Using configured size."
        )
        return cast(int, self.model_configs[model_type].get("embedding_size", 0))

    async def load_model(self, model_type: RecognitionModel) -> bool:
        """Load face recognition model with GPU optimization"""
        if (
            self.current_model_type == model_type
            and self.current_model is not None
        ):
            self.logger.debug(
                f"Model {model_type.value} already loaded and matches requested type."
            )
            return True

        self.logger.info(
            f"Initiating load for model: {model_type.value}. "
            f"Current loaded: "
            f"{self.current_model_type.value if self.current_model_type else 'None'}"
        )
        self._cleanup_previous_model()  # Resets self.current_model & type

        model_config = self._validate_model_config(model_type)
        if not model_config:
            return False  # Error logged in _validate_model_config
        model_path = cast(str, model_config["model_path"])

        session_options = self._configure_session_options()
        if not session_options:
            return False  # Error logged in _configure_session_options

        providers = self._configure_providers(model_type)

        new_session = self._create_onnx_session(model_path, providers, session_options)

        if not new_session:
            # Ensure clean state if session creation failed
            self._cleanup_previous_model()
            return False

        self.current_model = new_session
        self.current_model_type = model_type
        self._log_model_success(model_type)

        # Validate and update embedding size
        actual_embedding_size = self._validate_model_output_shape(model_type)
        if actual_embedding_size > 0:
            self.model_configs[model_type]["embedding_size"] = actual_embedding_size
            self.logger.info(
                f"Updated embedding size for {model_type.value} to "
                f"{actual_embedding_size}"
            )
        else:
            self.logger.warning(
                f"Could not determine actual embedding size for {model_type.value}. "
                f"Using configured value."
            )

        if self.vram_manager:
            self.logger.info(f"Requesting VRAM allocation for {model_type.value}")
            try:
                await self.vram_manager.request_model_allocation(
                    f"{model_type.value}-face-recognition",
                    "high",
                    "face_recognition_service",
                )
            except Exception as e:
                self.logger.warning(
                    f"⚠️ VRAM allocation request failed for {model_type.value}: {e}"
                )
                # Depending on policy, this might not be a critical failure

        self.logger.info(
            f"✅ Successfully loaded and configured model: {model_type.value}"
        )
        return True

    async def _warmup_model(self) -> None:
        """Warm up model for optimal performance"""
        try:
            if self.current_model is None or self.current_model_type is None:
                self.logger.warning(
                    "⚠️ Model or model type not available for warmup."
                )
                return

            # self.current_model_type is confirmed to be RecognitionModel here
            self.logger.info(
                f"🔥 Warming up {self.current_model_type.value} model..."
            )

            # Get model configuration
            if self.current_model_type not in self.model_configs:
                self.logger.error(
                    f"❌ Model config not found for {self.current_model_type.value}"
                )
                return
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

            if self.current_model_type is None:
                self.logger.error(
                    "❌ Current model type not set for preprocessing."
                )
                return None

            # self.current_model_type is confirmed to be RecognitionModel here
            # Get model configuration
            if self.current_model_type not in self.model_configs:
                self.logger.error(
                    f"❌ Model config not found for {self.current_model_type.value}"
                )
                return None
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

    async def _switch_model_if_needed(self, model_name: Optional[str]) -> None:
        """Switch model if a different one is requested"""
        if model_name:
            should_switch = False
            if self.current_model_type is None:
                should_switch = True
            elif model_name != self.current_model_type.value:
                should_switch = True

            if should_switch:
                try:
                    target_model = RecognitionModel(model_name.lower())
                    await self.load_model(target_model)
                except ValueError:
                    self.logger.warning(
                        f"⚠️ Unknown model: {model_name}, using current model"
                    )
            # If model_name matches current_model_type.value, do nothing.
        # If model_name is None, do nothing.

    def _run_model_inference(self, input_tensor: np.ndarray) -> Optional[np.ndarray]:
        """Run model inference and return embedding vector"""
        try:
            if self.current_model is None:
                self.logger.error("❌ Model is not loaded")
                return None

            input_name = self.current_model.get_inputs()[0].name
            outputs = self.current_model.run(None, {input_name: input_tensor})

            if not outputs or len(outputs) == 0:
                self.logger.error("❌ No outputs from model")
                return None

            # Extract and normalize embedding
            embedding_vector: np.ndarray = outputs[0]  # Explicitly type hint
            if len(embedding_vector.shape) > 1:
                embedding_vector = embedding_vector[0]

            norm = np.linalg.norm(embedding_vector)
            if norm > 1e-8:
                embedding_vector = embedding_vector / norm
            else:
                self.logger.warning("⚠️ Very small embedding norm")

            return embedding_vector

        except Exception as inference_error:
            self.logger.error(f"❌ Inference failed: {inference_error}")
            return None

    def _create_embedding_object(
        self, embedding_vector: np.ndarray, processing_time: float
    ) -> FaceEmbedding:
        """Create FaceEmbedding object with all metadata"""
        quality_score = self._calculate_embedding_quality(embedding_vector)

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

        return embedding

    def _update_extraction_stats(
        self, embedding: FaceEmbedding, processing_time: float
    ) -> None:
        """Update performance statistics after extraction"""
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

    async def extract_embedding(
        self, face_image: np.ndarray, model_name: Optional[str] = None
    ) -> Optional[FaceEmbedding]:
        """Extract face embedding with enhanced error handling and structure"""
        start_time = time.time()

        await self._switch_model_if_needed(model_name)

        if not await self._ensure_model_ready():
            self.stats.update_extraction_stats(
                time.time() - start_time, success=False
            )
            # Errors logged by _ensure_model_ready or its callees
            return None

        input_tensor = self._preprocess_image(face_image)
        if input_tensor is None:
            self.stats.update_extraction_stats(
                time.time() - start_time, success=False
            )
            # _preprocess_image logs its own error
            return None

        embedding_vector = self._run_model_inference(input_tensor)
        if embedding_vector is None:
            self.stats.update_extraction_stats(
                time.time() - start_time, success=False
            )
            # _run_model_inference logs its own error
            return None

        # Final steps: object creation and stats update
        try:
            processing_time = time.time() - start_time
            embedding = self._create_embedding_object(
                embedding_vector, processing_time
            )
            self._update_extraction_stats(embedding, processing_time)

            self.logger.debug(
                f"✅ Embedding extracted: {embedding_vector.shape}, "
                f"quality={embedding.quality_score:.1f}, "
                f"time={processing_time * 1000:.1f}ms"
            )
            return embedding
        except Exception as e:
            # Catch errors from the final creation/stats steps
            self.stats.update_extraction_stats(
                time.time() - start_time, success=False
            )
            self.logger.error(
                f"❌ Embedding finalization failed: {e}", exc_info=True
            )
            return None

    def compare_faces(
        self,
        embedding1: Optional[np.ndarray],
        embedding2: Optional[np.ndarray],
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

            # At this point, embedding1 and embedding2 are np.ndarray
            # Normalize embeddings if needed
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            embedding1_normalized: np.ndarray = embedding1
            embedding2_normalized: np.ndarray = embedding2

            if norm1 > 1e-8:
                embedding1_normalized = embedding1 / norm1
            else:
                # Optionally log a warning or handle zero vector
                self.logger.warning("Embedding 1 has near-zero norm.")

            if norm2 > 1e-8:
                embedding2_normalized = embedding2 / norm2
            else:
                # Optionally log a warning or handle zero vector
                self.logger.warning("Embedding 2 has near-zero norm.")

            # Calculate cosine similarity
            dot_product = np.dot(embedding1_normalized, embedding2_normalized)
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

    def _search_gallery(
        self, query_embedding: FaceEmbedding, gallery: FaceGallery
    ) -> List[FaceMatch]:
        """Search for matches in the gallery"""
        matches: List[FaceMatch] = []

        if self.current_model_type is None:
            self.logger.error("❌ Current model type not set for gallery search.")
            return matches

        for person_id, person_data in gallery.items():
            max_similarity = 0.0
            person_embeddings = person_data.get("embeddings", [])

            for stored_embedding in person_embeddings:
                if isinstance(stored_embedding, np.ndarray):
                    # self.current_model_type is confirmed not None here
                    model_type_value = self.current_model_type.value
                    comparison = self.compare_faces(
                        query_embedding.vector,
                        stored_embedding,
                        model_type_value,
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

        return matches

    def _process_matches(self, matches: List[FaceMatch], top_k: int) -> List[FaceMatch]:
        """Sort, rank and limit matches"""
        # Sort matches by confidence
        matches.sort(key=lambda x: x.confidence, reverse=True)

        # Take top K matches
        if len(matches) > top_k:
            matches = matches[:top_k]

        # Add rank information
        for i, match in enumerate(matches):
            match.rank = i + 1

        return matches

    def _determine_best_match(
        self, matches: List[FaceMatch]
    ) -> Tuple[Optional[FaceMatch], float]:
        """Determine the best match and confidence"""
        best_match = None
        final_confidence = 0.0

        if matches and matches[0].confidence >= self.config.similarity_threshold:
            best_match = matches[0]
            final_confidence = matches[0].confidence

        return best_match, final_confidence

    async def recognize_face(
        self,
        face_image: np.ndarray,
        gallery: FaceGallery,
        model_name: Optional[str] = None,
        top_k: int = 5,
    ) -> FaceRecognitionResult:
        """Recognize face against gallery with refactored structure"""
        start_time = time.time()
        # Cache model type in case of errors before it's updated by extract_embedding
        cached_model_type = self.current_model_type

        embedding_start_time = time.time()
        query_embedding = await self.extract_embedding(face_image, model_name)
        embedding_time = time.time() - embedding_start_time

        if query_embedding is None or query_embedding.vector is None:
            self.logger.warning(
                "Query embedding extraction failed during recognize_face."
            )
            return FaceRecognitionResult(
                matches=[],
                best_match=None,
                confidence=0.0,
                processing_time=time.time() - start_time,
                model_used=cached_model_type,  # Use cached model type
                error="Failed to extract query embedding for recognition",
                query_embedding=query_embedding, # Pass along for debugging
                embedding_time=embedding_time,
            )

        # Proceed with gallery search and matching if embedding is successful
        try:
            search_start_time = time.time()
            # query_embedding and query_embedding.vector are confirmed not None here
            matches = self._search_gallery(query_embedding, gallery)
            search_time = time.time() - search_start_time

            processed_matches = self._process_matches(matches, top_k)
            best_match, final_confidence = self._determine_best_match(
                processed_matches
            )

            total_processing_time = time.time() - start_time

            return FaceRecognitionResult(
                matches=processed_matches,
                best_match=best_match,
                confidence=final_confidence,
                processing_time=total_processing_time,
                # Reflects model used by extract_embedding
                model_used=self.current_model_type,
                query_embedding=query_embedding,
                total_candidates=len(gallery),
                search_time=search_time,
                embedding_time=embedding_time,
            )
        except Exception as e:
            self.logger.error(
                f"❌ Face recognition (gallery search/match) failed: {e}",
                exc_info=True,
            )
            return FaceRecognitionResult(
                matches=[],
                best_match=None,
                confidence=0.0,
                processing_time=time.time() - start_time,
                # Or cached_model_type if preferred
                model_used=self.current_model_type,
                error=f"Error during gallery search or match processing: {str(e)}",
                query_embedding=query_embedding, # Include successful embedding
                embedding_time=embedding_time,
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

    async def cleanup(self) -> None:
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
            if self.vram_manager and self.current_model_type:
                # self.current_model_type is confirmed not None here
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
