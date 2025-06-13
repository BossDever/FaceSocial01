# type: ignore
# flake8: noqa
# ruff: noqa
"""
Enhanced Face Recognition Service with GPU Optimization
ระบบจดจำใบหน้าที่ปรับปรุงแล้วพร้อม GPU optimization และ Multi-model support
"""

import logging
import numpy as np
import cv2
import os
import time
from typing import Any, Dict, List, Optional, Union, Tuple, cast

from .models import RecognitionModel, FaceEmbedding, RecognitionConfig
from ..common.utils import (
    ONNX_AVAILABLE,
    TORCH_AVAILABLE,
    ort,
    torch,
)
from ..common.stats import ModelPerformanceStats
from ...core.log_config import get_logger

logger = get_logger(__name__)

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
        self.logger = logging.getLogger(__name__)        # Parse configuration
        if config is None:
            config = {}

        self.config = RecognitionConfig(
            preferred_model=RecognitionModel(config.get("preferred_model", "facenet")),
            similarity_threshold=config.get("similarity_threshold", 0.50),
            unknown_threshold=config.get("unknown_threshold", 0.40),
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

    async def load_model(self, model_type: RecognitionModel) -> bool:
        """Load a specific recognition model"""
        try:
            self.logger.info(f"Loading model: {model_type.value}")
            
            # Clean up previous model if any
            self._cleanup_previous_model()
            
            # Configure providers
            providers = self._configure_providers(model_type)
            
            # Create ONNX session
            session = self._create_onnx_session(model_type, providers)
            if session is None:
                self.logger.error(f"Failed to create ONNX session for {model_type.value}")
                return False
            
            # Set current model
            self.current_model = session
            self.current_model_type = model_type
            
            # Log success
            self._log_model_success(model_type)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model {model_type.value}: {e}", exc_info=True)
            return False

    async def _warmup_model(self) -> None:
        """Warm up the model with a dummy input"""
        if self.current_model is None or self.current_model_type is None:
            return
            
        try:
            # Create dummy input
            model_config = self.model_configs.get(self.current_model_type)
            if not model_config:
                return
                
            input_size = model_config["input_size"]
            dummy_input = np.random.randn(1, 3, input_size[1], input_size[0]).astype(np.float32)
            
            # Run inference
            input_name = self.current_model.get_inputs()[0].name
            self.current_model.run(None, {input_name: dummy_input})
            
            self.logger.info(f"Model {self.current_model_type.value} warmed up successfully")
            
        except Exception as e:
            self.logger.warning(f"Model warmup failed: {e}")

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
                f"(enable_gpu_opt: {self.config.enable_gpu_optimization}, "
                f"TORCH_OK: {TORCH_AVAILABLE}, torch!=None: {torch is not None}, "
                f"cuda_avail: {torch.cuda.is_available() if torch else 'N/A'})."
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
                f"🔥 Attempting CUDAProvider for {model_type.value} " +
                f"with {gpu_mem_limit / (1024**2):.0f}MB limit."
            )
            return ("CUDAExecutionProvider", cuda_options)
        except Exception as cuda_error:
            self.logger.warning(
                f"⚠️ CUDAProvider config failed for {model_type.value}: " +
                f"{cuda_error}. Will use CPU."
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
                f"✅ CUDAProvider configured for {model_type.value}."
            )
        else:
            # Corrected f-string concatenation for logging
            self.logger.info(
                f"ℹ️ CUDAProvider not added for {model_type.value} "
                "(unavailable/disabled/error)."
            )

        providers.append("CPUExecutionProvider")
        self.logger.debug(f"Final providers for {model_type.value}: {providers}")
        return providers

    def _create_onnx_session(
        self,
        model_type: RecognitionModel,
        providers: List[Union[str, Tuple[str, Dict[str, Any]]]]
    ) -> Optional[Any]: # Reverted to Optional[Any] for ort.InferenceSession type hint
        """Creates an ONNX inference session."""
        if not ort:
            self.logger.error("ONNX Runtime not available, cannot create session.")
            return None
        
        model_config = self._validate_model_config(model_type)
        if not model_config:
            return None

        model_path = cast(str, model_config["model_path"])
        session_options = self._configure_session_options()
        if not session_options:
            return None

        try:
            self.logger.info(
                f"Creating ONNX session for {model_path} with providers: {providers}"
            )
            session = ort.InferenceSession(
                model_path, sess_options=session_options, providers=providers
            )
            self.logger.info(
                f"ONNX session created successfully for {model_type.value}."
            )
            return session
        except Exception as e:
            self.logger.error(
                f"❌ Failed to create ONNX session for {model_type.value} "
                f"from {model_path} with providers {providers}: {e}",
                exc_info=True
            )
            # Fallback to CPU if CUDA fails and CPU was not the only option tried
            provider_names = [p[0] if isinstance(p, tuple) else p for p in providers]
            if ("CUDAExecutionProvider" in provider_names and
                    "CPUExecutionProvider" in provider_names):
                self.logger.warning(
                    f"Attempting fallback to CPU-only for {model_type.value}"
                )
                try:
                    # Type hint for cpu_providers
                    cpu_providers: List[Union[str, Tuple[str, Dict[str, Any]]]]
                    cpu_providers = ["CPUExecutionProvider"]
                    
                    if session_options is None:
                         session_options = self._configure_session_options()
                         if session_options is None:
                             self.logger.error(
                                 "Failed to re-configure session options "
                                 "for CPU fallback."
                             )
                             return None

                    session = ort.InferenceSession(
                        model_path,
                        sess_options=session_options,
                        providers=cpu_providers
                    )
                    self.logger.info(
                        f"ONNX session for {model_type.value} created "
                        "on CPU after fallback."
                    )
                    return session
                except Exception as e_cpu:
                    self.logger.error(
                        f"❌ Fallback to CPU for {model_type.value} failed: {e_cpu}",
                        exc_info=True
                    )
            return None

    def _preprocess_image(
        self, image_np: np.ndarray, model_type: RecognitionModel
    ) -> Optional[np.ndarray]:
        """Preprocesses the image for the given model type."""
        model_config = self.model_configs.get(model_type)
        if not model_config:
            self.logger.error(
                f"No config for model type {model_type.value} in _preprocess_image"
            )
            return None

        input_size = cast(Tuple[int, int], model_config["input_size"])
        mean = cast(List[float], model_config["mean"])
        std = cast(List[float], model_config["std"])

        try:
            # Resize
            img_resized = cv2.resize(image_np, input_size)

            # Normalize
            img_normalized = img_resized.astype(np.float32)
            if model_type == RecognitionModel.FACENET:
                 img_normalized = (img_normalized - mean[0]) / std[0]
            else: # General for ArcFace, AdaFace (0-1 range, then (img-mean)/std)
                img_normalized = img_normalized / 255.0
                img_normalized = (
                    img_normalized - np.array(mean, dtype=np.float32)
                ) / np.array(std, dtype=np.float32)

            # Transpose to NCHW format
            img_transposed = np.transpose(img_normalized, (2, 0, 1))
            img_expanded = np.expand_dims(img_transposed, axis=0) # Add batch dim
            return img_expanded
        except Exception as e:
            self.logger.error(
                f"Error preprocessing image for {model_type.value}: {e}", exc_info=True
            )
            return None

    def _extract_embedding(
        self, preprocessed_image: np.ndarray, model_type: RecognitionModel
    ) -> Optional[np.ndarray]:
        """Extracts face embedding using the current model."""
        if self.current_model is None or self.current_model_type != model_type:
            self.logger.error(
                f"Model {model_type.value} not loaded/mismatch, cannot extract."
            )
            return None
        
        if not ort: # Safeguard
            self.logger.error("ONNX Runtime not available for embedding extraction.")
            return None

        try:
            input_name = self.current_model.get_inputs()[0].name
            if preprocessed_image.dtype != np.float32:
                preprocessed_image = preprocessed_image.astype(np.float32)

            ort_inputs = {input_name: preprocessed_image}
            
            start_time = time.time()
            ort_outs = self.current_model.run(None, ort_inputs)
            embedding: np.ndarray = np.array(ort_outs[0], dtype=np.float32).flatten()
            end_time = time.time()

            extraction_duration = end_time - start_time
            self.stats.update_extraction_stats(
                time_taken=extraction_duration, success=True
            )
            
            norm = np.linalg.norm(embedding)
            if norm == 0:
                return embedding
            normalized_embedding: np.ndarray = embedding / norm
            
            return normalized_embedding
        except Exception as e:
            self.logger.error(
                f"Error extracting embedding with {model_type.value}: {e}",
                exc_info=True
            )
            # Use update_extraction_stats
            self.stats.update_extraction_stats(time_taken=0, success=False)
            return None

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

    async def add_face_from_image(
        self,
        image_bytes: bytes,
        person_name: str,
        person_id: Optional[str] = None,
        model_name: Optional[Union[str, RecognitionModel]] = None,
    ) -> Dict[str, Any]:
        """
        Adds a face from an image to the database.

        Args:
            image_bytes: Bytes of the image file.
            person_name: Name of the person.
            person_id: Optional unique ID for the person. If None, uses person_name.
            model_name: Specific recognition model to use for embedding.
                        Defaults to current.        Returns:
            A dictionary containing the result of the operation.
        """
        if person_id is None:
            person_id = person_name

        self.logger.info(
            f"Attempting to add face for person_id: {person_id} (Name: {person_name})"
        )

        try:
            img_np = self._decode_image(image_bytes, person_id)
            if img_np is None:
                return {"success": False, "error": "Failed to decode image."}

            current_model_type_to_use = await self._ensure_model_loaded(
                model_name
            )
            if current_model_type_to_use is None:
                return {
                    "success": False,
                    "error": "Failed to load or determine model.",
                }            # 🆕 Apply Face Enhancement Pipeline for Registration (ปิด augmentation ชั่วคราว)
            self.logger.info(f"🔄 Processing face with enhancement pipeline for {person_id}")
            processed_faces = self._process_face_for_registration(img_np, enable_augmentation=False)
            
            # Extract embeddings from all processed face variations
            all_embeddings = []
            successful_extractions = 0
            
            for i, processed_face in enumerate(processed_faces):
                self.logger.debug(f"Processing face variation {i+1}/{len(processed_faces)}")
                
                preprocessed_image = self._preprocess_image(
                    processed_face, current_model_type_to_use
                )
                if preprocessed_image is None:
                    self.logger.warning(f"Preprocessing failed for variation {i+1} of {person_id}")
                    continue

                embedding_vector = self._extract_embedding(
                    preprocessed_image, current_model_type_to_use
                )
                if embedding_vector is None:
                    self.logger.warning(f"Embedding extraction failed for variation {i+1} of {person_id}")
                    continue
                
                all_embeddings.append({
                    'vector': embedding_vector,
                    'variation_index': i,
                    'is_original': i == 0  # First one is always the original processed face
                })
                successful_extractions += 1
            
            if not all_embeddings:
                self.logger.error(f"All embedding extractions failed for {person_id}")
                return {"success": False, "error": "All embedding extractions failed."}
            
            self.logger.info(f"✅ Successfully extracted {successful_extractions}/{len(processed_faces)} embeddings for {person_id}")

            # Store all embedding variations for better recognition accuracy
            if person_id not in self.face_database:
                self.face_database[person_id] = []
            
            created_face_ids = []
            for emb_data in all_embeddings:
                face_id = f"{person_id}_{int(time.time())}_{emb_data['variation_index']}"
                
                new_embedding = FaceEmbedding(
                    id=face_id,
                    person_id=person_id,
                    person_name=person_name,
                    vector=emb_data['vector'],
                    model_type=current_model_type_to_use,
                    extraction_time=time.time(),
                    quality_score=0.0,  # Could be enhanced with quality assessment
                    metadata={
                        "original_person_id": person_id,
                        "timestamp": time.time(),
                        "variation_index": emb_data['variation_index'],
                        "is_original": emb_data['is_original'],
                        "processing_pipeline": "enhanced_registration_v1"
                    },
                )
                
                self.face_database[person_id].append(new_embedding)
                created_face_ids.append(face_id)

            self.logger.info(
                f"✅ Added {len(created_face_ids)} face embeddings for {person_id} (Name: {person_name}) " +
                f"using model {current_model_type_to_use.value}."
            )

            # Return information about all created embeddings
            primary_embedding = all_embeddings[0]['vector']  # Use first (original processed) as representative
            return {
                "success": True,
                "message": "Face added successfully with enhanced processing.",
                "face_ids": created_face_ids,
                "person_id": person_id,
                "person_name": person_name,
                "model_used": current_model_type_to_use.value,
                "embeddings_count": len(created_face_ids),
                "processing_stages": ["preprocessing", "pose_normalization", "data_augmentation"],
                "embedding_preview": primary_embedding[:5].tolist(),
            }

        except Exception as e:
            self.logger.error(
                f"❌ Error adding face for {person_id} (Name: {person_name}): {e}",
                exc_info=True,
            )
            error_message = f"An unexpected error occurred: {str(e)}"
            return {"success": False, "error": error_message}

    def _decode_image(
        self, image_bytes: bytes, person_id_for_log: str
    ) -> Optional[np.ndarray]:
        """Decodes image bytes into a NumPy array."""
        log_msg_prefix = f"Decoding image for {person_id_for_log}."
        img_len = len(image_bytes) if isinstance(image_bytes, bytes) else 'N/A'
        self.logger.debug(f"{log_msg_prefix} Type: {type(image_bytes)}, Len: {img_len}")
        if not isinstance(image_bytes, bytes):
            self.logger.error(
                f"Invalid image_bytes type: {type(image_bytes)} " +
                f"for {person_id_for_log}"
            )
            return None
        try:
            img_buffer = np.frombuffer(image_bytes, np.uint8)
            img_np = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
            if img_np is None:
                self.logger.error(f"cv2.imdecode failed for {person_id_for_log}.")
                return None
            return img_np
        except Exception as e:
            self.logger.error(
                f"Exception during image decoding for {person_id_for_log}: {e}",
                exc_info=True,
            )
            return None

    async def _ensure_model_loaded(
        self, model_name: Optional[Union[str, RecognitionModel]]
    ) -> Optional[RecognitionModel]:
        """Ensures the correct model is loaded, switching if necessary."""
        self.logger.info(f"🔍 _ensure_model_loaded called with model_name: {model_name}")
        self.logger.info(f"🔍 Current model type: {self.current_model_type}")
        self.logger.info(f"🔍 Current model exists: {self.current_model is not None}")
        
        target_model_type: Optional[RecognitionModel] = None
        if model_name:
            if isinstance(model_name, RecognitionModel):
                target_model_type = model_name
            elif isinstance(model_name, str):
                try:
                    target_model_type = RecognitionModel(model_name.lower())
                except ValueError:
                    self.logger.error(f"Invalid model_name: {model_name} provided.")
                    return None
        
        self.logger.info(f"🔍 Target model type: {target_model_type}")
        
        load_needed = False
        model_to_load: Optional[RecognitionModel] = None

        if target_model_type:
            if not self.current_model or self.current_model_type != target_model_type:
                load_needed = True
                model_to_load = target_model_type
                self.logger.info(f"🔍 Model switch needed: current={self.current_model_type}, target={target_model_type}")
        elif not self.current_model:
            load_needed = True
            model_to_load = self.config.preferred_model
            self.logger.info(f"🔍 No current model, loading preferred: {model_to_load}")

        if load_needed and model_to_load:
            self.logger.info(f"Attempting to load model: {model_to_load.value}.")
            if not await self.load_model(model_to_load):
                self.logger.error(f"Failed to load model {model_to_load.value}.")
                return None
        
        self.logger.info(f"🔍 Returning current model type: {self.current_model_type}")
        return self.current_model_type

    def _process_recognized_image(
        self,
        img_np: np.ndarray,
        current_model_type_to_use: RecognitionModel
    ) -> Optional[np.ndarray]:
        """Helper to preprocess and extract embedding for recognize_faces."""
        preprocessed_image = self._preprocess_image(
            img_np, current_model_type_to_use
        )
        if preprocessed_image is None:
            self.logger.error("Image preprocessing failed during recognition.")
            return None

        embedding_vector = self._extract_embedding(
            preprocessed_image, current_model_type_to_use
        )
        if embedding_vector is None:
            self.logger.error("Embedding extraction failed during recognition.")
            return None
        return embedding_vector

    async def recognize_faces(
        self,
        image_bytes: bytes,
        model_name: Optional[Union[str, RecognitionModel]] = None,
    ) -> Dict[str, Any]:
        """
        Recognize faces in an image.

        Args:
            image_bytes: Bytes of the image file.
            model_name: Specific recognition model to use.
                        Defaults to current model.

        Returns:
            A dictionary containing the recognition results.
        """
        self.logger.info(f"Recognizing faces in image ({len(image_bytes)} bytes).")

        try:
            img_np = self._decode_image(image_bytes, "recognition_task")
            if img_np is None:
                return {"success": False, "error": "Failed to decode image."}

            current_model_type_to_use = await self._ensure_model_loaded(model_name)
            if current_model_type_to_use is None:
                return {
                    "success": False,
                    "error": "Failed to load model for recognition.",
                }

            emb_vector = self._process_recognized_image(
                img_np, current_model_type_to_use
            )
            if emb_vector is None:
                return {"success": False, "error": "Image process/embed failed."}

            results = self._compare_embedding_to_database(
                emb_vector, current_model_type_to_use
            )

            if not results:
                self.logger.info("No matching faces found in the database.")
                return {
                    "success": True,
                    "results": [],
                    "message": "No matching faces found.",
                }

            # Sort results by similarity
            results = sorted(results, key=lambda x: x["similarity"], reverse=True)

            top_result = results[0]
            # is_known = top_result["similarity"] >= self.config.similarity_threshold
            # This is always true due to the earlier filter

            self.logger.info(
                f"Top match for recognition: Person ID {top_result['person_id']}, "
                f"Name {top_result['person_name']}, "
                f"Similarity {top_result['similarity']:.4f}"
            )

            return {
                "success": True,
                "results": results, # Return all sorted results
                "top_match": top_result, # Explicitly include top match
                "message": f"Found {len(results)} potential match(es).",
            }

        except Exception as e:
            self.logger.error(
                f"❌ Error recognizing faces: {e}", exc_info=True
            )
            return {"success": False, "error": f"Unexpected error: {str(e)}"}

    async def recognize_faces_with_gallery(
        self,
        image_bytes: bytes,
        gallery: Dict[str, Any],
        model_name: Optional[Union[str, RecognitionModel]] = None,
    ) -> Dict[str, Any]:
        """
        Recognize faces using external gallery instead of internal database.

        Args:
            image_bytes: Bytes of the image file.
            gallery: External gallery data in format:
                    {
                        "person_id": {
                            "name": "Person Name",
                            "embeddings": [[embedding1], [embedding2], ...]
                        }
                    }
            model_name: Specific recognition model to use.

        Returns:
            A dictionary containing the recognition results.
        """
        logger.info(
            f"Recognizing faces with external gallery ({len(image_bytes)} bytes, "
            f"{len(gallery)} people)."
        )
        processing_start_time = time.time()
        embedding_time = 0.0
        search_time = 0.0

        try:
            img_np = self._decode_image(image_bytes, "recognition_task_with_gallery")
            if img_np is None:
                return {"success": False, "error": "Failed to decode image."}

            current_model_type_to_use = await self._ensure_model_loaded(model_name)
            if current_model_type_to_use is None:
                return {
                    "success": False,
                    "error": "Failed to load model for recognition.",
                }

            embed_start_time = time.time()
            emb_vector = self._process_recognized_image(
                img_np, current_model_type_to_use
            )
            embedding_time = time.time() - embed_start_time

            if emb_vector is None:
                return {"success": False, "error": "Image process/embed failed."}

            if emb_vector is not None:
                query_norm = np.linalg.norm(emb_vector)
                if query_norm > 0:
                    emb_vector = emb_vector / query_norm
                else:
                    logger.warning("Query embedding is a zero vector.")
            
            search_start_time = time.time()
            results = self._compare_embedding_to_gallery(emb_vector, gallery)
            search_time = time.time() - search_start_time
            
            total_processing_time = time.time() - processing_start_time
            emb_list = emb_vector.tolist() if emb_vector is not None else []

            if not results:
                logger.info("No matching faces found in the provided gallery.")
                return {
                    "success": True,
                    "query_embedding": emb_list,
                    "matches": [],
                    "best_match": None,
                    "processing_time": total_processing_time,
                    "embedding_time": embedding_time,
                    "search_time": search_time,
                    "total_candidates": len(gallery),
                    "message": "No matching faces found in gallery.",
                }

            results = sorted(results, key=lambda x: x["similarity"], reverse=True)
            top_result = results[0]

            logger.info(
                f"Top match for gallery: ID {top_result['person_id']}, "
                f"Name {top_result.get('person_name', top_result['person_id'])}, "
                f"Sim {top_result['similarity']:.4f}"
            )

            return {
                "success": True,
                "query_embedding": emb_list,
                "matches": results,
                "best_match": top_result,
                "processing_time": total_processing_time,
                "embedding_time": embedding_time,
                "search_time": search_time,
                "total_candidates": len(gallery),
                "message": f"Found {len(results)} potential match(es) in gallery.",
            }

        except Exception as e:
            logger.error(
                f"❌ Error recognizing faces with gallery: {e}", exc_info=True
            )
            return {"success": False, "error": f"Unexpected error: {str(e)}"}

    def _compare_embedding_to_gallery(
        self, query_embedding: np.ndarray, gallery: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compares a single query embedding against all persons and their embeddings in a gallery."""
        matches: List[Dict[str, Any]] = []
        if not isinstance(gallery, dict):
            # logger.error("Gallery is not a dictionary, skipping gallery comparison.")
            return matches

        # Ensure query_embedding is a 1D numpy array
        if query_embedding.ndim > 1:
            query_embedding = query_embedding.flatten()

        for person_id, person_data in gallery.items():
            if not isinstance(person_data, dict):
                # logger.warning(f"Data for person {person_id} is not a dict, skipping.")
                continue

            name = person_data.get("name", person_id)
            embeddings_data = person_data.get("embeddings")

            # logger.debug(f"Processing gallery person: {person_id}, Name: {name}")

            if embeddings_data is None:
                # logger.warning(f"No embeddings found for person {person_id}, skipping.")
                continue

            best_match_for_person = self._compare_single_embedding_to_gallery_person(
                query_embedding, embeddings_data, person_id, name
            )
            if best_match_for_person:
                matches.append(best_match_for_person)
        
        # Sort matches by similarity, highest first
        matches.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        # logger.debug(f"Found {len(matches)} potential matches in gallery for query.")
        return matches

    # pylint: disable=too-complex
    def _process_single_gallery_embedding(
        self,
        query_embedding: np.ndarray,
        gallery_embedding_data: Any,
        person_id: str,
        embedding_index: int,  # For logging/debugging
    ) -> Optional[Dict[str, Any]]:
        """Processes a single gallery embedding against the query embedding."""
        try:
            if isinstance(gallery_embedding_data, dict):
                # Handle structured embedding (e.g., {"embedding": [...], "source": "..."})
                actual_embedding = gallery_embedding_data.get("embedding")
                # embedding_source = gallery_embedding_data.get("source", "unknown")
            elif isinstance(gallery_embedding_data, (list, np.ndarray)):
                actual_embedding = gallery_embedding_data
                # embedding_source = "direct_list_or_array"
            else:
                return None

            if actual_embedding is None:
                # logger.warning(f\"Actual embedding is None for {person_id} at index {embedding_index}\")
                return None

            gallery_embedding_np = np.array(actual_embedding, dtype=np.float32)

            if gallery_embedding_np.ndim == 0: # Scalar, not a valid embedding
                # logger.warning(f\"Skipping scalar emb for {person_id}, idx {embedding_index}\")
                return None

            gallery_embedding_normalized = self._normalize_embedding(gallery_embedding_np)

            if gallery_embedding_normalized is None:
                # logger.warning(f\"Normalization failed for {person_id} at index {embedding_index}\")
                return None
            
            if gallery_embedding_normalized.ndim > 1:
                gallery_embedding_normalized = gallery_embedding_normalized.flatten()
            if query_embedding.ndim > 1: # Should be pre-flattened, but ensure
                query_embedding = query_embedding.flatten()

            # Ensure both embeddings are 1D and have compatible shapes for cosine similarity
            if gallery_embedding_normalized.ndim != 1 or query_embedding.ndim != 1:
                # logger.error(
                #     f\"Embeddings for {person_id} (idx {embedding_index}) or query have incorrect \"
                #     f\"dimensions after flattening. Gallery: {gallery_embedding_normalized.ndim}D, \"
                #     f\"Query: {query_embedding.ndim}D. Skipping.\"
                # )
                return None
            
            # Cosine similarity expects 1D vectors.
            # If model produces 2D [1, dim], flatten.
            # Handled by flatten calls above.

            similarity = self._cosine_similarity(
                query_embedding, gallery_embedding_normalized
            )

            # logger.debug(
            #     f\"Similarity for {person_id} (idx {embedding_index}, \"
            #     f\"source: {embedding_source}): {similarity}\"
            # )

            if similarity >= self.config.unknown_threshold:
                return {
                    "person_id": person_id,
                    "name": person_id, # Placeholder, actual name should be from gallery structure
                    "similarity": float(similarity),
                    "match_type": "gallery",
                    # "embedding_source": embedding_source,
                }
        except Exception as e: # pylint: disable=broad-except, unused-variable
            # logger.error(
            #     f\"Error processing gallery embedding for {person_id} at index {embedding_index}: {e}\",
            #     exc_info=True
            # )
            pass # Instead of return None, let it fall through or handle specific exceptions
        return None

    def _compare_single_embedding_to_gallery_person(
        self,
        query_embedding: np.ndarray,
        embeddings_to_check: Union[List[Any], Dict[str, Any]],
        current_person_id: str,
        current_person_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Compares query embedding to all embeddings of a single person from the gallery."""
        best_match_for_person: Optional[Dict[str, Any]] = None
        highest_similarity_for_person = self.config.unknown_threshold - 0.01 # Start below threshold

        if isinstance(embeddings_to_check, list): # List of embeddings
            for embedding_index, gallery_embedding_data in enumerate(embeddings_to_check):
                # pylint: disable=line-too-long
                # logger.debug(f\"Comparing query to {current_person_id} embedding \"
                #              f\"{embedding_index + 1}/{len(embeddings_to_check)} with type \"
                #              f\"{type(gallery_embedding_data)}\")
                # pylint: enable=line-too-long

                match_details = self._process_single_gallery_embedding(
                    query_embedding, gallery_embedding_data, current_person_id, embedding_index
                )

                if (match_details and
                        match_details["similarity"] > highest_similarity_for_person):
                    highest_similarity_for_person = match_details["similarity"]
                    best_match_for_person = match_details
                    if best_match_for_person: # Ensure it's not None
                        best_match_for_person["name"] = current_person_name # Set correct name

        elif isinstance(embeddings_to_check, dict):  # Single embedding dictionary
            # TODO: handle single embedding dict similar to list case
            pass

        return best_match_for_person

    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Ensure both embeddings are normalized (unit vectors)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            # Ensure similarity is in valid range [-1, 1]
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0

    def _normalize_embedding(self, embedding: np.ndarray) -> Optional[np.ndarray]:
        """Normalize an embedding to unit vector"""
        try:
            embedding = embedding.flatten()
            norm = np.linalg.norm(embedding)
            if norm == 0:
                return None
            return embedding / norm
        except Exception as e:
            self.logger.error(f"Error normalizing embedding: {e}")
            return None

    def _compare_embedding_to_database(
        self, query_embedding: np.ndarray, model_type: RecognitionModel
    ) -> List[Dict[str, Any]]:
        """Compare query embedding against the internal face database"""
        matches = []
        
        if not self.face_database:
            return matches
            
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        for person_id, embeddings in self.face_database.items():
            best_similarity = 0.0
            best_embedding = None
            
            for face_embedding in embeddings:
                # Check if model types match
                if face_embedding.model_type != model_type:
                    continue
                    
                # Calculate similarity
                similarity = self._cosine_similarity(query_embedding, face_embedding.vector)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_embedding = face_embedding
            
            # Check if similarity meets threshold
            if best_similarity >= self.config.unknown_threshold:
                matches.append({
                    "person_id": person_id,
                    "person_name": best_embedding.person_name if best_embedding else person_id,
                    "similarity": float(best_similarity),
                    "confidence": float(best_similarity),
                    "match_type": "database",
                    "model_type": model_type.value
                })
        
        return matches

    # Face Enhancement Methods for Registration
    def _face_pose_normalization(self, face_image: np.ndarray) -> np.ndarray:
        """
        ปรับมุมท่าทางใบหน้าให้เป็นมาตรฐาน
        Face Pose Normalization - การปรับมุมท่าทางใบหน้าให้เป็นมาตรฐาน
        """
        try:
            # Detect facial landmarks for pose estimation
            height, width = face_image.shape[:2]
            
            # Simple pose normalization using histogram equalization and contrast adjustment
            # Convert to grayscale for pose analysis
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
            
            # Apply histogram equalization to normalize lighting
            equalized = cv2.equalizeHist(gray)
            
            # Convert back to BGR if original was color
            if len(face_image.shape) == 3:
                normalized_face = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
                # Blend with original to maintain color information
                normalized_face = cv2.addWeighted(face_image, 0.7, normalized_face, 0.3, 0)
            else:
                normalized_face = equalized
            
            # Apply bilateral filter to smooth while preserving edges
            normalized_face = cv2.bilateralFilter(normalized_face, 9, 75, 75)
            
            self.logger.debug("Face pose normalization completed successfully")
            return normalized_face
            
        except Exception as e:
            self.logger.error(f"Error in face pose normalization: {e}")
            return face_image  # Return original if normalization fails

    def _face_preprocessing(self, face_image: np.ndarray) -> np.ndarray:
        """
        การประมวลผลเบื้องต้นของใบหน้า
        Face Preprocessing - การประมวลผลเบื้องต้นของใบหน้า
        """
        try:
            # 1. Noise reduction using Non-local Means Denoising
            if len(face_image.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(face_image, None, 10, 10, 7, 21)
            else:
                denoised = cv2.fastNlMeansDenoising(face_image, None, 10, 7, 21)
            
            # 2. Contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
            if len(denoised.shape) == 3:
                # Convert to LAB color space for better contrast enhancement
                lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_channel = clahe.apply(l_channel)
                
                # Merge channels back
                enhanced = cv2.merge([l_channel, a_channel, b_channel])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(denoised)
            
            # 3. Sharpening filter
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # 4. Blend original and processed image
            result = cv2.addWeighted(enhanced, 0.8, sharpened, 0.2, 0)
            
            self.logger.debug("Face preprocessing completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in face preprocessing: {e}")
            return face_image  # Return original if preprocessing fails

    def _face_data_augmentation(self, face_image: np.ndarray) -> List[np.ndarray]:
        """
        การเพิ่มข้อมูลใบหน้า
        Face Data Augmentation - การเพิ่มข้อมูลใบหน้าสำหรับเพิ่มความแม่นยำ
        """
        augmented_faces = [face_image]  # Original image
        
        try:
            height, width = face_image.shape[:2]
            
            # 1. Brightness variations
            for brightness_delta in [-20, -10, 10, 20]:
                bright_face = cv2.convertScaleAbs(face_image, alpha=1.0, beta=brightness_delta)
                augmented_faces.append(bright_face)
            
            # 2. Contrast variations
            for contrast_alpha in [0.8, 0.9, 1.1, 1.2]:
                contrast_face = cv2.convertScaleAbs(face_image, alpha=contrast_alpha, beta=0)
                augmented_faces.append(contrast_face)
            
            # 3. Small rotations (-10 to +10 degrees)
            for angle in [-10, -5, 5, 10]:
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_face = cv2.warpAffine(face_image, rotation_matrix, (width, height))
                augmented_faces.append(rotated_face)
            
            # 4. Small translations
            for tx, ty in [(-5, 0), (5, 0), (0, -5), (0, 5)]:
                translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
                translated_face = cv2.warpAffine(face_image, translation_matrix, (width, height))
                augmented_faces.append(translated_face)
            
            # 5. Gaussian noise (subtle)
            noise = np.random.normal(0, 10, face_image.shape).astype(np.uint8)
            noisy_face = cv2.add(face_image, noise)
            augmented_faces.append(noisy_face)
            
            # 6. Gaussian blur (very light)
            blurred_face = cv2.GaussianBlur(face_image, (3, 3), 0.5)
            augmented_faces.append(blurred_face)
            
            self.logger.info(f"Generated {len(augmented_faces)} augmented face variations")
            return augmented_faces
            
        except Exception as e:
            self.logger.error(f"Error in face data augmentation: {e}")
            return [face_image]  # Return original if augmentation fails

    def _process_face_for_registration(self, face_image: np.ndarray, enable_augmentation: bool = True) -> List[np.ndarray]:
        """
        ประมวลผลใบหน้าสำหรับการลงทะเบียนด้วยการปรับปรุงคุณภาพและเพิ่มข้อมูล
        Process face for registration with quality enhancement and data augmentation
        """
        try:
            self.logger.info("🔄 Starting face processing for registration...")
            
            # Step 1: Face Preprocessing - การประมวลผลเบื้องต้น
            self.logger.debug("Step 1: Face Preprocessing")
            preprocessed_face = self._face_preprocessing(face_image)
            
            # Step 2: Face Pose Normalization - การปรับมุมท่าทาง
            self.logger.debug("Step 2: Face Pose Normalization")
            normalized_face = self._face_pose_normalization(preprocessed_face)
            
            # Step 3: Face Data Augmentation - การเพิ่มข้อมูล (ถ้าเปิดใช้งาน)
            if enable_augmentation:
                self.logger.debug("Step 3: Face Data Augmentation")
                augmented_faces = self._face_data_augmentation(normalized_face)
                self.logger.info(f"✅ Face processing completed with {len(augmented_faces)} variations")
                return augmented_faces
            else:
                self.logger.info("✅ Face processing completed (no augmentation)")
                return [normalized_face]
                
        except Exception as e:
            self.logger.error(f"❌ Error in face processing for registration: {e}")
            return [face_image]  # Return original if all processing fails

    def get_service_info(self) -> Dict[str, Any]:
        """Get comprehensive service information for health check"""
        try:
            # Check if service is initialized
            service_status = "online" if self.current_model is not None else "offline"
            
            # Get model information
            model_info = {}
            if self.current_model_type:
                model_info = {
                    "current_model": self.current_model_type.value,
                    "model_loaded": self.current_model is not None,
                    "embedding_dimension": self.config.embedding_dimension,
                    "similarity_threshold": self.config.similarity_threshold,
                    "unknown_threshold": self.config.unknown_threshold,
                    "batch_size": self.config.batch_size,
                    "gpu_enabled": self.config.enable_gpu_optimization,
                }
            
            # Get face database statistics
            database_stats = {
                "total_identities": len(self.face_database),
                "total_embeddings": sum(len(embeddings) for embeddings in self.face_database.values()),
                "registered_names": list(self.face_database.keys()) if self.face_database else []
            }
            
            # Get performance statistics
            performance_stats = {
                "total_recognitions": getattr(self.stats, 'total_recognitions', 0),
                "successful_recognitions": getattr(self.stats, 'successful_recognitions', 0),
                "average_processing_time": getattr(self.stats, 'average_processing_time', 0.0),
                "last_recognition_time": getattr(self.stats, 'last_recognition_time', 0.0),
            }
            
            return {
                "service_status": service_status,
                "model_info": model_info,
                "database_stats": database_stats,
                "performance_stats": performance_stats,
                "configuration": {
                    "preferred_model": self.config.preferred_model.value,
                    "quality_threshold": self.config.quality_threshold,
                    "enable_gpu_optimization": self.config.enable_gpu_optimization,
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting service info: {e}")
            return {
                "service_status": "error",
                "error": str(e),
                "model_info": {},
                "database_stats": {"total_identities": 0, "total_embeddings": 0},
                "performance_stats": {},
                "configuration": {}
            }
