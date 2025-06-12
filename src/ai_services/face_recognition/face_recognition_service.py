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
    RecognitionModel,
    ModelPerformanceStats,
    RecognitionConfig,
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
                        Defaults to current.

        Returns:
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
                }

            preprocessed_image = self._preprocess_image(
                img_np, current_model_type_to_use
            )
            if preprocessed_image is None:
                self.logger.error(f"Preprocessing failed for {person_id}.")
                return {"success": False, "error": "Image preprocessing failed."}

            embedding_vector = self._extract_embedding(
                preprocessed_image, current_model_type_to_use
            )
            if embedding_vector is None:
                self.logger.error(f"Embedding extraction failed for {person_id}.")
                return {"success": False, "error": "Embedding extraction failed."}

            face_id = f"{person_id}_{int(time.time())}"

            new_embedding = FaceEmbedding(
                id=face_id,
                person_id=person_id,
                person_name=person_name,
                vector=embedding_vector,
                model_type=current_model_type_to_use,
                extraction_time=time.time(),  # Placeholder for extraction duration
                quality_score=0.0,  # Placeholder: Implement quality assessment
                metadata={
                    "original_person_id": person_id,
                    "timestamp": time.time(),
                },
            )

            if person_id not in self.face_database:
                self.face_database[person_id] = []
            self.face_database[person_id].append(new_embedding)

            self.logger.info(
                f"✅ Added face {face_id} for {person_id} (Name: {person_name}) " +
                f"using model {current_model_type_to_use.value}."
            )

            return {
                "success": True,
                "message": "Face added successfully.",
                "face_id": face_id,
                "person_id": person_id,
                "person_name": person_name,
                "model_used": current_model_type_to_use.value,
                "embedding_preview": embedding_vector[:5].tolist(),
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
        
        load_needed = False
        model_to_load: Optional[RecognitionModel] = None

        if target_model_type:
            if not self.current_model or self.current_model_type != target_model_type:
                load_needed = True
                model_to_load = target_model_type
        elif not self.current_model:
            load_needed = True
            model_to_load = self.config.preferred_model

        if load_needed and model_to_load:
            self.logger.info(f"Attempting to load model: {model_to_load.value}.")
            if not await self.load_model(model_to_load):
                self.logger.error(f"Failed to load model {model_to_load.value}.")
                return None
        
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

    def _compare_embedding_to_database(
        self,
        query_embedding: np.ndarray,
        model_used_for_query: RecognitionModel,
    ) -> List[Dict[str, Any]]:
        """Compares a query embedding to all embeddings in the database."""
        matches: List[Dict[str, Any]] = []
        if query_embedding is None or query_embedding.size == 0:
            self.logger.warning("Query embedding is empty, cannot compare.")
            return matches

        for person_id, embeddings in self.face_database.items():
            for db_embedding_obj in embeddings:
                if db_embedding_obj.vector is None:
                    self.logger.debug(
                        f"Skipping empty DB embedding for {person_id}, "
                        f"ID: {db_embedding_obj.id}"
                    )
                    continue

                # Ensure model compatibility before comparison
                db_model_type_str = (
                    db_embedding_obj.model_type.value
                    if db_embedding_obj.model_type
                    else "unknown"
                )
                query_model_type_str = model_used_for_query.value

                if db_model_type_str != query_model_type_str:
                    self.logger.debug(
                        f"Skipping comparison: DB model ({db_model_type_str}) "
                        f"!= Query model ({query_model_type_str}) for "
                        f"{person_id} / {db_embedding_obj.id}."
                    )
                    continue

                similarity = self._cosine_similarity(
                    query_embedding, db_embedding_obj.vector
                )

                if similarity >= self.config.unknown_threshold:
                    match_data = {
                        "person_id": person_id,
                        "person_name": db_embedding_obj.person_name,
                        "similarity": float(similarity),
                        "face_id_db": db_embedding_obj.id,
                        "model_match": db_model_type_str == query_model_type_str,
                    }
                    matches.append(match_data)
                    self.logger.debug(
                        f"Match found: {person_id} with similarity {similarity:.4f} "
                        f"(DB Face ID: {db_embedding_obj.id}, "
                        f"Query Model: {query_model_type_str})"
                    )
        return matches

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute the cosine similarity between two embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0
        # Avoid division by zero
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
        return float(similarity)

    async def load_model(
        self, model_type: RecognitionModel, force_reload: bool = False
    ) -> bool:
        """Load a specific recognition model"""
        self.logger.info(
            f"🔄 Loading model: {model_type.value} (Force reload: {force_reload})"
        )
        
        if (self.current_model_type == model_type and
            self.current_model is not None and not force_reload):
            self.logger.info(f"Model {model_type.value} is already loaded.")
            return True

        if self.current_model_type != model_type or force_reload:
             self._cleanup_previous_model()

        if not force_reload and model_type.value in self.models_cache:
            self.logger.debug(f"Model {model_type.value} found in cache.")
            self.current_model = self.models_cache[model_type.value]
            self.current_model_type = model_type
            self._log_model_success(model_type)
            return True

        model_config = self._validate_model_config(model_type)
        if not model_config:
            return False

        providers = self._configure_providers(model_type)
        
        onnx_session = self._create_onnx_session(model_type, providers)
        if not onnx_session:
            self.logger.error(
                f"Failed to create ONNX session for {model_type.value} "
                "(all attempts)."
            )
            return False

        self._cleanup_previous_model() # Clean up before assigning new
        self.current_model = onnx_session
        self.current_model_type = model_type
        # Use .value for dict key
        self.models_cache[model_type.value] = self.current_model

        self._log_model_success(model_type)
        return True

    async def _warmup_model(self) -> None:
        """Warm up the currently loaded model"""
        if not self.current_model or not self.current_model_type:
            self.logger.warning("No model loaded, skipping warmup.")
            return

        current_model_type_val = self.current_model_type.value

        model_config = self.model_configs.get(self.current_model_type)
        if not model_config:
            self.logger.error(
                f"No config for model {current_model_type_val} during warmup."
            )
            return
        
        input_size_any = model_config.get("input_size")
        if not (isinstance(input_size_any, tuple) and
           len(input_size_any) == 2 and
           all(isinstance(d, int) for d in input_size_any)):
            self.logger.error(
                f"Invalid input_size for {current_model_type_val} in warmup: "
                f"{input_size_any}."
            )
            return
        
        input_size = cast(Tuple[int, int], input_size_any)

        self.logger.info(f"🔥 Warming up {current_model_type_val} model...")
        try:
            dummy_raw_image = np.random.randint(
                0, 256, (input_size[1], input_size[0], 3), dtype=np.uint8
            )
            preprocessed_dummy = self._preprocess_image(
                dummy_raw_image, self.current_model_type
            )

            if preprocessed_dummy is not None:
                _ = self._extract_embedding(
                    preprocessed_dummy, self.current_model_type
                )
                self.logger.info(
                    f"✅ {current_model_type_val} model warmed up."
                )
            else:
                self.logger.error(
                    f"Preprocessing dummy for {current_model_type_val} "
                    "warmup failed."
                )
        
        except Exception as e:
            self.logger.error(
                f"Error during model warmup for {current_model_type_val}: {e}",
                exc_info=True
            )
