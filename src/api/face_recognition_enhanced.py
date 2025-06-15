"""
Enhanced Face Recognition Service with Multi-Model Support - Fixed Version
ใช้ % distribution: facenet onnx 50% adaface onnx 25% arcface 25%
แก้ไขปัญหาการจัดการโมเดลหลายตัวและ performance optimization
"""

import logging
import numpy as np
import cv2
import os
import time
import random
import asyncio
from typing import Any, Dict, List, Optional, Union, Tuple, cast
from dataclasses import dataclass

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

@dataclass
class ModelDistribution:
    """Model distribution configuration"""
    facenet_percentage: float = 50.0
    adaface_percentage: float = 25.0
    arcface_percentage: float = 25.0
    
    def get_random_model(self) -> RecognitionModel:
        """Get random model based on distribution"""
        rand_value = random.uniform(0, 100)
        
        if rand_value < self.facenet_percentage:
            return RecognitionModel.FACENET
        elif rand_value < self.facenet_percentage + self.adaface_percentage:
            return RecognitionModel.ADAFACE
        else:
            return RecognitionModel.ARCFACE

class FaceRecognitionEnhancedService:
    """Enhanced Face Recognition Service with Multi-Model Support"""
    
    def __init__(
        self, 
        vram_manager: Any = None, 
        config: Optional[Dict[str, Any]] = None,
        enable_multi_framework: bool = False,
        frameworks: Optional[List[str]] = None,
        models_path: str = "./model/face-recognition/"
    ) -> None:
        self.vram_manager = vram_manager
        self.logger = logging.getLogger(__name__)
        
        # Parse configuration
        if config is None:
            config = {}
        
        # Multi-framework support
        self.enable_multi_framework = config.get("enable_multi_framework", enable_multi_framework)
        self.requested_frameworks = config.get("frameworks", frameworks) or []
        self.models_path = config.get("models_path", models_path)

        # Model distribution configuration
        self.model_distribution = ModelDistribution()
        
        # Enhanced configuration
        self.config = RecognitionConfig(
            preferred_model=RecognitionModel(config.get("preferred_model", "facenet")),
            similarity_threshold=config.get("similarity_threshold", 0.50),
            unknown_threshold=config.get("unknown_threshold", 0.40),
            embedding_dimension=config.get("embedding_dimension", 512),
            enable_gpu_optimization=config.get("enable_gpu_optimization", True),
            batch_size=config.get("batch_size", 8),
            quality_threshold=config.get("quality_threshold", 0.2),
        )

        # Multi-model management
        self.loaded_models: Dict[RecognitionModel, Any] = {}
        self.model_load_order: List[RecognitionModel] = []
        self.max_loaded_models = 3  # All three models can be loaded
        
        # Current active model
        self.current_model = None
        self.current_model_type = None

        # Face database
        self.face_database: Dict[str, List[FaceEmbedding]] = {}

        # Performance tracking
        self.stats = ModelPerformanceStats()
        self.model_stats: Dict[str, Dict[str, Any]] = {}

        # Model configurations
        self.model_configs: Dict[RecognitionModel, Dict[str, Union[str, tuple, list, int]]] = {
            RecognitionModel.FACENET: {
                "model_path": "model/face-recognition/facenet_vggface2.onnx",
                "input_size": (160, 160),
                "mean": [127.5, 127.5, 127.5],
                "std": [128.0, 128.0, 128.0],
                "embedding_size": 512,
                "weight": 0.5,  # 50%
            },
            RecognitionModel.ADAFACE: {
                "model_path": "model/face-recognition/adaface_ir101.onnx",
                "input_size": (112, 112),
                "mean": [127.5, 127.5, 127.5],
                "std": [127.5, 127.5, 127.5],
                "embedding_size": 512,
                "weight": 0.25,  # 25%
            },
            RecognitionModel.ARCFACE: {
                "model_path": "model/face-recognition/arcface_r100.onnx",
                "input_size": (112, 112),
                "mean": [127.5, 127.5, 127.5],
                "std": [127.5, 127.5, 127.5],
                "embedding_size": 512,
                "weight": 0.25,  # 25%
            },
        }

        self.logger.info("🚀 Enhanced Face Recognition Service initialized")
        self.logger.info(f"📊 Model distribution: FaceNet {self.model_distribution.facenet_percentage}%, "
                        f"AdaFace {self.model_distribution.adaface_percentage}%, "
                        f"ArcFace {self.model_distribution.arcface_percentage}%")

    async def initialize(self) -> bool:
        """Initialize the enhanced service"""
        try:
            self.logger.info("🔧 Initializing Enhanced Face Recognition Service...")

            if not ONNX_AVAILABLE:
                self.logger.error("❌ ONNX Runtime not available")
                return False

            # Initialize model statistics
            for model_type in RecognitionModel:
                self.model_stats[model_type.value] = {
                    "usage_count": 0,
                    "success_count": 0,
                    "total_time": 0.0,
                    "average_time": 0.0,
                    "last_used": 0.0
                }

            # Pre-load preferred model
            success = await self.load_model(self.config.preferred_model)
            if not success:
                self.logger.warning(f"⚠️ Failed to load preferred model {self.config.preferred_model.value}")
                
            # Try to pre-load other models based on distribution
            await self._preload_models()

            self.logger.info("✅ Enhanced Face Recognition Service initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error initializing enhanced service: {e}")
            return False

    async def _preload_models(self) -> None:
        """Pre-load models based on distribution"""
        try:
            model_priority = [
                RecognitionModel.FACENET,  # Highest priority (50%)
                RecognitionModel.ADAFACE,  # Medium priority (25%)
                RecognitionModel.ARCFACE   # Lower priority (25%)
            ]
            
            for model_type in model_priority:
                if model_type not in self.loaded_models:
                    try:
                        success = await self.load_model(model_type, preload=True)
                        if success:
                            self.logger.info(f"✅ Pre-loaded model: {model_type.value}")
                        else:
                            self.logger.warning(f"⚠️ Failed to pre-load model: {model_type.value}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Pre-load failed for {model_type.value}: {e}")
                        
        except Exception as e:
            self.logger.error(f"❌ Error in model preloading: {e}")

    async def load_model(self, model_type: RecognitionModel, preload: bool = False) -> bool:
        """Load a specific model with enhanced management"""
        try:
            # Check if model is already loaded
            if model_type in self.loaded_models:
                if not preload:
                    self.current_model = self.loaded_models[model_type]
                    self.current_model_type = model_type
                    self.logger.info(f"🔄 Switched to already loaded model: {model_type.value}")
                return True

            self.logger.info(f"📥 Loading model: {model_type.value}")
            
            # Check if we need to unload models (memory management)
            if len(self.loaded_models) >= self.max_loaded_models:
                await self._manage_model_memory()
            
            # Configure providers
            providers = self._configure_providers(model_type)
            
            # Create ONNX session
            session = self._create_onnx_session(model_type, providers)
            if session is None:
                self.logger.error(f"❌ Failed to create ONNX session for {model_type.value}")
                return False
            
            # Store the loaded model
            self.loaded_models[model_type] = session
            self.model_load_order.append(model_type)
            
            # Set as current if not preloading
            if not preload:
                self.current_model = session
                self.current_model_type = model_type
            
            # Warm up the model
            await self._warmup_model(model_type, session)
            
            self.logger.info(f"✅ Model {model_type.value} loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error loading model {model_type.value}: {e}", exc_info=True)
            return False

    async def _manage_model_memory(self) -> None:
        """Manage model memory by unloading least recently used models"""
        try:
            if len(self.loaded_models) < self.max_loaded_models:
                return
                
            # Find least recently used model
            lru_model = None
            oldest_time = float('inf')
            
            for model_type, stats in self.model_stats.items():
                try:
                    model_enum = RecognitionModel(model_type)
                    if model_enum in self.loaded_models:
                        if stats['last_used'] < oldest_time:
                            oldest_time = stats['last_used']
                            lru_model = model_enum
                except ValueError:
                    continue
            
            # Unload the LRU model
            if lru_model and lru_model != self.current_model_type:
                self.logger.info(f"🗑️ Unloading LRU model: {lru_model.value}")
                await self._unload_model(lru_model)
                
        except Exception as e:
            self.logger.error(f"❌ Error in model memory management: {e}")

    async def _unload_model(self, model_type: RecognitionModel) -> None:
        """Unload a specific model"""
        try:
            if model_type in self.loaded_models:
                del self.loaded_models[model_type]
                if model_type in self.model_load_order:
                    self.model_load_order.remove(model_type)
                
                # Clear GPU memory
                if TORCH_AVAILABLE and torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.logger.info(f"🗑️ Model {model_type.value} unloaded successfully")
                
        except Exception as e:
            self.logger.error(f"❌ Error unloading model {model_type.value}: {e}")

    async def _warmup_model(self, model_type: RecognitionModel, session: Any) -> None:
        """Warm up a specific model"""
        try:
            model_config = self.model_configs.get(model_type)
            if not model_config:
                return
                
            input_size = cast(Tuple[int, int], model_config["input_size"])
            dummy_input = np.random.randn(1, 3, input_size[1], input_size[0]).astype(np.float32)
            
            input_name = session.get_inputs()[0].name
            session.run(None, {input_name: dummy_input})
            
            self.logger.debug(f"🔥 Model {model_type.value} warmed up successfully")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Model warmup failed for {model_type.value}: {e}")

    def _configure_providers(self, model_type: RecognitionModel) -> List[Union[str, Tuple[str, Dict[str, Any]]]]:
        """Configure execution providers for ONNX"""
        providers: List[Union[str, Tuple[str, Dict[str, Any]]]] = []

        if (self.config.enable_gpu_optimization and 
            TORCH_AVAILABLE and torch and torch.cuda.is_available()):
            
            try:
                gpu_mem_limit = int(1.5 * 1024 * 1024 * 1024)  # 1.5GB per model
                cuda_options: Dict[str, Any] = {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                    "gpu_mem_limit": gpu_mem_limit,
                    "cudnn_conv_algo_search": "HEURISTIC",
                }
                providers.append(("CUDAExecutionProvider", cuda_options))
                self.logger.debug(f"🔥 CUDA provider configured for {model_type.value}")
            except Exception as e:
                self.logger.warning(f"⚠️ CUDA provider failed for {model_type.value}: {e}")

        providers.append("CPUExecutionProvider")
        return providers

    def _create_onnx_session(self, model_type: RecognitionModel, providers: List[Union[str, Tuple[str, Dict[str, Any]]]]) -> Optional[Any]:
        """Create ONNX inference session"""
        if not ort:
            return None
        
        model_config = self.model_configs.get(model_type)
        if not model_config:
            return None

        model_path = cast(str, model_config["model_path"])
        if not os.path.exists(model_path):
            self.logger.error(f"❌ Model file not found: {model_path}")
            return None

        try:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
            return session
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create ONNX session for {model_type.value}: {e}")
            return None

    async def add_face_from_image(
        self,
        image_bytes: bytes,
        person_name: str,
        person_id: Optional[str] = None,
        model_name: Optional[Union[str, RecognitionModel]] = None,
        use_distribution: bool = True
    ) -> Dict[str, Any]:
        """Add face with enhanced model selection"""
        if person_id is None:
            person_id = person_name

        try:
            # Decode image
            img_np = self._decode_image(image_bytes, person_id)
            if img_np is None:
                return {"success": False, "error": "Failed to decode image"}

            # Select model based on distribution or specific request
            selected_model = await self._select_model_for_task(model_name, use_distribution)
            
            # Extract embedding
            embedding_result = await self._extract_embedding_enhanced(img_np, selected_model)
            if not embedding_result["success"]:
                return embedding_result

            # Store in database
            face_id = f"{person_id}_{int(time.time())}"
            new_embedding = FaceEmbedding(
                id=face_id,
                person_id=person_id,
                person_name=person_name,
                vector=embedding_result["embedding"],
                model_type=selected_model,
                extraction_time=time.time(),
                quality_score=embedding_result.get("quality_score", 0.0),
                metadata={
                    "timestamp": time.time(),
                    "selected_by": "distribution" if use_distribution else "manual",
                    "processing_pipeline": "enhanced_v2"
                }
            )
            
            if person_id not in self.face_database:
                self.face_database[person_id] = []
            
            self.face_database[person_id].append(new_embedding)

            # Update statistics
            self._update_model_stats(selected_model, True, embedding_result.get("processing_time", 0.0))

            return {
                "success": True,
                "message": f"Face for {person_name} added successfully",
                "face_ids": [face_id],
                "person_id": person_id,
                "person_name": person_name,
                "model_used": selected_model.value,
                "embeddings_count": 1,
                "embedding_preview": embedding_result["embedding"][:5].tolist(),
                "full_embedding": embedding_result["embedding"].tolist(),
                "quality_score": embedding_result.get("quality_score", 0.0),
                "processing_time": embedding_result.get("processing_time", 0.0)
            }

        except Exception as e:
            self.logger.error(f"❌ Error adding face for {person_id}: {e}")
            return {"success": False, "error": str(e)}

    async def _select_model_for_task(
        self, 
        model_name: Optional[Union[str, RecognitionModel]] = None,
        use_distribution: bool = True
    ) -> RecognitionModel:
        """Select appropriate model for the task"""
        try:
            # If specific model requested
            if model_name:
                if isinstance(model_name, str):
                    requested_model = RecognitionModel(model_name.lower())
                else:
                    requested_model = model_name
                
                # Ensure model is loaded
                if requested_model not in self.loaded_models:
                    await self.load_model(requested_model)
                
                return requested_model
            
            # Use distribution-based selection
            if use_distribution:
                selected_model = self.model_distribution.get_random_model()
                
                # Ensure selected model is loaded
                if selected_model not in self.loaded_models:
                    await self.load_model(selected_model)
                
                return selected_model
            
            # Fallback to preferred model
            preferred_model = self.config.preferred_model
            if preferred_model not in self.loaded_models:
                await self.load_model(preferred_model)
            
            return preferred_model
            
        except Exception as e:
            self.logger.error(f"❌ Error selecting model: {e}")
            # Ultimate fallback
            return RecognitionModel.FACENET

    async def _extract_embedding_enhanced(
        self, 
        image: np.ndarray, 
        model_type: RecognitionModel
    ) -> Dict[str, Any]:
        """Extract embedding with enhanced processing"""
        start_time = time.time()
        
        try:
            # Ensure model is loaded and set as current
            if model_type not in self.loaded_models:
                success = await self.load_model(model_type)
                if not success:
                    return {"success": False, "error": f"Failed to load model {model_type.value}"}
            
            # Set current model
            self.current_model = self.loaded_models[model_type]
            self.current_model_type = model_type
            
            # Preprocess image
            preprocessed = self._preprocess_image(image, model_type)
            if preprocessed is None:
                return {"success": False, "error": "Image preprocessing failed"}
            
            # Extract embedding
            embedding = self._extract_embedding_onnx(preprocessed, model_type)
            if embedding is None:
                return {"success": False, "error": "Embedding extraction failed"}
            
            processing_time = time.time() - start_time
            
            # Calculate quality score (simplified)
            quality_score = self._calculate_quality_score(image, embedding)
            
            return {
                "success": True,
                "embedding": embedding,
                "model_used": model_type.value,
                "processing_time": processing_time,
                "quality_score": quality_score,
                "dimension": len(embedding)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced embedding extraction failed: {e}")
            return {"success": False, "error": str(e)}

    def _preprocess_image(self, image: np.ndarray, model_type: RecognitionModel) -> Optional[np.ndarray]:
        """Preprocess image for specific model"""
        try:
            model_config = self.model_configs.get(model_type)
            if not model_config:
                return None

            input_size = cast(Tuple[int, int], model_config["input_size"])
            mean = cast(List[float], model_config["mean"])
            std = cast(List[float], model_config["std"])

            # Resize
            img_resized = cv2.resize(image, input_size)
            
            # Normalize
            img_normalized = img_resized.astype(np.float32)
            
            if model_type == RecognitionModel.FACENET:
                img_normalized = (img_normalized - mean[0]) / std[0]
            else:
                img_normalized = (img_normalized - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)

            # Transpose to NCHW format
            img_transposed = np.transpose(img_normalized, (2, 0, 1))
            img_expanded = np.expand_dims(img_transposed, axis=0)
            
            return img_expanded
            
        except Exception as e:
            self.logger.error(f"❌ Image preprocessing failed for {model_type.value}: {e}")
            return None

    def _extract_embedding_onnx(self, preprocessed_image: np.ndarray, model_type: RecognitionModel) -> Optional[np.ndarray]:
        """Extract embedding using ONNX model"""
        try:
            if model_type not in self.loaded_models:
                self.logger.error(f"Model {model_type.value} not loaded")
                return None

            session = self.loaded_models[model_type]
            input_name = session.get_inputs()[0].name
            
            # Run inference
            ort_outs = session.run(None, {input_name: preprocessed_image})
            
            # Handle different output formats
            if model_type == RecognitionModel.ADAFACE and len(ort_outs) >= 2:
                embedding = np.array(ort_outs[0], dtype=np.float32).flatten()
            else:
                embedding = np.array(ort_outs[0], dtype=np.float32).flatten()
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"❌ ONNX embedding extraction failed for {model_type.value}: {e}")
            return None

    def _calculate_quality_score(self, image: np.ndarray, embedding: np.ndarray) -> float:
        """Calculate image quality score (simplified implementation)"""
        try:
            # Simple quality metrics
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Brightness check
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 127.5) / 127.5
            
            # Contrast check (using standard deviation)
            contrast = np.std(gray)
            contrast_score = min(contrast / 64.0, 1.0)
            
            # Embedding norm check
            embedding_norm = np.linalg.norm(embedding)
            embedding_score = min(embedding_norm, 1.0)
            
            # Combined score
            quality_score = (brightness_score * 0.3 + contrast_score * 0.4 + embedding_score * 0.3)
            
            return float(np.clip(quality_score, 0.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"❌ Quality score calculation failed: {e}")
            return 0.5

    def _update_model_stats(self, model_type: RecognitionModel, success: bool, processing_time: float) -> None:
        """Update model usage statistics"""
        try:
            stats = self.model_stats[model_type.value]
            stats["usage_count"] += 1
            stats["last_used"] = time.time()
            stats["total_time"] += processing_time
            
            if success:
                stats["success_count"] += 1
            
            if stats["usage_count"] > 0:
                stats["average_time"] = stats["total_time"] / stats["usage_count"]
                
        except Exception as e:
            self.logger.error(f"❌ Error updating model stats: {e}")

    async def recognize_faces(
        self,
        image_bytes: bytes,
        model_name: Optional[Union[str, RecognitionModel]] = None,
        use_distribution: bool = True
    ) -> Dict[str, Any]:
        """Enhanced face recognition with model distribution"""
        try:
            # Decode image
            img_np = self._decode_image(image_bytes, "recognition_task")
            if img_np is None:
                return {"success": False, "error": "Failed to decode image"}

            # Select model
            selected_model = await self._select_model_for_task(model_name, use_distribution)
            
            # Extract embedding
            embedding_result = await self._extract_embedding_enhanced(img_np, selected_model)
            if not embedding_result["success"]:
                return embedding_result

            # Compare with database
            matches = self._compare_with_database(embedding_result["embedding"], selected_model)
            
            # Update statistics
            self._update_model_stats(selected_model, True, embedding_result.get("processing_time", 0.0))

            # Sort matches by similarity
            matches = sorted(matches, key=lambda x: x["similarity"], reverse=True)
            best_match = matches[0] if matches else None

            return {
                "success": True,
                "matches": matches,
                "results": matches,  # Legacy field
                "best_match": best_match,
                "top_match": best_match,  # Legacy field
                "model_used": selected_model.value,
                "processing_time": embedding_result.get("processing_time", 0.0),
                "total_candidates": len(self.face_database),
                "message": f"Found {len(matches)} potential match(es)"
            }

        except Exception as e:
            self.logger.error(f"❌ Enhanced recognition failed: {e}")
            return {"success": False, "error": str(e)}

    def _compare_with_database(self, query_embedding: np.ndarray, model_type: RecognitionModel) -> List[Dict[str, Any]]:
        """Compare embedding with database"""
        matches = []
        
        for person_id, embeddings in self.face_database.items():
            best_similarity = 0.0
            best_embedding = None
            
            for face_embedding in embeddings:
                if face_embedding.model_type != model_type:
                    continue
                    
                similarity = self._cosine_similarity(query_embedding, face_embedding.vector)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_embedding = face_embedding
            
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

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        try:
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = np.dot(emb1, emb2) / (norm1 * norm2)
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"❌ Cosine similarity calculation failed: {e}")
            return 0.0

    def _decode_image(self, image_bytes: bytes, context: str) -> Optional[np.ndarray]:
        """Decode image bytes"""
        try:
            img_buffer = np.frombuffer(image_bytes, np.uint8)
            img_np = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
            return img_np
        except Exception as e:
            self.logger.error(f"❌ Image decoding failed for {context}: {e}")
            return None

    async def extract_embedding_only(
        self,
        image_bytes: bytes,
        model_name: Optional[Union[str, RecognitionModel]] = None,
        normalize: bool = True,
        use_distribution: bool = False
    ) -> Dict[str, Any]:
        """Extract embedding only without storing"""
        try:
            # Decode image
            img_np = self._decode_image(image_bytes, "embedding_extraction")
            if img_np is None:
                return {"success": False, "error": "Failed to decode image"}

            # Select model
            selected_model = await self._select_model_for_task(model_name, use_distribution)
            
            # Extract embedding
            embedding_result = await self._extract_embedding_enhanced(img_np, selected_model)
            
            # Update statistics
            self._update_model_stats(selected_model, embedding_result["success"], 
                                   embedding_result.get("processing_time", 0.0))

            if embedding_result["success"]:
                return {
                    "success": True,
                    "embedding": embedding_result["embedding"].tolist(),
                    "full_embedding": embedding_result["embedding"].tolist(),
                    "embedding_preview": embedding_result["embedding"][:5].tolist(),
                    "model_used": selected_model.value,
                    "dimension": len(embedding_result["embedding"]),
                    "normalized": normalize,
                    "processing_time": embedding_result.get("processing_time", 0.0),
                    "quality_score": embedding_result.get("quality_score", 0.0)
                }
            else:
                return embedding_result

        except Exception as e:
            self.logger.error(f"❌ Extract embedding only failed: {e}")
            return {"success": False, "error": str(e)}

    async def compare_faces(
        self,
        image1_bytes: bytes,
        image2_bytes: bytes,
        model_name: Optional[Union[str, RecognitionModel]] = None,
        use_distribution: bool = False
    ) -> Dict[str, Any]:
        """Compare two faces with enhanced processing"""
        try:
            # Extract embeddings for both images
            emb1_result = await self.extract_embedding_only(image1_bytes, model_name, use_distribution=use_distribution)
            emb2_result = await self.extract_embedding_only(image2_bytes, model_name, use_distribution=use_distribution)
            
            if not emb1_result["success"] or not emb2_result["success"]:
                return {
                    "success": False,
                    "error": "Failed to extract embeddings from one or both images"
                }

            # Calculate similarity
            emb1 = np.array(emb1_result["embedding"], dtype=np.float32)
            emb2 = np.array(emb2_result["embedding"], dtype=np.float32)
            
            similarity = self._cosine_similarity(emb1, emb2)
            threshold = 0.5
            is_match = similarity >= threshold

            return {
                "success": True,
                "similarity": float(similarity),
                "is_match": bool(is_match),
                "is_same_person": bool(is_match),
                "confidence": float(similarity),
                "distance": float(1.0 - similarity),
                "threshold_used": float(threshold),
                "model_used": emb1_result["model_used"],
                "processing_time": emb1_result.get("processing_time", 0.0) + emb2_result.get("processing_time", 0.0)
            }

        except Exception as e:
            self.logger.error(f"❌ Face comparison failed: {e}")
            return {"success": False, "error": str(e)}

    def get_service_info(self) -> Dict[str, Any]:
        """Get enhanced service information"""
        try:
            return {
                "service_status": "online" if self.loaded_models else "offline",
                "loaded_models": list(model.value for model in self.loaded_models.keys()),
                "current_model": self.current_model_type.value if self.current_model_type else None,
                "model_distribution": {
                    "facenet": f"{self.model_distribution.facenet_percentage}%",
                    "adaface": f"{self.model_distribution.adaface_percentage}%",
                    "arcface": f"{self.model_distribution.arcface_percentage}%"
                },
                "model_statistics": self.model_stats,
                "database_stats": {
                    "total_identities": len(self.face_database),
                    "total_embeddings": sum(len(embeddings) for embeddings in self.face_database.values())
                },
                "configuration": {
                    "max_loaded_models": self.max_loaded_models,
                    "enable_gpu_optimization": self.config.enable_gpu_optimization,
                    "similarity_threshold": self.config.similarity_threshold,
                    "unknown_threshold": self.config.unknown_threshold
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting service info: {e}")
            return {"service_status": "error", "error": str(e)}

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get enhanced performance statistics"""
        try:
            total_usage = sum(stats["usage_count"] for stats in self.model_stats.values())
            
            return {
                "model_statistics": self.model_stats,
                "distribution_stats": {
                    "total_requests": total_usage,
                    "facenet_usage": self.model_stats["facenet"]["usage_count"],
                    "adaface_usage": self.model_stats["adaface"]["usage_count"],
                    "arcface_usage": self.model_stats["arcface"]["usage_count"],
                    "actual_distribution": {
                        "facenet": f"{(self.model_stats['facenet']['usage_count'] / max(total_usage, 1)) * 100:.1f}%",
                        "adaface": f"{(self.model_stats['adaface']['usage_count'] / max(total_usage, 1)) * 100:.1f}%",
                        "arcface": f"{(self.model_stats['arcface']['usage_count'] / max(total_usage, 1)) * 100:.1f}%"
                    }
                },
                "memory_info": {
                    "loaded_models": len(self.loaded_models),
                    "max_models": self.max_loaded_models
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting performance stats: {e}")
            return {}

    def get_available_frameworks(self) -> List[str]:
        """Get available frameworks"""
        available = ["facenet", "adaface", "arcface"]
        
        if self.enable_multi_framework:
            for framework in self.requested_frameworks:
                if framework not in available:
                    try:
                        if framework == "deepface":
                            import deepface
                            available.append("deepface")
                        elif framework == "facenet_pytorch":
                            import facenet_pytorch
                            available.append("facenet_pytorch")
                        elif framework == "dlib":
                            import dlib
                            available.append("dlib")
                        elif framework == "insightface":
                            import insightface
                            available.append("insightface")
                        elif framework == "edgeface":
                            available.append("edgeface")
                    except ImportError:
                        pass
        
        return available

    async def shutdown(self) -> None:
        """Shutdown enhanced service"""
        try:
            self.logger.info("🛑 Shutting down Enhanced Face Recognition Service...")
            
            # Unload all models
            for model_type in list(self.loaded_models.keys()):
                await self._unload_model(model_type)
            
            # Clear database
            self.face_database.clear()
            
            # Clear statistics
            self.model_stats.clear()
            
            self.logger.info("✅ Enhanced Face Recognition Service shut down successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")

# Export the enhanced service
__all__ = ["FaceRecognitionEnhancedService"]