"""
Core Configuration Module
Settings and configuration management for the Face Recognition System
"""

import os
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class Settings:
    """Application settings and configuration"""

    # Project Information
    project_name: str = "Face Recognition System"
    version: str = "2.0.0"
    description: str = (
        "Professional Face Detection, Recognition & Analysis System "
        "with GPU optimization"
    )

    # API Configuration
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = True
    reload: bool = True
    log_level: str = "info"
    cors_origins: list = field(default_factory=lambda: ["*"])

    # VRAM Manager Configuration
    vram_config: Dict[str, Any] = field(default_factory=lambda: {
        "reserved_vram_mb": 512,
        "max_history_size": 1000,
        "model_vram_estimates": {
            "yolov9c-face": 1024 * 1024 * 1024,  # 1GB
            "yolov9e-face": 1536 * 1024 * 1024,  # 1.5GB
            "yolov11m-face": 1024 * 1024 * 1024,  # 1GB
            "facenet-face-recognition": 512 * 1024 * 1024,  # 512MB
            "adaface-face-recognition": 512 * 1024 * 1024,  # 512MB
            "arcface-face-recognition": 512 * 1024 * 1024,  # 512MB
        }
    })

    # Face Detection Service Configuration
    detection_config: Dict[str, Any] = field(default_factory=lambda: {
        "use_enhanced_detector": False,
        "yolov9c_model_path": "model/face-detection/yolov9c-face-lindevs.onnx",
        "yolov9e_model_path": "model/face-detection/yolov9e-face-lindevs.onnx",
        "yolov11m_model_path": "model/face-detection/yolov11m-face.pt",
        "max_usable_faces_yolov9": 12,
        "min_agreement_ratio": 0.5,
        "min_quality_threshold": 40,
        "iou_threshold_agreement": 0.3,
        "conf_threshold": 0.10,
        "iou_threshold_nms": 0.35,
        "img_size": 640,
        "quality_config": {
            "min_quality_threshold": 40,
            "size_weight": 30,
            "area_weight": 25,
            "confidence_weight": 30,
            "aspect_weight": 15,
            "excellent_size": (80, 80),
            "good_size": (50, 50),
            "acceptable_size": (24, 24),
            "minimum_size": (8, 8),
            "bonus_score_for_high_confidence": 5.0,
            "high_confidence_threshold": 0.7,
        },
        "fallback_config": {
            "enable_fallback_system": True,
            "max_fallback_attempts": 3,
            "fallback_models": [
                {
                    "model_name": "yolov11m",
                    "conf_threshold": 0.15,
                    "iou_threshold": 0.35,
                    "min_faces_to_accept": 1,
                },
                {
                    "model_name": "yolov9c",
                    "conf_threshold": 0.05,
                    "iou_threshold": 0.3,
                    "min_faces_to_accept": 1,
                },
                {
                    "model_name": "opencv_haar",
                    "scale_factor": 1.1,
                    "min_neighbors": 3,
                    "min_size": (20, 20),
                    "min_faces_to_accept": 1,
                },
            ],
            "min_detections_after_fallback": 1,
            "always_run_all_fallbacks_if_zero_initial": True,
        },
        "filter_min_quality": 30.0,
        "filter_min_quality_final": 40.0,
    })    # Face Recognition Service Configuration
    recognition_config: Dict[str, Any] = field(default_factory=lambda: {
        "preferred_model": "facenet",
        "similarity_threshold": 0.60,
        "unknown_threshold": 0.55,
        "embedding_dimension": 512,
        "enable_gpu_optimization": True,
        "batch_size": 8,
        "quality_threshold": 0.2,
        "cuda_memory_fraction": 0.8,
        "use_cuda_graphs": False,
        "parallel_processing": True,
        "enable_quality_assessment": True,
        "auto_model_selection": True,
        "enable_unknown_detection": True,
        "enable_multi_framework": True,
        "frameworks": ["deepface", "facenet_pytorch", "dlib", "insightface", "edgeface"],
        "models_path": "./model/face-recognition/"
    })

    # Face Analysis Service Configuration
    analysis_config: Dict[str, Any] = field(default_factory=lambda: {
        "detection": {},  # Will use detection_config
        "recognition": {},  # Will use recognition_config
        "default_mode": "full_analysis",
        "default_quality_level": "balanced",
        "enable_batch_processing": True,
        "max_batch_size": 16,
        "enable_parallel_processing": True,
        "default_confidence_threshold": 0.5,
        "default_max_faces": 50,
        "default_gallery_top_k": 5,
    })

    # API Configuration Details
    api_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_upload_size": 10 * 1024 * 1024,  # 10MB
        "allowed_image_types": ["image/jpeg", "image/png", "image/jpg"],
        "docs_url": "/docs",
        "redoc_url": "/redoc",
    })

    # Logging Configuration
    logging_config: Dict[str, Any] = field(default_factory=lambda: {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "log_file": "logs/app.log",
        "max_file_size": 10 * 1024 * 1024,  # 10MB
        "backup_count": 5,
        "enable_console_logging": True,
        "enable_file_logging": True,
    })


def get_settings() -> Settings:
    """
    Get application settings with environment variable overrides
    """
    settings = Settings()

    # Override from environment variables
    if host_env := os.getenv("FACE_RECOGNITION_HOST"):
        settings.host = host_env

    if port_env := os.getenv("FACE_RECOGNITION_PORT"):
        try:
            settings.port = int(port_env)
        except ValueError:
            pass

    if log_level_env := os.getenv("FACE_RECOGNITION_LOG_LEVEL"):
        log_level = log_level_env.upper()
        if log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            settings.log_level = log_level.lower()

    if gpu_env := os.getenv("FACE_RECOGNITION_GPU_ENABLED"):
        gpu_enabled = gpu_env.lower()
        if gpu_enabled in ["true", "1", "yes"]:
            settings.recognition_config["enable_gpu_optimization"] = True
        elif gpu_enabled in ["false", "0", "no"]:
            settings.recognition_config["enable_gpu_optimization"] = False

    if model_dir_env := os.getenv("FACE_RECOGNITION_MODEL_DIR"):
        # Update model paths
        settings.detection_config.update({
            "yolov9c_model_path": os.path.join(
                model_dir_env, "face-detection/yolov9c-face-lindevs.onnx"
            ),
            "yolov9e_model_path": os.path.join(
                model_dir_env, "face-detection/yolov9e-face-lindevs.onnx"
            ),
            "yolov11m_model_path": os.path.join(
                model_dir_env, "face-detection/yolov11m-face.pt"
            ),
        })

    return settings


def get_model_paths() -> Dict[str, str]:
    """Get model file paths"""
    settings = get_settings()

    return {
        "yolov9c": settings.detection_config["yolov9c_model_path"],
        "yolov9e": settings.detection_config["yolov9e_model_path"],
        "yolov11m": settings.detection_config["yolov11m_model_path"],
        "facenet": "model/face-recognition/facenet_vggface2.onnx",
        "adaface": "model/face-recognition/adaface_ir101.onnx",
        "arcface": "model/face-recognition/arcface_r100.onnx",
    }


def validate_model_files() -> Dict[str, bool]:
    """Validate that required model files exist"""
    model_paths = get_model_paths()
    validation_results = {}

    for model_name, model_path in model_paths.items():
        validation_results[model_name] = os.path.exists(model_path)

    return validation_results


def create_required_directories() -> None:
    """Create required directories for the application"""
    directories = [
        "logs",
        "output",
        "output/detection",
        "output/recognition",
        "output/analysis",
        "model",
        "model/face-detection",
        "model/face-recognition",
        "temp",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# Environment-specific configurations
def get_development_settings() -> Settings:
    """Get settings optimized for development"""
    settings = get_settings()
    settings.debug = True
    settings.reload = True
    settings.log_level = "debug"
    settings.logging_config["level"] = "DEBUG"
    return settings


def get_production_settings() -> Settings:
    """Get settings optimized for production"""
    settings = get_settings()
    settings.debug = False
    settings.reload = False
    settings.log_level = "info"
    settings.logging_config["level"] = "INFO"
    return settings


def get_testing_settings() -> Settings:
    """Get settings optimized for testing"""
    settings = get_settings()
    settings.port = 8081  # Different port for testing
    settings.log_level = "warning"
    settings.logging_config["level"] = "WARNING"
    # Disable GPU for testing to avoid conflicts
    settings.recognition_config["enable_gpu_optimization"] = False
    return settings