"""
Configuration Management for Face Recognition System
Centralized settings and model paths
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application Info
    app_name: str = "Face Recognition System"
    version: str = "2.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server Settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8080, env="PORT")
    
    # Model Paths (relative to project root)
    model_base_path: str = Field(default="models", env="MODEL_BASE_PATH")
    
    # Face Detection Models
    yolov9c_path: str = "models/face_detection/yolov9c-face-lindevs.onnx"
    yolov9e_path: str = "models/face_detection/yolov9e-face-lindevs.onnx"
    yolov11m_path: str = "models/face_detection/yolov11m-face.pt"
    
    # Face Recognition Models
    facenet_path: str = "models/face_recognition/facenet_vggface2.onnx"
    adaface_path: str = "models/face_recognition/adaface_ir101.onnx"
    arcface_path: str = "models/face_recognition/arcface_r100.onnx"
    
    # GPU Settings
    enable_gpu: bool = Field(default=True, env="ENABLE_GPU")
    reserved_vram_mb: int = Field(default=512, env="RESERVED_VRAM_MB")
    
    # Detection Settings
    default_confidence_threshold: float = 0.5
    default_iou_threshold: float = 0.4
    max_faces_per_image: int = 50
    min_face_size: int = 32
    
    # Recognition Settings
    similarity_threshold: float = 0.6
    embedding_dimension: int = 512
    max_gallery_size: int = 1000
    
    # Performance Settings
    batch_size: int = 8
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = "logs/app.log"
    
    # Output Settings
    output_base_path: str = "output"
    save_detection_images: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @property
    def project_root(self) -> Path:
        """Get project root directory"""
        return Path(__file__).parent.parent.parent
    
    @property
    def vram_config(self) -> Dict[str, Any]:
        """VRAM Manager configuration"""
        return {
            "reserved_vram_mb": self.reserved_vram_mb,
            "model_vram_estimates": {
                # Face Detection Models (in bytes)
                "yolov9c-face": 512 * 1024 * 1024,      # 512MB
                "yolov9e-face": 2048 * 1024 * 1024,     # 2GB
                "yolov11m-face": 2 * 1024 * 1024 * 1024, # 2GB
                
                # Face Recognition Models (in bytes)
                "facenet": 94 * 1024 * 1024,            # 94MB
                "adaface": 260 * 1024 * 1024,           # 260MB
                "arcface": 249 * 1024 * 1024,           # 249MB
            }
        }
    
    @property
    def detection_config(self) -> Dict[str, Any]:
        """Face Detection configuration"""
        return {
            "model_paths": {
                "yolov9c": str(self.project_root / self.yolov9c_path),
                "yolov9e": str(self.project_root / self.yolov9e_path),
                "yolov11m": str(self.project_root / self.yolov11m_path),
            },
            "default_model": "yolov9c",
            "confidence_threshold": self.default_confidence_threshold,
            "iou_threshold": self.default_iou_threshold,
            "max_faces": self.max_faces_per_image,
            "min_face_size": self.min_face_size,
            "enable_gpu": self.enable_gpu,
            "batch_size": self.batch_size,
            "fallback_strategy": [
                {"model": "yolov9c", "conf": 0.4},
                {"model": "yolov9e", "conf": 0.3},
                {"model": "yolov11m", "conf": 0.2},
            ]
        }
    
    @property
    def recognition_config(self) -> Dict[str, Any]:
        """Face Recognition configuration"""
        return {
            "model_paths": {
                "facenet": str(self.project_root / self.facenet_path),
                "adaface": str(self.project_root / self.adaface_path),
                "arcface": str(self.project_root / self.arcface_path),
            },
            "default_model": "facenet",
            "similarity_threshold": self.similarity_threshold,
            "embedding_dimension": self.embedding_dimension,
            "enable_gpu": self.enable_gpu,
            "batch_size": self.batch_size,
            "max_gallery_size": self.max_gallery_size,
        }
    
    @property
    def analysis_config(self) -> Dict[str, Any]:
        """Face Analysis configuration"""
        return {
            "enable_detection": True,
            "enable_recognition": True,
            "enable_quality_assessment": True,
            "quality_threshold": 0.5,
            "max_processing_time": self.request_timeout,
            "parallel_processing": True,
            "save_results": self.save_detection_images,
            "output_path": str(self.project_root / self.output_base_path),
        }
    
    def validate_model_paths(self) -> Dict[str, bool]:
        """Validate that all model files exist"""
        model_paths = {
            "yolov9c": self.project_root / self.yolov9c_path,
            "yolov9e": self.project_root / self.yolov9e_path,
            "yolov11m": self.project_root / self.yolov11m_path,
            "facenet": self.project_root / self.facenet_path,
            "adaface": self.project_root / self.adaface_path,
            "arcface": self.project_root / self.arcface_path,
        }
        
        return {name: path.exists() for name, path in model_paths.items()}


# Singleton settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get settings singleton"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_model_paths() -> Dict[str, str]:
    """Get all model file paths"""
    settings = get_settings()
    return {
        "detection": {
            "yolov9c": str(settings.project_root / settings.yolov9c_path),
            "yolov9e": str(settings.project_root / settings.yolov9e_path),
            "yolov11m": str(settings.project_root / settings.yolov11m_path),
        },
        "recognition": {
            "facenet": str(settings.project_root / settings.facenet_path),
            "adaface": str(settings.project_root / settings.adaface_path),
            "arcface": str(settings.project_root / settings.arcface_path),
        }
    }


def validate_environment() -> Dict[str, Any]:
    """Validate environment setup"""
    settings = get_settings()
    
    # Check model files
    model_status = settings.validate_model_paths()
    
    # Check directories
    directories = ["logs", "output", "models"]
    dir_status = {}
    
    for directory in directories:
        dir_path = settings.project_root / directory
        dir_status[directory] = dir_path.exists()
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                dir_status[directory] = True
            except Exception as e:
                dir_status[directory] = f"Error: {e}"
    
    # Check GPU availability
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass
    
    return {
        "model_files": model_status,
        "directories": dir_status,
        "gpu_available": gpu_available,
        "settings_valid": True
    }