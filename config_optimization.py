"""
Configuration Optimization for Face Recognition System
ปรับแต่งการตั้งค่าเพื่อลด timeout และปรับปรุงประสิทธิภาพ
"""

# แก้ไขไฟล์ src/core/config.py
# เพิ่มการตั้งค่าใหม่เหล่านี้

OPTIMIZED_CONFIG = {
    # VRAM Manager - ปรับลดการใช้ VRAM
    "vram_config": {
        "reserved_vram_mb": 256,  # ลดจาก 512 เป็น 256
        "model_vram_estimates": {
            "yolov9c-face": 512 * 1024 * 1024,   # ลดเป็น 512MB
            "yolov9e-face": 768 * 1024 * 1024,   # ลดเป็น 768MB
            "yolov11m-face": 512 * 1024 * 1024,  # ลดเป็น 512MB
            "facenet-face-recognition": 256 * 1024 * 1024,   # ลดเป็น 256MB
            "adaface-face-recognition": 256 * 1024 * 1024,   # ลดเป็น 256MB
            "arcface-face-recognition": 256 * 1024 * 1024,   # ลดเป็น 256MB
        }
    },
    
    # Face Recognition - ปรับการตั้งค่าเพื่อประสิทธิภาพ
    "recognition_config": {
        "preferred_model": "facenet",
        "similarity_threshold": 0.50,  # ลดจาก 0.60
        "unknown_threshold": 0.40,     # ลดจาก 0.55
        "batch_size": 4,               # ลดจาก 8
        "enable_gpu_optimization": True,
        "cuda_memory_fraction": 0.6,   # ลดจาก 0.8
        "parallel_processing": False,  # เปลี่ยนเป็น sequential
        "enable_multi_framework": True,
        "frameworks": ["deepface", "facenet_pytorch"],  # ลดจำนวน frameworks
        "max_loaded_models": 2,  # จำกัดการโหลดโมเดลพร้อมกัน
    },
    
    # Face Detection - ปรับเพื่อความเร็ว
    "detection_config": {
        "conf_threshold": 0.15,        # เพิ่มจาก 0.10
        "iou_threshold_nms": 0.40,     # เพิ่มจาก 0.35
        "max_usable_faces_yolov9": 8,  # ลดจาก 12
        "min_quality_threshold": 30,   # ลดจาก 40
        "fallback_config": {
            "enable_fallback_system": True,
            "max_fallback_attempts": 2,  # ลดจาก 3
        }
    },
    
    # API Configuration - เพิ่ม timeout
    "api_config": {
        "max_upload_size": 5 * 1024 * 1024,  # ลดเป็น 5MB
        "request_timeout": 300,  # 5 นาที
        "connection_timeout": 60,  # 1 นาที
    }
}

# ฟังก์ชันสำหรับปรับใช้ config ใหม่
def apply_optimized_config():
    """
    ใช้การตั้งค่าที่ปรับปรุงแล้ว
    เรียกใช้ฟังก์ชันนี้ก่อนเริ่ม services
    """
    import os
    
    # ตั้งค่า environment variables
    os.environ["FACE_RECOGNITION_GPU_MEMORY_FRACTION"] = "0.6"
    os.environ["FACE_RECOGNITION_BATCH_SIZE"] = "4"
    os.environ["FACE_RECOGNITION_PARALLEL_PROCESSING"] = "false"
    os.environ["FACE_RECOGNITION_MAX_MODELS"] = "2"
    
    print("✅ Applied optimized configuration")
    return OPTIMIZED_CONFIG

if __name__ == "__main__":
    config = apply_optimized_config()
    print("📋 Optimized Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")