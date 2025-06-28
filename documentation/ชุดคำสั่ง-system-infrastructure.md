# ชุดคำสั่ง: System Infrastructure & Common Services
## โครงสร้างระบบและบริการสาธารณะ

### 📋 สารบัญ
8.1 [ภาพรวม System Infrastructure](#81-ภาพรวม-system-infrastructure)
8.2 [VRAM Manager](#82-vram-manager)
8.3 [Base Service Class](#83-base-service-class)
8.4 [Common Utilities](#84-common-utilities)
8.5 [Dependency Injection](#85-dependency-injection)
8.6 [Service Statistics](#86-service-statistics)
8.7 [Performance Monitoring](#87-performance-monitoring)
8.8 [Error Handling](#88-error-handling)

---

## 8.1 ภาพรวม System Infrastructure

ระบบโครงสร้างพื้นฐานที่รองรับการทำงานของ AI Services ทั้งหมด รวมถึงการจัดการ GPU Memory, Base Classes, Utilities และ Performance Monitoring

### 🏗️ สถาปัตยกรรม
```
Common Services
├── VRAM Manager (GPU Memory Management)
├── Base Service Class (Abstract Interface)
├── Utilities (Helper Functions)
├── Statistics (Performance Tracking)
└── Error Handling (Centralized Error Management)
```

### 🎯 ฟีเจอร์หลัก
- **VRAM Management**: จัดการ GPU Memory อย่างมีประสิทธิภาพ
- **Service Architecture**: Base class สำหรับ AI Services
- **Performance Monitoring**: ติดตามการใช้งานและประสิทธิภาพ
- **Resource Management**: จัดการทรัพยากรระบบ
- **Error Handling**: จัดการข้อผิดพลาดแบบรวมศูนย์

---

## 8.2 VRAM Manager

### 8.2.1 VRAM Manager Class

### 🔧 VRAM Manager Class
```python
from src.ai_services.common.vram_manager import VRAMManager, AllocationPriority, AllocationLocation
from dataclasses import dataclass
from enum import Enum

class AllocationPriority(Enum):
    """Memory allocation priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class AllocationLocation(Enum):
    """Memory allocation location"""
    GPU = "gpu"
    CPU = "cpu"

@dataclass
class ModelAllocation:
    """Model memory allocation information"""
    model_id: str
    priority: AllocationPriority
    service_id: str
    location: AllocationLocation
    vram_allocated: int
    status: str
    timestamp: float = 0.0

class VRAMManager:
    """GPU Memory Manager for AI Models"""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model_allocations: Dict[str, ModelAllocation] = {}
        self.total_vram = self._get_total_vram()
        self.allocated_vram = 0
        self.lock = asyncio.Lock()
```

### 8.2.2 VRAM Configuration
```python
# การตั้งค่า VRAM Manager
vram_config = {
    "max_vram_usage": 0.8,  # ใช้ VRAM สูงสุด 80%
    "reserve_vram_mb": 1024,  # สำรอง VRAM 1GB
    "allocation_strategy": "priority_based",
    "cleanup_threshold": 0.9,  # เมื่อใช้ถึง 90% จะทำ cleanup
    "enable_monitoring": True,
    "log_allocations": True,
}

# สร้าง VRAM Manager
vram_manager = VRAMManager(vram_config)
await vram_manager.initialize()
```

### 8.2.3 การใช้งาน VRAM Manager
```python
async def vram_management_example():
    """ตัวอย่างการจัดการ VRAM"""
    
    # ตรวจสอบสถานะ VRAM
    vram_info = await vram_manager.get_memory_info()
    print(f"VRAM ทั้งหมด: {vram_info['total_vram_gb']:.1f} GB")
    print(f"VRAM ที่ใช้: {vram_info['allocated_vram_gb']:.1f} GB")
    print(f"VRAM ว่าง: {vram_info['free_vram_gb']:.1f} GB")
    
    # จองพื้นที่ VRAM สำหรับโมเดล
    allocation_request = {
        "model_id": "yolov11m-face",
        "service_id": "face_detection",
        "priority": AllocationPriority.HIGH,
        "estimated_vram_mb": 512,
        "location": AllocationLocation.GPU
    }
    
    allocation_result = await vram_manager.allocate_memory(allocation_request)
    
    if allocation_result["success"]:
        print(f"✅ จองพื้นที่ VRAM สำเร็จ: {allocation_result['allocated_mb']} MB")
    else:
        print(f"❌ ไม่สามารถจองพื้นที่ VRAM: {allocation_result['error']}")
    
    # ดูรายการการจอง
    allocations = await vram_manager.get_allocations()
    for alloc_id, allocation in allocations.items():
        print(f"โมเดล: {allocation.model_id}")
        print(f"  บริการ: {allocation.service_id}")
        print(f"  ความสำคัญ: {allocation.priority.value}")
        print(f"  VRAM: {allocation.vram_allocated / (1024*1024):.1f} MB")
        print(f"  สถานะ: {allocation.status}")

async def vram_cleanup_example():
    """ตัวอย่างการทำความสะอาด VRAM"""
    
    # ตรวจสอบ VRAM usage
    usage_percentage = await vram_manager.get_usage_percentage()
    print(f"การใช้ VRAM: {usage_percentage:.1f}%")
    
    if usage_percentage > 85.0:
        print("🧹 กำลังทำความสะอาด VRAM...")
        
        # ปล่อย VRAM ของโมเดลที่ไม่ได้ใช้
        cleanup_result = await vram_manager.cleanup_unused_models()
        print(f"ปล่อย VRAM: {cleanup_result['freed_mb']} MB")
        
        # บังคับ garbage collection
        await vram_manager.force_garbage_collection()
        
        # ตรวจสอบ usage ใหม่
        new_usage = await vram_manager.get_usage_percentage()
        print(f"การใช้ VRAM หลัง cleanup: {new_usage:.1f}%")

async def vram_priority_management():
    """การจัดการความสำคัญของการใช้ VRAM"""
    
    # จอง VRAM สำหรับโมเดลความสำคัญสูง
    critical_models = [
        {
            "model_id": "face_detection_primary",
            "service_id": "face_detection",
            "priority": AllocationPriority.CRITICAL,
            "estimated_vram_mb": 512
        },
        {
            "model_id": "face_recognition_primary", 
            "service_id": "face_recognition",
            "priority": AllocationPriority.HIGH,
            "estimated_vram_mb": 1024
        }
    ]
    
    for model_request in critical_models:
        result = await vram_manager.allocate_memory(model_request)
        if result["success"]:
            print(f"✅ จองพื้นที่สำหรับ {model_request['model_id']}")
        else:
            print(f"❌ ไม่สามารถจองพื้นที่สำหรับ {model_request['model_id']}")
    
    # เมื่อ VRAM เต็ม ระบบจะปล่อยโมเดลความสำคัญต่ำออกก่อน
    if await vram_manager.get_usage_percentage() > 90:
        freed_allocations = await vram_manager.free_low_priority_models()
        print(f"ปล่อยโมเดลความสำคัญต่ำ: {len(freed_allocations)} โมเดล")
```

---

## 8.3 Base Service Class

### 8.3.1 Base Service Implementation
```python
from src.ai_services.common.base_service import BaseAIService
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseAIService(ABC):
    """Base class for all AI services"""
    
    def __init__(self) -> None:
        self.service_name = "base_service"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._is_initialized = False
        self.stats = {
            "initialization_time": 0.0,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_processing_time": 0.0,
            "processing_times": []
        }
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the service"""
        pass
    
    async def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        return {
            "service_name": self.service_name,
            "class_name": self.__class__.__name__,
            "initialized": self._is_initialized,
            "stats": self.stats
        }
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        self.logger.info(f"Cleaning up {self.service_name}")
    
    def update_stats(self, processing_time: float, success: bool = True) -> None:
        """Update service statistics"""
        self.stats["total_requests"] += 1
        
        if success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
        
        self.stats["processing_times"].append(processing_time)
        
        # Calculate average processing time
        if self.stats["processing_times"]:
            self.stats["average_processing_time"] = (
                sum(self.stats["processing_times"]) / 
                len(self.stats["processing_times"])
            )
```

### 8.3.2 Service Implementation Example
```python
class ExampleAIService(BaseAIService):
    """ตัวอย่างการสร้าง AI Service"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.service_name = "example_ai_service"
        self.config = config or {}
        self.model = None
        
    async def initialize(self) -> bool:
        """Initialize the service"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Initializing {self.service_name}...")
            
            # Load model (example)
            self.model = await self._load_model()
            
            self._is_initialized = True
            self.stats["initialization_time"] = time.time() - start_time
            
            self.logger.info(f"✅ {self.service_name} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize {self.service_name}: {e}")
            return False
    
    async def process(self, input_data: Any) -> Dict[str, Any]:
        """Process input data"""
        if not self._is_initialized:
            raise RuntimeError(f"{self.service_name} not initialized")
        
        start_time = time.time()
        
        try:
            # Process data
            result = await self._process_data(input_data)
            
            processing_time = time.time() - start_time
            self.update_stats(processing_time, success=True)
            
            return {
                "success": True,
                "result": result,
                "processing_time": processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_stats(processing_time, success=False)
            
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time
            }
    
    async def _load_model(self):
        """Load AI model (implement in subclass)"""
        # Simulate model loading
        await asyncio.sleep(2)
        return "mock_model"
    
    async def _process_data(self, input_data: Any):
        """Process data with model (implement in subclass)"""
        # Simulate processing
        await asyncio.sleep(0.1)
        return f"processed_{input_data}"
```

---

## 8.4 Common Utilities

### 8.4.1 Utility Functions
```python
from src.ai_services.common.utils import *
import cv2
import numpy as np
from typing import Union, Tuple, List

# Image processing utilities
def resize_image_keep_aspect(
    image: np.ndarray, 
    target_size: Union[int, Tuple[int, int]]
) -> np.ndarray:
    """ปรับขนาดรูปภาพโดยรักษาอัตราส่วน"""
    
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    
    height, width = image.shape[:2]
    target_width, target_height = target_size
    
    # Calculate scaling factor
    scale = min(target_width / width, target_height / height)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Create canvas with target size
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Center the resized image on canvas
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return canvas

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """คำนวณ Intersection over Union (IoU)"""
    
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_intersect = max(x1_1, x1_2)
    y1_intersect = max(y1_1, y1_2)
    x2_intersect = min(x2_1, x2_2)
    y2_intersect = min(y2_1, y2_2)
    
    if x2_intersect <= x1_intersect or y2_intersect <= y1_intersect:
        return 0.0
    
    intersection_area = (x2_intersect - x1_intersect) * (y2_intersect - y1_intersect)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def normalize_bbox(bbox: List[float], image_shape: Tuple[int, int]) -> List[float]:
    """Normalize bounding box coordinates to [0, 1]"""
    
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    
    return [
        x1 / width,
        y1 / height,
        x2 / width,
        y2 / height
    ]

def denormalize_bbox(normalized_bbox: List[float], image_shape: Tuple[int, int]) -> List[int]:
    """Denormalize bounding box coordinates to pixel values"""
    
    height, width = image_shape[:2]
    x1, y1, x2, y2 = normalized_bbox
    
    return [
        int(x1 * width),
        int(y1 * height),
        int(x2 * width),
        int(y2 * height)
    ]

# Performance utilities
class Timer:
    """Context manager สำหรับการวัดเวลา"""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        print(f"{self.description}: {self.elapsed_time:.3f}s")

# การใช้งาน Timer
async def performance_measurement_example():
    """ตัวอย่างการวัดประสิทธิภาพ"""
    
    # วัดเวลาการประมวลผล
    with Timer("Face Detection"):
        detection_result = await face_detection_service.detect_faces(image)
    
    with Timer("Face Recognition"):
        recognition_result = await face_recognition_service.recognize_faces(faces)
    
    # วัดเวลาด้วย decorator
    @measure_time
    async def process_image(image_path: str):
        image = cv2.imread(image_path)
        result = await face_analysis_service.analyze_faces(image)
        return result
    
    result = await process_image("test.jpg")

def measure_time(func):
    """Decorator สำหรับการวัดเวลา"""
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"{func.__name__}: {elapsed_time:.3f}s")
        return result
    return wrapper
```

### 8.4.2 Data Validation Utilities
```python
# Validation utilities
def validate_image_input(image: np.ndarray) -> bool:
    """ตรวจสอบความถูกต้องของ input image"""
    
    if image is None:
        return False
    
    if not isinstance(image, np.ndarray):
        return False
    
    if len(image.shape) not in [2, 3]:
        return False
    
    if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
        return False
    
    if image.size == 0:
        return False
    
    return True

def validate_bbox(bbox: List[float], image_shape: Tuple[int, int]) -> bool:
    """ตรวจสอบความถูกต้องของ bounding box"""
    
    if len(bbox) != 4:
        return False
    
    x1, y1, x2, y2 = bbox
    height, width = image_shape[:2]
    
    # Check bounds
    if x1 < 0 or y1 < 0 or x2 >= width or y2 >= height:
        return False
    
    # Check ordering
    if x1 >= x2 or y1 >= y2:
        return False
    
    return True

def sanitize_filename(filename: str) -> str:
    """ทำความสะอาดชื่อไฟล์"""
    
    import re
    
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip(' .')
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:255-len(ext)] + ext
    
    return sanitized
```

---

## 8.5 Dependency Injection

### 8.5.1 Service Dependencies
```python
# service_container.py
class ServiceContainer:
    """Container สำหรับการจัดการ dependencies"""
    
    def __init__(self):
        self.services = {}
        self.configs = {}
    
    def register_service(self, name: str, service_class, config: Dict[str, Any] = None):
        """ลงทะเบียน service"""
        self.services[name] = {
            "class": service_class,
            "config": config or {},
            "instance": None,
            "initialized": False
        }
    
    async def get_service(self, name: str):
        """ดึง service instance"""
        if name not in self.services:
            raise ValueError(f"Service '{name}' not registered")
        
        service_info = self.services[name]
        
        if service_info["instance"] is None:
            # Create instance
            service_info["instance"] = service_info["class"](service_info["config"])
            
            # Initialize if needed
            if not service_info["initialized"]:
                await service_info["instance"].initialize()
                service_info["initialized"] = True
        
        return service_info["instance"]
    
    async def initialize_all(self):
        """Initialize all registered services"""
        for name, service_info in self.services.items():
            await self.get_service(name)

# การใช้งาน Service Container
async def setup_service_container():
    """ตั้งค่า service container"""
    
    container = ServiceContainer()
    
    # Register services
    container.register_service(
        "vram_manager",
        VRAMManager,
        {"max_vram_usage": 0.8, "reserve_vram_mb": 1024}
    )
    
    container.register_service(
        "face_detection",
        FaceDetectionService,
        {"model_path": "model/face-detection/yolov11m-face.pt"}
    )
    
    container.register_service(
        "face_recognition", 
        FaceRecognitionService,
        {"model_path": "model/face-recognition/adaface_ir101.onnx"}
    )
    
    # Initialize all services
    await container.initialize_all()
    
    # Get services
    vram_manager = await container.get_service("vram_manager")
    face_detection = await container.get_service("face_detection")
    face_recognition = await container.get_service("face_recognition")
    
    return container
```

---

## 8.6 Service Statistics

### 8.6.1 Statistics Collection
```python
from src.ai_services.common.stats import ServiceStats
from dataclasses import dataclass
from typing import List, Dict, Any
import time

@dataclass
class RequestStat:
    """สถิติการร้องขอ"""
    timestamp: float
    processing_time: float
    success: bool
    service_name: str
    method_name: str
    error: str = None

class ServiceStats:
    """การเก็บสถิติของ services"""
    
    def __init__(self, max_records: int = 1000):
        self.max_records = max_records
        self.request_history: List[RequestStat] = []
        self.service_stats: Dict[str, Dict[str, Any]] = {}
    
    def record_request(
        self,
        service_name: str,
        method_name: str,
        processing_time: float,
        success: bool,
        error: str = None
    ):
        """บันทึกการร้องขอ"""
        
        stat = RequestStat(
            timestamp=time.time(),
            processing_time=processing_time,
            success=success,
            service_name=service_name,
            method_name=method_name,
            error=error
        )
        
        self.request_history.append(stat)
        
        # Keep only recent records
        if len(self.request_history) > self.max_records:
            self.request_history = self.request_history[-self.max_records:]
        
        # Update service stats
        self._update_service_stats(stat)
    
    def _update_service_stats(self, stat: RequestStat):
        """อัปเดตสถิติของ service"""
        
        if stat.service_name not in self.service_stats:
            self.service_stats[stat.service_name] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_processing_time": 0.0,
                "average_processing_time": 0.0,
                "min_processing_time": float('inf'),
                "max_processing_time": 0.0,
                "methods": {}
            }
        
        stats = self.service_stats[stat.service_name]
        stats["total_requests"] += 1
        
        if stat.success:
            stats["successful_requests"] += 1
        else:
            stats["failed_requests"] += 1
        
        stats["total_processing_time"] += stat.processing_time
        stats["average_processing_time"] = (
            stats["total_processing_time"] / stats["total_requests"]
        )
        
        stats["min_processing_time"] = min(
            stats["min_processing_time"], stat.processing_time
        )
        stats["max_processing_time"] = max(
            stats["max_processing_time"], stat.processing_time
        )
        
        # Method-specific stats
        if stat.method_name not in stats["methods"]:
            stats["methods"][stat.method_name] = {
                "count": 0,
                "success_count": 0,
                "total_time": 0.0
            }
        
        method_stats = stats["methods"][stat.method_name]
        method_stats["count"] += 1
        method_stats["total_time"] += stat.processing_time
        
        if stat.success:
            method_stats["success_count"] += 1
    
    def get_service_summary(self, service_name: str) -> Dict[str, Any]:
        """ดึงสรุปสถิติของ service"""
        
        if service_name not in self.service_stats:
            return {"error": f"No stats for service '{service_name}'"}
        
        stats = self.service_stats[service_name]
        success_rate = (
            stats["successful_requests"] / stats["total_requests"] 
            if stats["total_requests"] > 0 else 0.0
        )
        
        return {
            "service_name": service_name,
            "total_requests": stats["total_requests"],
            "success_rate": success_rate,
            "average_processing_time": stats["average_processing_time"],
            "min_processing_time": stats["min_processing_time"],
            "max_processing_time": stats["max_processing_time"],
            "methods": stats["methods"]
        }
    
    def get_overall_summary(self) -> Dict[str, Any]:
        """ดึงสรุปสถิติรวม"""
        
        total_requests = sum(
            stats["total_requests"] 
            for stats in self.service_stats.values()
        )
        
        total_successful = sum(
            stats["successful_requests"]
            for stats in self.service_stats.values()
        )
        
        overall_success_rate = (
            total_successful / total_requests 
            if total_requests > 0 else 0.0
        )
        
        return {
            "total_requests": total_requests,
            "overall_success_rate": overall_success_rate,
            "active_services": len(self.service_stats),
            "services": list(self.service_stats.keys())
        }

# การใช้งาน Statistics
global_stats = ServiceStats()

def track_performance(service_name: str, method_name: str):
    """Decorator สำหรับการติดตามประสิทธิภาพ"""
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error = None
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                processing_time = time.time() - start_time
                global_stats.record_request(
                    service_name=service_name,
                    method_name=method_name,
                    processing_time=processing_time,
                    success=success,
                    error=error
                )
        
        return wrapper
    return decorator

# ใช้งาน decorator
class TrackedFaceDetectionService(BaseAIService):
    
    @track_performance("face_detection", "detect_faces")
    async def detect_faces(self, image: np.ndarray):
        # Implementation here
        pass
```

---

## 8.7 Performance Monitoring

### 📈 Real-time Monitoring
```python
async def start_performance_monitoring():
    """เริ่ม monitoring แบบ real-time"""
    
    async def monitor_loop():
        while True:
            # ตรวจสอบ VRAM usage
            vram_info = await vram_manager.get_memory_info()
            
            if vram_info["usage_percentage"] > 90:
                logger.warning(f"🚨 High VRAM usage: {vram_info['usage_percentage']:.1f}%")
                await vram_manager.cleanup_unused_models()
            
            # ตรวจสอบ service performance
            overall_stats = global_stats.get_overall_summary()
            
            if overall_stats["overall_success_rate"] < 0.95:
                logger.warning(f"🚨 Low success rate: {overall_stats['overall_success_rate']:.1%}")
            
            # Log statistics every 5 minutes
            current_time = time.time()
            if current_time % 300 < 1:  # Every 5 minutes
                await log_performance_summary()
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    # Start monitoring task
    asyncio.create_task(monitor_loop())

async def log_performance_summary():
    """บันทึกสรุปประสิทธิภาพ"""
    
    logger.info("📊 Performance Summary:")
    
    # VRAM info
    vram_info = await vram_manager.get_memory_info()
    logger.info(f"  VRAM Usage: {vram_info['usage_percentage']:.1f}%")
    
    # Service stats
    overall_stats = global_stats.get_overall_summary()
    logger.info(f"  Total Requests: {overall_stats['total_requests']}")
    logger.info(f"  Success Rate: {overall_stats['overall_success_rate']:.1%}")
    
    # Individual service performance
    for service_name in overall_stats["services"]:
        service_stats = global_stats.get_service_summary(service_name)
        logger.info(f"  {service_name}:")
        logger.info(f"    Requests: {service_stats['total_requests']}")
        logger.info(f"    Success Rate: {service_stats['success_rate']:.1%}")
        logger.info(f"    Avg Time: {service_stats['average_processing_time']:.3f}s")

async def generate_performance_report():
    """สร้างรายงานประสิทธิภาพ"""
    
    report = {
        "timestamp": time.time(),
        "vram_info": await vram_manager.get_memory_info(),
        "overall_stats": global_stats.get_overall_summary(),
        "service_details": {}
    }
    
    # Get detailed stats for each service
    for service_name in report["overall_stats"]["services"]:
        report["service_details"][service_name] = global_stats.get_service_summary(service_name)
    
    # Save report to file
    import json
    report_filename = f"performance_report_{int(time.time())}.json"
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 Performance report saved: {report_filename}")
    
    return report
```

---

## 8.8 Error Handling

### 🚨 Centralized Error Management
```python
from enum import Enum
from typing import Optional, Dict, Any
import traceback

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SystemError:
    """ระบบจัดการข้อผิดพลาด"""
    
    def __init__(
        self,
        error_code: str,
        message: str,
        severity: ErrorSeverity,
        service_name: str,
        method_name: str = None,
        details: Dict[str, Any] = None
    ):
        self.error_code = error_code
        self.message = message
        self.severity = severity
        self.service_name = service_name
        self.method_name = method_name
        self.details = details or {}
        self.timestamp = time.time()
        self.traceback = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "service_name": self.service_name,
            "method_name": self.method_name,
            "details": self.details,
            "timestamp": self.timestamp,
            "traceback": self.traceback
        }

class ErrorHandler:
    """จัดการข้อผิดพลาดแบบรวมศูนย์"""
    
    def __init__(self):
        self.error_history: List[SystemError] = []
        self.error_counts: Dict[str, int] = {}
    
    def handle_error(
        self,
        error_code: str,
        message: str,
        severity: ErrorSeverity,
        service_name: str,
        method_name: str = None,
        details: Dict[str, Any] = None
    ) -> SystemError:
        """จัดการข้อผิดพลาด"""
        
        system_error = SystemError(
            error_code=error_code,
            message=message,
            severity=severity,
            service_name=service_name,
            method_name=method_name,
            details=details
        )
        
        self.error_history.append(system_error)
        self.error_counts[error_code] = self.error_counts.get(error_code, 0) + 1
        
        # Log error based on severity
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"🚨 CRITICAL ERROR [{error_code}]: {message}")
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"❌ ERROR [{error_code}]: {message}")
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(f"⚠️ WARNING [{error_code}]: {message}")
        else:
            logger.info(f"ℹ️ INFO [{error_code}]: {message}")
        
        # Take action based on severity
        if severity == ErrorSeverity.CRITICAL:
            asyncio.create_task(self._handle_critical_error(system_error))
        
        return system_error
    
    async def _handle_critical_error(self, error: SystemError):
        """จัดการข้อผิดพลาดร้ายแรง"""
        
        # Try to free up resources
        if "vram" in error.error_code.lower():
            await vram_manager.cleanup_unused_models()
            await vram_manager.force_garbage_collection()
        
        # Restart service if needed
        if error.service_name in ["face_detection", "face_recognition"]:
            logger.info(f"🔄 Attempting to restart {error.service_name} service")
            # Restart logic here
    
    def get_error_summary(self) -> Dict[str, Any]:
        """ดึงสรุปข้อผิดพลาด"""
        
        recent_errors = [
            error for error in self.error_history
            if time.time() - error.timestamp < 3600  # Last hour
        ]
        
        severity_counts = {}
        for error in recent_errors:
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "severity_counts": severity_counts,
            "top_error_codes": dict(
                sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            )
        }

# Global error handler
global_error_handler = ErrorHandler()

def handle_service_error(service_name: str, method_name: str = None):
    """Decorator สำหรับการจัดการข้อผิดพลาด"""
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_code = f"{service_name}.{method_name or func.__name__}.error"
                
                global_error_handler.handle_error(
                    error_code=error_code,
                    message=str(e),
                    severity=ErrorSeverity.HIGH,
                    service_name=service_name,
                    method_name=method_name or func.__name__,
                    details={"args": str(args), "kwargs": str(kwargs)}
                )
                
                raise
        
        return wrapper
    return decorator
```

---

## สรุป

ระบบโครงสร้างพื้นฐานและบริการสาธารณะให้การรองรับที่สำคัญสำหรับ AI Services ทั้งหมด:

### ✅ ฟีเจอร์หลัก
- **VRAM Management**: จัดการ GPU Memory อย่างมีประสิทธิภาพ
- **Service Architecture**: Base class และ dependency injection
- **Performance Monitoring**: ติดตามประสิทธิภาพแบบ real-time
- **Error Handling**: จัดการข้อผิดพลาดแบบรวมศูนย์
- **Statistics Collection**: เก็บและวิเคราะห์สถิติการใช้งาน

### 🎯 ประโยชน์
- ลดการใช้ Memory และเพิ่มประสิทธิภาพ
- การจัดการ Services แบบมาตรฐาน
- ติดตามปัญหาและแก้ไขได้รวดเร็ว
- รองรับการขยายระบบในอนาคต
- การดูแลรักษาที่ง่ายขึ้น

ระบบนี้เป็นรากฐานสำคัญที่ทำให้ AI Services ทั้งหมดทำงานได้อย่างเสถียรและมีประสิทธิภาพ
