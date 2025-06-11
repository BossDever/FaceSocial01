# cSpell:disable
"""
VRAM Manager สำหรับจัดการหน่วยความจำ GPU ในระบบ AI
Enhanced version with better error handling and logging
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List
import torch

logger = logging.getLogger(__name__)

class AllocationPriority(Enum):
    CRITICAL = "critical"  # ต้องอยู่บน GPU เสมอ
    HIGH = "high"          # ควรอยู่บน GPU
    MEDIUM = "medium"      # เป็นตัวเลือก
    LOW = "low"            # GPU ถ้าว่าง

class AllocationLocation(Enum):
    GPU = "gpu"
    CPU = "cpu"

@dataclass
class ModelAllocation:
    model_id: str
    priority: AllocationPriority
    service_id: str
    location: AllocationLocation
    vram_allocated: int
    status: str
    timestamp: float = 0.0

class VRAMManager:
    """
    ระบบบริหารจัดการหน่วยความจำ GPU สำหรับโมเดล AI
    Enhanced with better monitoring and error handling
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_allocations: Dict[str, ModelAllocation] = {}
        self.total_vram = self._get_total_vram()
        self.allocated_vram = 0
        self.lock = asyncio.Lock()
        
        # Statistics
        self.allocation_history: List[Dict[str, Any]] = []
        self.max_history_size = config.get("max_history_size", 1000)
        
        logger.info(f"VRAM Manager initialized with {self.total_vram/1024/1024:.1f}MB total VRAM")
        
    def _get_total_vram(self) -> int:
        """ตรวจสอบขนาด VRAM ทั้งหมดที่มี"""
        try:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                properties = torch.cuda.get_device_properties(device)
                total_memory = properties.total_memory
                
                # Reserve some memory for system
                reserved_mb = self.config.get("reserved_vram_mb", 512)
                reserved_bytes = reserved_mb * 1024 * 1024
                
                usable_memory = max(0, total_memory - reserved_bytes)
                
                logger.info(f"GPU: {properties.name}")
                logger.info(f"Total VRAM: {total_memory/1024/1024:.1f}MB")
                logger.info(f"Reserved: {reserved_mb}MB")
                logger.info(f"Usable: {usable_memory/1024/1024:.1f}MB")
                
                return usable_memory
            else:
                logger.warning("CUDA not available, using CPU-only mode")
                return 0
        except Exception as e:
            logger.error(f"Error getting VRAM info: {e}")
            return 0
    
    async def request_model_allocation(
        self,
        model_id: str,
        priority: str,
        service_id: str,
        vram_required: Optional[int] = None
    ) -> ModelAllocation:
        """
        ขอจัดสรร VRAM สำหรับโมเดล
        Enhanced with better error handling and logging
        """
        async with self.lock:
            try:
                # ตรวจสอบว่ามี GPU หรือไม่
                if self.total_vram == 0:
                    logger.warning(f"No GPU available for model {model_id}, using CPU")
                    allocation = ModelAllocation(
                        model_id=model_id,
                        priority=AllocationPriority(priority),
                        service_id=service_id,
                        location=AllocationLocation.CPU,
                        vram_allocated=0,
                        status="fallback_to_cpu_no_gpu",
                        timestamp=time.time()
                    )
                    self.model_allocations[model_id] = allocation
                    self._log_allocation_event("fallback_cpu", allocation)
                    return allocation
                    
                # ถ้าโมเดลถูกโหลดอยู่แล้ว
                if model_id in self.model_allocations:
                    allocation = self.model_allocations[model_id]
                    logger.info(f"Model {model_id} already allocated at {allocation.location.value}")
                    return allocation
                
                # คำนวณขนาด VRAM ที่ต้องการ
                if vram_required is None:
                    estimates = self.config.get("model_vram_estimates", {})
                    vram_required = estimates.get(model_id, 512 * 1024 * 1024)  # 512MB default
                
                # ตรวจสอบว่ามี VRAM พอหรือไม่
                available_vram = self.total_vram - self.allocated_vram
                
                # จัดสรร VRAM
                if available_vram >= vram_required or priority == AllocationPriority.CRITICAL.value:
                    # สำหรับ CRITICAL ถ้า VRAM ไม่พอ จะต้องย้ายโมเดลอื่นออก
                    if available_vram < vram_required and priority == AllocationPriority.CRITICAL.value:
                        freed = self._free_vram_for_critical_model(vram_required - available_vram)
                        if freed < vram_required - available_vram:
                            logger.error(f"Cannot free enough VRAM for critical model {model_id}")
                    
                    # ตรวจสอบ VRAM อีกครั้งหลังจากปล่อย
                    available_vram = self.total_vram - self.allocated_vram
                    if available_vram >= vram_required:
                        self.allocated_vram += vram_required
                        allocation = ModelAllocation(
                            model_id=model_id,
                            priority=AllocationPriority(priority),
                            service_id=service_id,
                            location=AllocationLocation.GPU,
                            vram_allocated=vram_required,
                            status="allocated_on_gpu",
                            timestamp=time.time()
                        )
                        self.model_allocations[model_id] = allocation
                        logger.info(f"Allocated {vram_required/1024/1024:.1f}MB VRAM for {model_id}")
                        self._log_allocation_event("allocated", allocation)
                        return allocation
                
                # ถ้า VRAM ไม่พอ ใช้ CPU แทน
                logger.warning(f"Insufficient VRAM for {model_id} ({vram_required/1024/1024:.1f}MB > {available_vram/1024/1024:.1f}MB)")
                allocation = ModelAllocation(
                    model_id=model_id,
                    priority=AllocationPriority(priority),
                    service_id=service_id,
                    location=AllocationLocation.CPU,
                    vram_allocated=0,
                    status="fallback_to_cpu_insufficient_vram",
                    timestamp=time.time()
                )
                self.model_allocations[model_id] = allocation
                self._log_allocation_event("fallback_cpu", allocation)
                return allocation
                
            except Exception as e:
                logger.error(f"Error in model allocation for {model_id}: {e}")
                # Return CPU allocation as fallback
                allocation = ModelAllocation(
                    model_id=model_id,
                    priority=AllocationPriority(priority),
                    service_id=service_id,
                    location=AllocationLocation.CPU,
                    vram_allocated=0,
                    status=f"error_fallback_cpu: {str(e)}",
                    timestamp=time.time()
                )
                self.model_allocations[model_id] = allocation
                return allocation
    
    def _free_vram_for_critical_model(self, vram_needed: int) -> int:
        """
        ย้ายโมเดลที่มีความสำคัญน้อยกว่าออกจาก GPU เพื่อให้มี VRAM พอสำหรับโมเดลที่สำคัญกว่า
        Returns: Amount of VRAM freed
        """
        # เรียงลำดับโมเดลตามความสำคัญจากน้อยไปมาก
        candidates = sorted(
            [a for a in self.model_allocations.values() if a.location == AllocationLocation.GPU],
            key=lambda x: self._priority_weight(x.priority)
        )
        
        vram_freed = 0
        for allocation in candidates:
            # ข้ามโมเดลที่มีความสำคัญสูงสุด
            if allocation.priority == AllocationPriority.CRITICAL:
                continue
                
            # ย้ายโมเดลออกจาก GPU
            vram_freed += allocation.vram_allocated
            logger.info(f"Moving model {allocation.model_id} to CPU to free {allocation.vram_allocated/1024/1024:.1f}MB")
            
            # อัปเดตสถานะ
            allocation.location = AllocationLocation.CPU
            allocation.status = "moved_to_cpu_for_critical"
            self.allocated_vram -= allocation.vram_allocated
            allocation.vram_allocated = 0
            
            self._log_allocation_event("moved_to_cpu", allocation)
            
            # ตรวจสอบว่าได้ VRAM พอแล้วหรือยัง
            if vram_freed >= vram_needed:
                break
                
        return vram_freed
    
    def _priority_weight(self, priority: AllocationPriority) -> int:
        """Convert priority to numeric weight for sorting"""
        weights = {
            AllocationPriority.LOW: 1,
            AllocationPriority.MEDIUM: 2,
            AllocationPriority.HIGH: 3,
            AllocationPriority.CRITICAL: 4
        }
        return weights.get(priority, 1)
    
    def _log_allocation_event(self, event_type: str, allocation: ModelAllocation):
        """Log allocation events for monitoring"""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "model_id": allocation.model_id,
            "service_id": allocation.service_id,
            "location": allocation.location.value,
            "vram_allocated": allocation.vram_allocated,
            "status": allocation.status
        }
        
        self.allocation_history.append(event)
        
        # Keep history size manageable
        if len(self.allocation_history) > self.max_history_size:
            self.allocation_history = self.allocation_history[-self.max_history_size//2:]
    
    async def release_model_allocation(self, model_id: str) -> bool:
        """
        คืน VRAM ที่ใช้โดยโมเดล
        Enhanced with better error handling
        """
        async with self.lock:
            try:
                if model_id in self.model_allocations:
                    allocation = self.model_allocations[model_id]
                    
                    # ลด VRAM ที่ใช้อยู่
                    if allocation.location == AllocationLocation.GPU:
                        self.allocated_vram -= allocation.vram_allocated
                        logger.info(f"Released {allocation.vram_allocated/1024/1024:.1f}MB VRAM from {model_id}")
                    
                    # Log event
                    self._log_allocation_event("released", allocation)
                    
                    # ลบการจัดสรร
                    del self.model_allocations[model_id]
                    return True
                else:
                    logger.warning(f"Model {model_id} not found in allocations")
                    return False
                    
            except Exception as e:
                logger.error(f"Error releasing allocation for {model_id}: {e}")
                return False
    
    async def get_vram_status(self) -> Dict[str, Any]:
        """
        ดูสถานะการใช้ VRAM ปัจจุบัน
        Enhanced with more detailed information
        """
        async with self.lock:
            try:
                # Get current GPU memory usage if available
                gpu_memory_used = 0
                gpu_memory_cached = 0
                
                if torch.cuda.is_available():
                    gpu_memory_used = torch.cuda.memory_allocated(0)
                    gpu_memory_cached = torch.cuda.memory_reserved(0)
                
                status = {
                    "total_vram": self.total_vram,
                    "allocated_vram": self.allocated_vram,
                    "available_vram": self.total_vram - self.allocated_vram,
                    "usage_percentage": (self.allocated_vram / self.total_vram * 100) if self.total_vram > 0 else 0,
                    "gpu_memory_used": gpu_memory_used,
                    "gpu_memory_cached": gpu_memory_cached,
                    "model_count": len(self.model_allocations),
                    "gpu_models": len([a for a in self.model_allocations.values() if a.location == AllocationLocation.GPU]),
                    "cpu_models": len([a for a in self.model_allocations.values() if a.location == AllocationLocation.CPU]),
                    "model_allocations": {
                        model_id: {
                            "service": allocation.service_id,
                            "priority": allocation.priority.value,
                            "location": allocation.location.value,
                            "vram_mb": allocation.vram_allocated / (1024 * 1024),
                            "status": allocation.status,
                            "timestamp": allocation.timestamp
                        }
                        for model_id, allocation in self.model_allocations.items()
                    }
                }
                
                return status
                
            except Exception as e:
                logger.error(f"Error getting VRAM status: {e}")
                return {
                    "error": str(e),
                    "total_vram": self.total_vram,
                    "allocated_vram": self.allocated_vram
                }
    
    async def get_allocation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent allocation history"""
        return self.allocation_history[-limit:] if self.allocation_history else []
    
    async def cleanup_unused_allocations(self) -> int:
        """Clean up allocations that might be stuck"""
        async with self.lock:
            cleaned = 0
            current_time = time.time()
            stale_threshold = 3600  # 1 hour
            
            to_remove = []
            for model_id, allocation in self.model_allocations.items():
                if current_time - allocation.timestamp > stale_threshold:
                    if allocation.location == AllocationLocation.GPU:
                        self.allocated_vram -= allocation.vram_allocated
                    to_remove.append(model_id)
                    cleaned += 1
            
            for model_id in to_remove:
                del self.model_allocations[model_id]
                logger.info(f"Cleaned up stale allocation for {model_id}")
            
            return cleaned