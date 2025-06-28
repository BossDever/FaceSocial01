import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import io
import base64
from dataclasses import dataclass
from collections import deque
import threading
import logging

from .simple_yolov11m_service import SimpleYOLOv11mService

logger = logging.getLogger(__name__)

@dataclass
class OptimizedFrame:
    """Optimized frame data structure"""
    frame_id: str
    image_data: np.ndarray
    timestamp: float
    quality_factor: float = 1.0
    skip_detection: bool = False

@dataclass
class CacheEntry:
    """Cache entry for detection results"""
    result: Dict[str, Any]
    timestamp: float
    frame_hash: str

class OptimizedFaceDetectionService:
    """
    Advanced optimized face detection service with:
    - Frame batching
    - Intelligent caching
    - Adaptive quality control
    - Skip frame optimization
    - Memory management
    """
    
    def __init__(self):
        self.yolo_service = SimpleYOLOv11mService()
        self.frame_buffer = deque(maxlen=5)  # Buffer for batching
        self.result_cache = {}  # Cache for recent results
        self.cache_ttl = 0.5  # Cache time-to-live in seconds
        self.max_cache_size = 100
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.processing_times = deque(maxlen=20)
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # Adaptive parameters
        self.adaptive_quality = 0.6
        self.adaptive_skip_frames = 1
        self.min_quality = 0.3
        self.max_quality = 0.9
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info("OptimizedFaceDetectionService initialized")

    async def initialize(self):
        """Initialize the YOLO service"""
        await self.yolo_service.initialize()
        logger.info("OptimizedFaceDetectionService ready")

    def _calculate_frame_hash(self, image_data: np.ndarray) -> str:
        """Calculate a simple hash for frame comparison"""
        # Downsample for hash calculation
        small = cv2.resize(image_data, (32, 32))
        return str(hash(small.tobytes()))

    def _is_similar_frame(self, frame_hash: str) -> Optional[Dict[str, Any]]:
        """Check if we have a cached result for a similar frame"""
        current_time = time.time()
        
        for cached_hash, cache_entry in list(self.result_cache.items()):
            # Remove expired entries
            if current_time - cache_entry.timestamp > self.cache_ttl:
                del self.result_cache[cached_hash]
                continue
                
            # Check for exact match
            if cached_hash == frame_hash:
                return cache_entry.result
                
        return None

    def _update_cache(self, frame_hash: str, result: Dict[str, Any]):
        """Update cache with new result"""
        current_time = time.time()
        
        # Clean old entries if cache is full
        if len(self.result_cache) >= self.max_cache_size:
            oldest_key = min(self.result_cache.keys(), 
                           key=lambda k: self.result_cache[k].timestamp)
            del self.result_cache[oldest_key]
        
        self.result_cache[frame_hash] = CacheEntry(
            result=result,
            timestamp=current_time,
            frame_hash=frame_hash
        )

    def _adaptive_resize(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Adaptively resize image based on performance"""
        height, width = image.shape[:2]
        
        # Calculate target size based on adaptive quality
        target_width = int(width * self.adaptive_quality)
        target_height = int(height * self.adaptive_quality)
        
        # Ensure minimum size
        target_width = max(target_width, 320)
        target_height = max(target_height, 240)
        
        if target_width != width or target_height != height:
            resized = cv2.resize(image, (target_width, target_height), 
                               interpolation=cv2.INTER_LINEAR)
            scale_factor = width / target_width
            return resized, scale_factor
        
        return image, 1.0

    def _update_performance_metrics(self, processing_time: float):
        """Update performance metrics and adaptive parameters"""
        with self._lock:
            current_time = time.time()
            self.processing_times.append(processing_time)
            self.frame_count += 1
            
            # Calculate FPS
            time_diff = current_time - self.last_fps_time
            if time_diff >= 1.0:  # Update FPS every second
                fps = self.frame_count / time_diff
                self.fps_history.append(fps)
                self.frame_count = 0
                self.last_fps_time = current_time
                
                # Adaptive optimization based on FPS
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                avg_processing_time = sum(self.processing_times) / len(self.processing_times)
                
                # Adjust quality based on performance
                if avg_fps < 8:  # Low FPS
                    self.adaptive_quality = max(self.min_quality, self.adaptive_quality - 0.05)
                    self.adaptive_skip_frames = min(3, self.adaptive_skip_frames + 1)
                elif avg_fps > 15:  # High FPS
                    self.adaptive_quality = min(self.max_quality, self.adaptive_quality + 0.02)
                    self.adaptive_skip_frames = max(0, self.adaptive_skip_frames - 1)
                
                logger.debug(f"FPS: {avg_fps:.1f}, Quality: {self.adaptive_quality:.2f}, "
                           f"Skip: {self.adaptive_skip_frames}, Processing: {avg_processing_time:.3f}s")

    def should_skip_frame(self) -> bool:
        """Determine if current frame should be skipped"""
        if self.adaptive_skip_frames <= 0:
            return False
            
        return (self.frame_count % (self.adaptive_skip_frames + 1)) != 0

    async def detect_faces_optimized(
        self,
        image_data: bytes,
        conf_threshold: float = 0.3,
        max_faces: int = 15,
        min_quality_threshold: float = 20.0
    ) -> Dict[str, Any]:
        """
        Optimized face detection with caching and adaptive processing
        """
        start_time = time.time()
        
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {
                    "success": False,
                    "error": "Unable to decode image",
                    "faces": [],
                    "processing_time": time.time() - start_time
                }
            
            # Check if we should skip this frame
            if self.should_skip_frame():
                # Return last cached result if available
                if self.result_cache:
                    latest_result = max(self.result_cache.values(), 
                                      key=lambda x: x.timestamp).result
                    latest_result["skipped"] = True
                    latest_result["processing_time"] = time.time() - start_time
                    return latest_result
            
            # Calculate frame hash for caching
            frame_hash = self._calculate_frame_hash(image)
            
            # Check cache first
            cached_result = self._is_similar_frame(frame_hash)
            if cached_result:
                cached_result["cached"] = True
                cached_result["processing_time"] = time.time() - start_time
                return cached_result
            
            # Adaptive resize
            processed_image, scale_factor = self._adaptive_resize(image)
            
            # Convert to PIL Image for YOLO processing
            pil_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
            
            # Run detection
            faces = await self.yolo_service.detect_faces(
                image=pil_image,
                conf_threshold=conf_threshold,
                max_faces=max_faces,
                min_quality_threshold=min_quality_threshold
            )
            
            # Scale back coordinates if image was resized
            if scale_factor != 1.0:
                for face in faces:
                    if 'bbox' in face:
                        bbox = face['bbox']
                        bbox['x1'] = int(bbox['x1'] * scale_factor)
                        bbox['y1'] = int(bbox['y1'] * scale_factor)
                        bbox['x2'] = int(bbox['x2'] * scale_factor)
                        bbox['y2'] = int(bbox['y2'] * scale_factor)
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "faces": faces,
                "processing_time": processing_time,
                "adaptive_quality": self.adaptive_quality,
                "scale_factor": scale_factor,
                "cached": False,
                "skipped": False,
                "frame_hash": frame_hash
            }
            
            # Update cache
            self._update_cache(frame_hash, result.copy())
            
            # Update performance metrics
            self._update_performance_metrics(processing_time)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in optimized face detection: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "faces": [],
                "processing_time": processing_time
            }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        with self._lock:
            avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
            avg_processing_time = (sum(self.processing_times) / len(self.processing_times) 
                                 if self.processing_times else 0)
            
            return {
                "average_fps": round(avg_fps, 2),
                "average_processing_time": round(avg_processing_time * 1000, 2),  # ms
                "adaptive_quality": round(self.adaptive_quality, 2),
                "adaptive_skip_frames": self.adaptive_skip_frames,
                "cache_size": len(self.result_cache),
                "total_frames_processed": sum(len(self.fps_history) * fps for fps in self.fps_history)
            }

    def reset_performance_stats(self):
        """Reset performance statistics"""
        with self._lock:
            self.fps_history.clear()
            self.processing_times.clear()
            self.frame_count = 0
            self.last_fps_time = time.time()

    def clear_cache(self):
        """Clear the result cache"""
        with self._lock:
            self.result_cache.clear()

    async def cleanup(self):
        """Cleanup resources"""
        self.clear_cache()
        await self.yolo_service.cleanup()
        logger.info("OptimizedFaceDetectionService cleaned up")

# Global instance
optimized_service = OptimizedFaceDetectionService()
