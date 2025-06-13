"""
Common statistics and performance tracking utilities for AI services.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class ModelPerformanceStats:
    """Model performance statistics tracking."""
    
    # Basic counters
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    
    # Timing statistics
    total_processing_time: float = 0.0
    min_processing_time: float = float('inf')
    max_processing_time: float = 0.0
    average_processing_time: float = 0.0
    
    # Extraction specific stats
    total_extractions: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    total_extraction_time: float = 0.0
    average_extraction_time: float = 0.0
    
    # Recognition specific stats
    total_recognitions: int = 0
    successful_recognitions: int = 0
    failed_recognitions: int = 0
    total_recognition_time: float = 0.0
    average_recognition_time: float = 0.0
    
    # Quality metrics
    high_quality_results: int = 0
    medium_quality_results: int = 0
    low_quality_results: int = 0
    
    # Performance history
    recent_times: List[float] = field(default_factory=list)
    max_history_size: int = 100
    
    def update_extraction_stats(self, time_taken: float, success: bool = True) -> None:
        """Update extraction statistics."""
        self.total_extractions += 1
        self.total_extraction_time += time_taken
        
        if success:
            self.successful_extractions += 1
        else:
            self.failed_extractions += 1
            
        # Update average
        if self.total_extractions > 0:
            self.average_extraction_time = (
                self.total_extraction_time / self.total_extractions
            )
        
        # Update general stats
        self._update_general_stats(time_taken, success)
    
    def update_recognition_stats(self, time_taken: float, success: bool = True) -> None:
        """Update recognition statistics."""
        self.total_recognitions += 1
        self.total_recognition_time += time_taken
        
        if success:
            self.successful_recognitions += 1
        else:
            self.failed_recognitions += 1
            
        # Update average
        if self.total_recognitions > 0:
            self.average_recognition_time = (
                self.total_recognition_time / self.total_recognitions
            )
        
        # Update general stats
        self._update_general_stats(time_taken, success)
    
    def update_quality_stats(self, quality_level: str) -> None:
        """Update quality statistics."""
        quality_level = quality_level.lower()
        if quality_level == 'high':
            self.high_quality_results += 1
        elif quality_level == 'medium':
            self.medium_quality_results += 1
        elif quality_level == 'low':
            self.low_quality_results += 1
    
    def _update_general_stats(self, time_taken: float, success: bool) -> None:
        """Update general performance statistics."""
        self.total_operations += 1
        self.total_processing_time += time_taken
        
        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1
        
        # Update timing stats
        self.min_processing_time = min(self.min_processing_time, time_taken)
        self.max_processing_time = max(self.max_processing_time, time_taken)
        
        if self.total_operations > 0:
            self.average_processing_time = (
                self.total_processing_time / self.total_operations
            )
        
        # Update recent times history
        self.recent_times.append(time_taken)
        if len(self.recent_times) > self.max_history_size:
            self.recent_times = self.recent_times[-self.max_history_size:]
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_operations == 0:
            return 0.0
        return self.successful_operations / self.total_operations
    
    @property
    def extraction_success_rate(self) -> float:
        """Calculate extraction success rate."""
        if self.total_extractions == 0:
            return 0.0
        return self.successful_extractions / self.total_extractions
    
    @property
    def recognition_success_rate(self) -> float:
        """Calculate recognition success rate."""
        if self.total_recognitions == 0:
            return 0.0
        return self.successful_recognitions / self.total_recognitions
    
    @property
    def operations_per_second(self) -> float:
        """Calculate operations per second based on recent performance."""
        if not self.recent_times or self.average_processing_time == 0:
            return 0.0
        return 1.0 / self.average_processing_time
    
    def get_recent_average_time(self, samples: int = 10) -> float:
        """Get average time for recent operations."""
        if not self.recent_times:
            return 0.0
        
        recent_samples = self.recent_times[-samples:]
        return sum(recent_samples) / len(recent_samples)
    
    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.total_processing_time = 0.0
        self.min_processing_time = float('inf')
        self.max_processing_time = 0.0
        self.average_processing_time = 0.0
        
        self.total_extractions = 0
        self.successful_extractions = 0
        self.failed_extractions = 0
        self.total_extraction_time = 0.0
        self.average_extraction_time = 0.0
        
        self.total_recognitions = 0
        self.successful_recognitions = 0
        self.failed_recognitions = 0
        self.total_recognition_time = 0.0
        self.average_recognition_time = 0.0
        
        self.high_quality_results = 0
        self.medium_quality_results = 0
        self.low_quality_results = 0
        
        self.recent_times.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "success_rate": self.success_rate,
            
            "timing": {
                "total_processing_time": self.total_processing_time,
                "average_processing_time": self.average_processing_time,
                "min_processing_time": self.min_processing_time if self.min_processing_time != float('inf') else 0.0,
                "max_processing_time": self.max_processing_time,
                "operations_per_second": self.operations_per_second,
            },
            
            "extraction": {
                "total_extractions": self.total_extractions,
                "successful_extractions": self.successful_extractions,
                "failed_extractions": self.failed_extractions,
                "extraction_success_rate": self.extraction_success_rate,
                "average_extraction_time": self.average_extraction_time,
            },
            
            "recognition": {
                "total_recognitions": self.total_recognitions,
                "successful_recognitions": self.successful_recognitions,
                "failed_recognitions": self.failed_recognitions,
                "recognition_success_rate": self.recognition_success_rate,
                "average_recognition_time": self.average_recognition_time,
            },
            
            "quality": {
                "high_quality_results": self.high_quality_results,
                "medium_quality_results": self.medium_quality_results,
                "low_quality_results": self.low_quality_results,
                "total_quality_assessments": (
                    self.high_quality_results + 
                    self.medium_quality_results + 
                    self.low_quality_results
                ),
            },
        }


@dataclass
class ServiceStats:
    """Service-level statistics tracking."""
    
    service_name: str
    start_time: float = field(default_factory=time.time)
    model_stats: Dict[str, ModelPerformanceStats] = field(default_factory=dict)
    
    def get_model_stats(self, model_name: str) -> ModelPerformanceStats:
        """Get or create model statistics."""
        if model_name not in self.model_stats:
            self.model_stats[model_name] = ModelPerformanceStats()
        return self.model_stats[model_name]
    
    def get_uptime(self) -> float:
        """Get service uptime in seconds."""
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert service statistics to dictionary."""
        return {
            "service_name": self.service_name,
            "uptime_seconds": self.get_uptime(),
            "start_time": self.start_time,
            "models": {
                model_name: stats.to_dict()
                for model_name, stats in self.model_stats.items()
            },
        }


# Global stats registry
_stats_registry: Dict[str, ServiceStats] = {}


def get_service_stats(service_name: str) -> ServiceStats:
    """Get or create service statistics."""
    if service_name not in _stats_registry:
        _stats_registry[service_name] = ServiceStats(service_name)
    return _stats_registry[service_name]


def get_all_stats() -> Dict[str, Dict[str, Any]]:
    """Get all service statistics."""
    return {
        service_name: stats.to_dict()
        for service_name, stats in _stats_registry.items()
    }


def reset_all_stats() -> None:
    """Reset all service statistics."""
    for stats in _stats_registry.values():
        for model_stats in stats.model_stats.values():
            model_stats.reset_stats()


def cleanup_stats() -> None:
    """Clean up statistics registry."""
    _stats_registry.clear()