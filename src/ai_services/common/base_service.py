"""
Base Service Class for AI Services
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class BaseAIService(ABC):
    """Base class for all AI services"""
    
    def __init__(self) -> None:
        self.service_name = "base_service"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the service"""
        pass
    
    async def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        return {
            "service_name": self.service_name,
            "class_name": self.__class__.__name__,
            "initialized": getattr(self, '_is_initialized', False)
        }
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        pass
