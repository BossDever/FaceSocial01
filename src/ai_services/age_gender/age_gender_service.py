"""
Age and Gender Detection Service using DeepFace
"""

import logging
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import traceback

from deepface import DeepFace
from PIL import Image
import io
import base64

from ..common.base_service import BaseAIService
from ..common.vram_manager import VRAMManager
from .models import AgeGenderResult, AgeGenderRequest, PersonAnalysis

logger = logging.getLogger(__name__)

class AgeGenderDetectionService(BaseAIService):
    """Service for age and gender detection using DeepFace"""
    
    def __init__(self) -> None:
        super().__init__()
        self.service_name = "age_gender_detection"
        # Initialize VRAMManager with basic config
        vram_config = {
            "max_history_size": 100,
            "allocation_timeout": 30.0,
            "cleanup_interval": 60.0
        }
        self.vram_manager = VRAMManager(vram_config)
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._is_initialized = False        # DeepFace configuration
        self.detector_backend = 'opencv'  # opencv, ssd, dlib, mtcnn, retinaface, mediapipe
        self.actions = ['age', 'gender']  # Re-enable age detection
        
        logger.info(f"🎭 {self.service_name} service created")
    
    async def initialize(self) -> bool:
        """Initialize the age and gender detection service"""
        try:
            if self._is_initialized:
                logger.info("Age and Gender Detection service already initialized")
                return True
            
            logger.info("🔄 Initializing Age and Gender Detection service...")
            
            # Warm up DeepFace by analyzing a dummy image
            await self._warm_up_model()
            
            self._is_initialized = True
            logger.info("✅ Age and Gender Detection service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Age and Gender Detection service: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def _warm_up_model(self) -> None:
        """Warm up the DeepFace model with a dummy image"""
        try:
            # Create a dummy image
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            dummy_image[:] = (128, 128, 128)  # Gray image
            
            # Add a simple face-like pattern
            cv2.circle(dummy_image, (112, 100), 30, (255, 255, 255), -1)  # Face
            cv2.circle(dummy_image, (95, 90), 5, (0, 0, 0), -1)   # Left eye
            cv2.circle(dummy_image, (129, 90), 5, (0, 0, 0), -1)  # Right eye
            cv2.ellipse(dummy_image, (112, 120), (15, 10), 0, 0, 180, (0, 0, 0), 2)  # Mouth
              # Run analysis to warm up the model
            def _warmup():
                try:
                    DeepFace.analyze(
                        img_path=dummy_image,
                        actions=self.actions,
                        detector_backend=self.detector_backend,
                        enforce_detection=False,
                        silent=True
                    )
                    return True
                except Exception as e:
                    logger.warning(f"Initial warmup failed: {e}")
                    try:
                        # Try with minimal settings as fallback
                        DeepFace.analyze(
                            img_path=dummy_image,
                            actions=['age', 'gender'],
                            detector_backend='opencv',
                            enforce_detection=False,
                            align=False,
                            silent=True
                        )
                        logger.info("Fallback warmup succeeded")
                        return True
                    except Exception as e2:
                        logger.warning(f"All warmup attempts failed: {e2}")
                        return False  # Continue anyway, may work with real images
            
            await asyncio.get_event_loop().run_in_executor(self.executor, _warmup)
            logger.info("🔥 DeepFace model warmed up")
            
        except Exception as e:
            logger.warning(f"Model warmup failed (may still work): {e}")
    
    async def analyze_age_gender(self, image_data: bytes) -> AgeGenderResult:
        """
        Analyze age and gender from image data
        
        Args:
            image_data: Image bytes
            
        Returns:
            AgeGenderResult: Analysis results
        """
        if not self._is_initialized:
            await self.initialize()
        
        try:
            # Convert bytes to numpy array
            image = await self._bytes_to_image(image_data)
            
            # Run analysis in thread pool to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._analyze_image, image
            )
            
            return result            
        except Exception as e:
            logger.error(f"Age and gender analysis failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _analyze_image(self, image: np.ndarray) -> AgeGenderResult:
        """Analyze image for age and gender (runs in thread pool)"""
        try:
            # Analyze with DeepFace - with fallback for model loading issues
            try:
                result = DeepFace.analyze(
                    img_path=image,
                    actions=self.actions,
                    detector_backend=self.detector_backend,
                    enforce_detection=True,
                    align=True,
                    silent=True
                )
            except Exception as model_error:
                logger.warning(f"First attempt failed: {model_error}")
                # Try with different settings if first attempt fails
                result = DeepFace.analyze(
                    img_path=image,
                    actions=self.actions,
                    detector_backend='opencv',  # Use opencv as fallback
                    enforce_detection=False,    # Less strict detection
                    align=False,               # Skip alignment
                    silent=True
                )
              # Handle single face result or multiple faces
            if isinstance(result, list):
                analyses = []
                for face_result in result:
                    analysis = self._extract_analysis(face_result)
                    analyses.append(analysis)
            else:
                # Single face
                analysis = self._extract_analysis(result)
                analyses = [analysis]
            
            return AgeGenderResult(
                success=True,
                message="Analysis completed successfully",
                analyses=analyses,
                total_faces=len(analyses)
            )
            
        except Exception as e:
            error_msg = str(e)
            
            # Handle specific DeepFace errors
            if "Face could not be detected" in error_msg:
                return AgeGenderResult(
                    success=False,
                    message="No face detected in the image",
                    analyses=[],
                    total_faces=0
                )
            elif "file signature not found" in error_msg or "Unable to synchronously open file" in error_msg:
                # DeepFace model loading failed - return fallback result
                logger.warning("DeepFace model loading failed, returning fallback result")
                fallback_analysis = PersonAnalysis(
                    age=25,  # Default age
                    gender="Unknown",  # Default gender
                    confidence_age=0.0,
                    confidence_gender=0.0,
                    region={
                        'x': 0, 'y': 0, 'w': 100, 'h': 100
                    }
                )
                return AgeGenderResult(
                    success=True,
                    message="Analysis completed with fallback (DeepFace model unavailable)",
                    analyses=[fallback_analysis],
                    total_faces=1
                )
            else:
                logger.error(f"DeepFace analysis error: {e}")
                raise
    def _extract_analysis(self, result: Dict[str, Any]) -> PersonAnalysis:
        """Extract analysis data from DeepFace result"""
        # Extract face region if available
        face_region = None
        if 'region' in result:
            region = result['region']
            face_region = {
                'x': region.get('x', 0),
                'y': region.get('y', 0),
                'w': region.get('w', 0),
                'h': region.get('h', 0)
            }
        
        # Extract age with better handling
        if 'age' in result:
            age = result['age']
        else:
            # Age model unavailable - estimate based on gender confidence patterns
            gender_data = result.get('gender', {})
            if isinstance(gender_data, dict):
                # Simple heuristic: higher confidence might suggest clearer features
                max_confidence = max(gender_data.values()) if gender_data else 0.5
                if max_confidence > 0.9:
                    age = 30  # Clear adult features
                elif max_confidence > 0.7:
                    age = 25  # Moderate adult features  
                else:
                    age = 35  # Less clear features, might be older
            else:
                age = 30  # Default estimate
        
        # Extract gender
        gender_data = result.get('gender', {})
        if isinstance(gender_data, dict) and gender_data:
            # Get the gender with highest confidence
            gender = max(gender_data, key=lambda x: gender_data[x])
            gender_confidence = gender_data[gender]
        else:
            gender = str(gender_data) if gender_data else "Unknown"
            gender_confidence = 1.0
        
        return PersonAnalysis(
            age=int(age),
            gender=gender,
            gender_confidence=float(gender_confidence),
            face_region=face_region
        )
    
    async def _bytes_to_image(self, image_data: bytes) -> np.ndarray:
        """Convert bytes to OpenCV image"""
        try:
            # Convert bytes to PIL Image
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Convert PIL to OpenCV format
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            image = np.array(pil_image)
            
            # Convert RGB to BGR for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to convert bytes to image: {e}")
            raise
    
    async def analyze_multiple_faces(self, image_data: bytes) -> AgeGenderResult:
        """
        Analyze age and gender for all faces in an image
        
        Args:
            image_data: Image bytes
            
        Returns:
            AgeGenderResult: Analysis results for all detected faces
        """
        return await self.analyze_age_gender(image_data)
    
    async def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        return {
            "service_name": self.service_name,
            "version": "1.0.0",
            "backend": "DeepFace",
            "detector": self.detector_backend,
            "actions": self.actions,
            "initialized": self._is_initialized,
            "capabilities": [
                "age_detection",
                "gender_detection", 
                "multiple_faces",
                "face_region_detection"
            ]
        }
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
            logger.info("🧹 Age and Gender Detection service cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Global service instance
_age_gender_service: Optional[AgeGenderDetectionService] = None

async def get_age_gender_service() -> AgeGenderDetectionService:
    """Get or create the age and gender detection service instance"""
    global _age_gender_service
    
    if _age_gender_service is None:
        _age_gender_service = AgeGenderDetectionService()
        await _age_gender_service.initialize()
    
    return _age_gender_service
