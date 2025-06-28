# ชุดคำสั่ง: การวิเคราะห์อายุและเพศ
## Age & Gender Detection ด้วย DeepFace

### 📋 สารบัญ
5.1 [ภาพรวม Age & Gender Detection](#51-ภาพรวม-age--gender-detection)
5.2 [DeepFace Age Gender Service](#52-deepface-age-gender-service)
5.3 [API Implementation](#53-api-implementation)
5.4 [Frontend Integration](#54-frontend-integration)
5.5 [Batch Processing](#55-batch-processing)
5.6 [Performance Optimization](#56-performance-optimization)
5.7 [Error Handling](#57-error-handling)
5.8 [Testing และ Validation](#58-testing-และ-validation)

---

## 5.1 ภาพรวม Age & Gender Detection

ระบบวิเคราะห์อายุและเพศจากใบหน้าด้วย DeepFace เพื่อเพิ่มข้อมูลประชากรศาสตร์ให้กับแพลตฟอร์ม

### 📊 ความสามารถการวิเคราะห์
- **Age Detection**: ประมาณอายุจากใบหน้าด้วยความแม่นยำสูง
- **Gender Recognition**: จำแนกเพศจากใบหน้า  
- **Batch Processing**: ประมวลผลหลายภาพพร้อมกัน
- **Real-time Analysis**: วิเคราะห์แบบเรียลไทม์

---

## 5.2 DeepFace Age Gender Service

### 5.2.1 การตั้งค่าและเริ่มต้น Age Gender Service

```python
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

logger = logging.getLogger(__name__)

class AgeGenderDetectionService:
    """Service for age and gender detection using DeepFace"""
    
    def __init__(self) -> None:
        self.service_name = "age_gender_detection"
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._is_initialized = False
        
        # DeepFace configuration
        self.detector_backend = 'opencv'  # opencv, ssd, dlib, mtcnn, retinaface, mediapipe
        self.actions = ['age', 'gender']
        
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
                        return True
                    except Exception as e2:
                        logger.error(f"Fallback warmup also failed: {e2}")
                        return False
            
            # Run warmup in thread
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(self.executor, _warmup)
            
            if success:
                logger.info("🔥 DeepFace age/gender models warmed up successfully")
            else:
                logger.warning("⚠️ DeepFace warmup failed, but continuing...")
                
        except Exception as e:
            logger.warning(f"⚠️ Model warmup failed: {e}")

    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized"""
        return self._is_initialized
```

### 5.2.2 ชุดคำสั่งการวิเคราะห์อายุและเพศ

```python
    async def analyze_age_gender(self, image_data: bytes) -> Dict[str, Any]:
        """
        Analyze age and gender from image data
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            if not self._is_initialized:
                raise RuntimeError("Service not initialized")
            
            start_time = asyncio.get_event_loop().time()
            
            # Convert bytes to numpy array
            image_array = self._bytes_to_numpy(image_data)
            
            # Run analysis in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, 
                self._analyze_sync, 
                image_array
            )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            result['processing_time'] = processing_time
            
            return result
            
        except Exception as e:
            logger.error(f"Age and gender analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'faces_detected': 0,
                'people': [],
                'processing_time': 0.0
            }
    
    def _bytes_to_numpy(self, image_data: bytes) -> np.ndarray:
        """Convert image bytes to numpy array"""
        try:
            # Open image with PIL
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Failed to convert bytes to numpy: {e}")
            raise ValueError(f"Invalid image data: {e}")
    
    def _analyze_sync(self, image: np.ndarray) -> Dict[str, Any]:
        """Synchronous analysis method for thread pool"""
        try:
            # Run DeepFace analysis
            analysis_result = DeepFace.analyze(
                img_path=image,
                actions=self.actions,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                silent=True
            )
            
            # Process results
            people = []
            
            # DeepFace returns a list if multiple faces, single dict if one face
            if isinstance(analysis_result, list):
                face_results = analysis_result
            else:
                face_results = [analysis_result]
            
            for i, face_result in enumerate(face_results):
                try:
                    # Extract age and gender information
                    age = face_result.get('age', 0)
                    gender_info = face_result.get('gender', {})
                    
                    # Get dominant gender and confidence
                    if isinstance(gender_info, dict):
                        gender = max(gender_info, key=gender_info.get)
                        gender_confidence = gender_info[gender] / 100.0  # Convert to 0-1
                    else:
                        gender = 'Unknown'
                        gender_confidence = 0.0
                    
                    # Extract face region if available
                    region = face_result.get('region', {})
                    face_region = {
                        'x': region.get('x', 0),
                        'y': region.get('y', 0),
                        'width': region.get('w', 0),
                        'height': region.get('h', 0)
                    }
                    
                    # Create person analysis
                    person = {
                        'person_id': i + 1,
                        'age': int(age),
                        'age_range': self._get_age_range(age),
                        'gender': gender.lower(),
                        'gender_confidence': float(gender_confidence),
                        'face_region': face_region,
                        'dominant_emotion': None,  # Can be added if emotion analysis is enabled
                        'race': None  # Can be added if race analysis is enabled
                    }
                    
                    people.append(person)
                    
                except Exception as e:
                    logger.warning(f"Failed to process face {i}: {e}")
                    continue
            
            return {
                'success': True,
                'faces_detected': len(people),
                'people': people,
                'detector_backend': self.detector_backend,
                'actions_performed': self.actions
            }
            
        except Exception as e:
            logger.error(f"DeepFace analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'faces_detected': 0,
                'people': [],
                'detector_backend': self.detector_backend
            }
    
    def _get_age_range(self, age: float) -> str:
        """Convert age to age range category"""
        age = int(age)
        
        if age < 13:
            return 'child'
        elif age < 20:
            return 'teenager'
        elif age < 30:
            return 'young_adult'
        elif age < 45:
            return 'adult'
        elif age < 60:
            return 'middle_aged'
        else:
            return 'senior'

    async def analyze_age_gender_base64(self, base64_string: str) -> Dict[str, Any]:
        """Analyze age and gender from base64 encoded image"""
        try:
            # Remove data URL prefix if present
            if base64_string.startswith('data:'):
                base64_string = base64_string.split(',')[1]
            
            # Decode base64 to bytes
            image_data = base64.b64decode(base64_string)
            
            # Analyze
            return await self.analyze_age_gender(image_data)
            
        except Exception as e:
            logger.error(f"Base64 age/gender analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'faces_detected': 0,
                'people': [],
                'processing_time': 0.0
            }

    async def batch_analyze(self, images: List[bytes]) -> Dict[str, Any]:
        """Analyze multiple images for age and gender"""
        try:
            if len(images) > 10:  # Limit batch size
                raise ValueError("Maximum 10 images per batch")
            
            start_time = asyncio.get_event_loop().time()
            results = []
            total_faces = 0
            
            for i, image_data in enumerate(images):
                try:
                    result = await self.analyze_age_gender(image_data)
                    result['image_index'] = i
                    results.append(result)
                    
                    if result['success']:
                        total_faces += result['faces_detected']
                        
                except Exception as e:
                    results.append({
                        'image_index': i,
                        'success': False,
                        'error': str(e),
                        'faces_detected': 0,
                        'people': []
                    })
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Calculate statistics
            age_stats = self._calculate_age_statistics(results)
            gender_stats = self._calculate_gender_statistics(results)
            
            return {
                'success': True,
                'total_images': len(images),
                'total_faces': total_faces,
                'processing_time': processing_time,
                'results': results,
                'statistics': {
                    'age': age_stats,
                    'gender': gender_stats
                }
            }
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_images': len(images),
                'total_faces': 0,
                'results': []
            }

    def _calculate_age_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate age statistics from analysis results"""
        ages = []
        age_ranges = {'child': 0, 'teenager': 0, 'young_adult': 0, 'adult': 0, 'middle_aged': 0, 'senior': 0}
        
        for result in results:
            if result.get('success', False):
                for person in result.get('people', []):
                    age = person.get('age', 0)
                    age_range = person.get('age_range', 'unknown')
                    
                    ages.append(age)
                    if age_range in age_ranges:
                        age_ranges[age_range] += 1
        
        if ages:
            return {
                'average_age': sum(ages) / len(ages),
                'min_age': min(ages),
                'max_age': max(ages),
                'age_ranges': age_ranges,
                'total_people': len(ages)
            }
        else:
            return {
                'average_age': 0,
                'min_age': 0,
                'max_age': 0,
                'age_ranges': age_ranges,
                'total_people': 0
            }

    def _calculate_gender_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate gender statistics from analysis results"""
        gender_counts = {'male': 0, 'female': 0, 'unknown': 0}
        total_confidence = {'male': 0.0, 'female': 0.0}
        
        for result in results:
            if result.get('success', False):
                for person in result.get('people', []):
                    gender = person.get('gender', 'unknown').lower()
                    confidence = person.get('gender_confidence', 0.0)
                    
                    if gender in gender_counts:
                        gender_counts[gender] += 1
                        if gender in total_confidence:
                            total_confidence[gender] += confidence
                    else:
                        gender_counts['unknown'] += 1
        
        # Calculate average confidence
        avg_confidence = {}
        for gender in ['male', 'female']:
            if gender_counts[gender] > 0:
                avg_confidence[gender] = total_confidence[gender] / gender_counts[gender]
            else:
                avg_confidence[gender] = 0.0
        
        total_people = sum(gender_counts.values())
        
        return {
            'gender_counts': gender_counts,
            'gender_percentages': {
                gender: (count / total_people * 100) if total_people > 0 else 0
                for gender, count in gender_counts.items()
            },
            'average_confidence': avg_confidence,
            'total_people': total_people
        }

# Global service instance
_age_gender_service = None

async def get_age_gender_service() -> AgeGenderDetectionService:
    """Get or create the age gender service instance"""
    global _age_gender_service
    
    if _age_gender_service is None:
        _age_gender_service = AgeGenderDetectionService()
        await _age_gender_service.initialize()
    
    return _age_gender_service
```

## 5.3 ชุดคำสั่ง Age & Gender API Endpoints

### 5.3.1 ชุดคำสั่ง Data Models

```python
"""
Age and Gender Detection Data Models
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class PersonAnalysis(BaseModel):
    """Analysis result for a single person"""
    person_id: int
    age: int
    age_range: str  # child, teenager, young_adult, adult, middle_aged, senior
    gender: str  # male, female
    gender_confidence: float
    face_region: Dict[str, int]
    dominant_emotion: Optional[str] = None
    race: Optional[str] = None

class AgeGenderResult(BaseModel):
    """Result of age and gender analysis"""
    success: bool
    faces_detected: int
    people: List[PersonAnalysis]
    processing_time: float
    detector_backend: str
    actions_performed: List[str]
    error: Optional[str] = None

class AgeGenderRequest(BaseModel):
    """Request for age and gender analysis"""
    image: str = Field(..., description="Base64 encoded image")
    detect_multiple: bool = Field(True, description="Detect multiple faces")
    detector_backend: str = Field("opencv", description="Face detector backend")

class BatchAgeGenderRequest(BaseModel):
    """Request for batch age and gender analysis"""
    images: List[str] = Field(..., description="List of base64 encoded images")
    detector_backend: str = Field("opencv", description="Face detector backend")

class BatchAgeGenderResult(BaseModel):
    """Result of batch age and gender analysis"""
    success: bool
    total_images: int
    total_faces: int
    processing_time: float
    results: List[AgeGenderResult]
    statistics: Dict[str, Any]
    error: Optional[str] = None

class AgeGenderStats(BaseModel):
    """Statistics for age and gender analysis"""
    age_stats: Dict[str, Any]
    gender_stats: Dict[str, Any]
    total_people: int
```

### 5.3.2 ชุดคำสั่ง API Endpoints

```python
"""
Age and Gender Detection API Endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import logging
import time
import base64
import io
from PIL import Image
import asyncio

logger = logging.getLogger(__name__)

# Create router
age_gender_router = APIRouter(
    prefix="/age-gender",
    tags=["Age & Gender Detection"],
    responses={404: {"description": "Not found"}}
)

@age_gender_router.post("/analyze", response_model=AgeGenderResult)
async def analyze_age_gender(
    file: UploadFile = File(..., description="Image file to analyze")
) -> AgeGenderResult:
    """
    Analyze age and gender from uploaded image
    
    - **file**: Image file (JPG, PNG, WebP)
    - Returns age and gender information for all detected faces
    """
    try:
        start_time = time.time()
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPG, PNG, WebP)"
            )
        
        # Read image data
        image_data = await file.read()
        
        if len(image_data) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        # Get service and analyze
        service = await get_age_gender_service()
        result = await service.analyze_age_gender(image_data)
        
        # Add processing time
        result['processing_time'] = time.time() - start_time
        
        return AgeGenderResult(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Age and gender analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@age_gender_router.post("/analyze-base64", response_model=AgeGenderResult)
async def analyze_age_gender_base64(
    request: AgeGenderRequest
) -> AgeGenderResult:
    """
    Analyze age and gender from base64 encoded image
    
    - **image**: Base64 encoded image data
    - **detect_multiple**: Whether to detect multiple faces (default: true)
    - **detector_backend**: Face detector backend (opencv, ssd, dlib, mtcnn, retinaface, mediapipe)
    """
    try:
        start_time = time.time()
        
        # Get service and analyze
        service = await get_age_gender_service()
        result = await service.analyze_age_gender_base64(request.image)
        
        # Add processing time
        result['processing_time'] = time.time() - start_time
        
        return AgeGenderResult(**result)
        
    except Exception as e:
        logger.error(f"Base64 age/gender analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@age_gender_router.post("/batch-analyze", response_model=BatchAgeGenderResult)
async def batch_analyze_age_gender(
    request: BatchAgeGenderRequest
) -> BatchAgeGenderResult:
    """
    Analyze age and gender for multiple images
    
    - **images**: List of base64 encoded images (max 10)
    - **detector_backend**: Face detector backend
    """
    try:
        if len(request.images) > 10:
            raise HTTPException(
                status_code=422,
                detail="Maximum 10 images allowed per batch"
            )
        
        # Convert base64 images to bytes
        image_bytes_list = []
        for i, image_b64 in enumerate(request.images):
            try:
                if image_b64.startswith('data:'):
                    image_b64 = image_b64.split(',')[1]
                image_data = base64.b64decode(image_b64)
                image_bytes_list.append(image_data)
            except Exception as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid base64 image at index {i}: {str(e)}"
                )
        
        # Get service and analyze
        service = await get_age_gender_service()
        result = await service.batch_analyze(image_bytes_list)
        
        return BatchAgeGenderResult(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch age/gender analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysis failed: {str(e)}"
        )

@age_gender_router.get("/status")
async def get_age_gender_status():
    """
    Get the status of the age and gender detection service
    """
    try:
        service = await get_age_gender_service()
        
        return {
            "service": "Age & Gender Detection",
            "status": "ready" if service.is_initialized else "not_ready",
            "detector_backend": service.detector_backend,
            "actions": service.actions,
            "description": "DeepFace-based age and gender detection"
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "service": "Age & Gender Detection",
            "status": "error",
            "error": str(e)
        }

@age_gender_router.get("/statistics/{time_period}")
async def get_analysis_statistics(
    time_period: str  # daily, weekly, monthly
):
    """
    Get analysis statistics for a specific time period
    """
    try:
        # This would typically query a database
        # For now, return mock statistics
        return {
            "time_period": time_period,
            "total_analyses": 0,
            "total_faces": 0,
            "age_distribution": {
                "child": 0,
                "teenager": 0,
                "young_adult": 0,
                "adult": 0,
                "middle_aged": 0,
                "senior": 0
            },
            "gender_distribution": {
                "male": 0,
                "female": 0,
                "unknown": 0
            },
            "average_processing_time": 0.0
        }
        
    except Exception as e:
        logger.error(f"Statistics retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )
```

## 5.4 ชุดคำสั่งการใช้งาน Age & Gender Detection ใน Frontend

### 5.4.1 ชุดคำสั่ง React Hook สำหรับ Age & Gender Analysis

```tsx
import { useState, useCallback } from 'react';

interface PersonAnalysis {
  person_id: number;
  age: number;
  age_range: string;
  gender: string;
  gender_confidence: number;
  face_region: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

interface AgeGenderResult {
  success: boolean;
  faces_detected: number;
  people: PersonAnalysis[];
  processing_time: number;
  detector_backend: string;
  actions_performed: string[];
  error?: string;
}

export const useAgeGenderAnalysis = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AgeGenderResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const analyzeImage = useCallback(async (
    imageBase64: string,
    detectorBackend: string = 'opencv'
  ) => {
    setIsAnalyzing(true);
    setError(null);

    try {
      const response = await fetch('/api/age-gender/analyze-base64', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageBase64,
          detect_multiple: true,
          detector_backend: detectorBackend
        })
      });

      const result = await response.json();
      
      if (result.success) {
        setResults(result);
      } else {
        setError(result.error || 'การวิเคราะห์ล้มเหลว');
      }
    } catch (err) {
      setError('เกิดข้อผิดพลาดในการเชื่อมต่อ');
      console.error('Age/Gender analysis error:', err);
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  const analyzeFile = useCallback(async (file: File) => {
    setIsAnalyzing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/age-gender/analyze', {
        method: 'POST',
        body: formData
      });

      const result = await response.json();
      
      if (result.success) {
        setResults(result);
      } else {
        setError(result.error || 'การวิเคราะห์ล้มเหลว');
      }
    } catch (err) {
      setError('เกิดข้อผิดพลาดในการเชื่อมต่อ');
      console.error('Age/Gender analysis error:', err);
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  const batchAnalyze = useCallback(async (
    imageList: string[],
    detectorBackend: string = 'opencv'
  ) => {
    setIsAnalyzing(true);
    setError(null);

    try {
      const response = await fetch('/api/age-gender/batch-analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          images: imageList,
          detector_backend: detectorBackend
        })
      });

      const result = await response.json();
      
      if (result.success) {
        setResults(result);
      } else {
        setError(result.error || 'การวิเคราะห์ล้มเหลว');
      }
    } catch (err) {
      setError('เกิดข้อผิดพลาดในการเชื่อมต่อ');
      console.error('Batch age/gender analysis error:', err);
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  return {
    analyzeImage,
    analyzeFile,
    batchAnalyze,
    isAnalyzing,
    results,
    error,
    reset: () => {
      setResults(null);
      setError(null);
    }
  };
};
```

### 5.4.2 ชุดคำสั่ง Age & Gender Analysis Component

```tsx
import React, { useState, useRef } from 'react';
import { 
  Card, 
  Button, 
  Upload, 
  Space, 
  Typography, 
  Alert, 
  Table, 
  Tag, 
  Progress,
  Select
} from 'antd';
import { 
  UploadOutlined, 
  UserOutlined, 
  ManOutlined, 
  WomanOutlined 
} from '@ant-design/icons';
import { useAgeGenderAnalysis } from './useAgeGenderAnalysis';

const { Title, Text } = Typography;
const { Option } = Select;

interface AgeGenderAnalysisProps {
  onAnalysisComplete?: (result: any) => void;
}

const AgeGenderAnalysis: React.FC<AgeGenderAnalysisProps> = ({
  onAnalysisComplete
}) => {
  const { analyzeFile, isAnalyzing, results, error, reset } = useAgeGenderAnalysis();
  const [detectorBackend, setDetectorBackend] = useState('opencv');

  const handleFileUpload = async (file: File) => {
    await analyzeFile(file);
    if (results && onAnalysisComplete) {
      onAnalysisComplete(results);
    }
    return false; // Prevent default upload
  };

  const getAgeRangeColor = (ageRange: string) => {
    const colors = {
      child: 'purple',
      teenager: 'blue',
      young_adult: 'green',
      adult: 'orange',
      middle_aged: 'red',
      senior: 'grey'
    };
    return colors[ageRange as keyof typeof colors] || 'default';
  };

  const getAgeRangeText = (ageRange: string) => {
    const texts = {
      child: 'เด็ก',
      teenager: 'วัยรุ่น',
      young_adult: 'วัยหนุ่มสาว',
      adult: 'ผู้ใหญ่',
      middle_aged: 'วัยกลางคน',
      senior: 'ผู้สูงอายุ'
    };
    return texts[ageRange as keyof typeof texts] || ageRange;
  };

  const columns = [
    {
      title: 'บุคคลที่',
      dataIndex: 'person_id',
      key: 'person_id',
      width: 80,
    },
    {
      title: 'อายุ',
      dataIndex: 'age',
      key: 'age',
      render: (age: number, record: any) => (
        <Space>
          <Text strong>{age} ปี</Text>
          <Tag color={getAgeRangeColor(record.age_range)}>
            {getAgeRangeText(record.age_range)}
          </Tag>
        </Space>
      ),
    },
    {
      title: 'เพศ',
      dataIndex: 'gender',
      key: 'gender',
      render: (gender: string, record: any) => (
        <Space>
          {gender === 'male' ? <ManOutlined style={{ color: '#1890ff' }} /> : <WomanOutlined style={{ color: '#eb2f96' }} />}
          <Text>{gender === 'male' ? 'ชาย' : 'หญิง'}</Text>
          <Text type="secondary">({(record.gender_confidence * 100).toFixed(1)}%)</Text>
        </Space>
      ),
    },
    {
      title: 'ตำแหน่งใบหน้า',
      dataIndex: 'face_region',
      key: 'face_region',
      render: (region: any) => (
        <Text code>
          {region.x}, {region.y} ({region.width}×{region.height})
        </Text>
      ),
    },
  ];

  return (
    <Card
      title={
        <Space>
          <UserOutlined />
          <Title level={4} style={{ margin: 0 }}>
            วิเคราะห์อายุและเพศ
          </Title>
        </Space>
      }
    >
      <Space direction="vertical" style={{ width: '100%' }}>
        {/* Controls */}
        <Space>
          <Text>โมเดลตรวจจับใบหน้า:</Text>
          <Select
            value={detectorBackend}
            onChange={setDetectorBackend}
            style={{ width: 150 }}
          >
            <Option value="opencv">OpenCV</Option>
            <Option value="mtcnn">MTCNN</Option>
            <Option value="retinaface">RetinaFace</Option>
            <Option value="mediapipe">MediaPipe</Option>
          </Select>
        </Space>

        {/* Upload */}
        <Upload
          beforeUpload={handleFileUpload}
          showUploadList={false}
          accept="image/*"
        >
          <Button
            type="primary"
            icon={<UploadOutlined />}
            loading={isAnalyzing}
            size="large"
          >
            อัปโหลดภาพเพื่อวิเคราะห์
          </Button>
        </Upload>

        {/* Progress */}
        {isAnalyzing && (
          <div>
            <Text>กำลังวิเคราะห์อายุและเพศ...</Text>
            <Progress percent={50} status="active" />
          </div>
        )}

        {/* Error */}
        {error && (
          <Alert
            message="เกิดข้อผิดพลาด"
            description={error}
            type="error"
            showIcon
            closable
            onClose={reset}
          />
        )}

        {/* Results */}
        {results && (
          <Card
            size="small"
            title="ผลการวิเคราะห์"
            extra={
              <Button size="small" onClick={reset}>
                ล้างผล
              </Button>
            }
          >
            <Space direction="vertical" style={{ width: '100%' }}>
              {/* Summary */}
              <div>
                <Text strong>
                  ตรวจพบ: {results.faces_detected} คน
                </Text>
                <br />
                <Text type="secondary">
                  เวลาประมวลผล: {results.processing_time.toFixed(2)} วินาที
                </Text>
                <br />
                <Text type="secondary">
                  โมเดล: {results.detector_backend}
                </Text>
              </div>

              {/* People Table */}
              {results.people && results.people.length > 0 && (
                <Table
                  dataSource={results.people}
                  columns={columns}
                  rowKey="person_id"
                  pagination={false}
                  size="small"
                />
              )}

              {/* Statistics */}
              {results.people && results.people.length > 1 && (
                <Card size="small" title="สถิติ">
                  <Space direction="vertical">
                    <div>
                      <Text strong>อายุเฉลี่ย: </Text>
                      <Text>
                        {(results.people.reduce((sum, p) => sum + p.age, 0) / results.people.length).toFixed(1)} ปี
                      </Text>
                    </div>
                    <div>
                      <Text strong>การกระจายเพศ: </Text>
                      <Space>
                        <Tag color="blue">
                          ชาย: {results.people.filter(p => p.gender === 'male').length} คน
                        </Tag>
                        <Tag color="pink">
                          หญิง: {results.people.filter(p => p.gender === 'female').length} คน
                        </Tag>
                      </Space>
                    </div>
                  </Space>
                </Card>
              )}
            </Space>
          </Card>
        )}
      </Space>
    </Card>
  );
};

export default AgeGenderAnalysis;
```

---

*เอกสารนี้แสดงชุดคำสั่งหลักสำหรับการวิเคราะห์อายุและเพศโดยใช้ DeepFace รวมถึง API endpoints และ React components สำหรับการใช้งานในระบบ*
