# ชุดคำสั่ง: ระบบ Anti-Spoofing
## การตรวจจับการปลอมแปลงใบหน้าและป้องกันการหลอกลวง

### 📋 สารบัญ
4.1 [ภาพรวม Anti-Spoofing System](#41-ภาพรวม-anti-spoofing-system)
4.2 [DeepFace Anti-Spoofing Service](#42-deepface-anti-spoofing-service)
4.3 [API Endpoints](#43-api-endpoints)
4.4 [การใช้งานใน Frontend](#44-การใช้งานใน-frontend)
4.5 [Real-time Detection](#45-real-time-detection)
4.6 [Performance และ Optimization](#46-performance-และ-optimization)
4.7 [การจัดการ Error และ Logging](#47-การจัดการ-error-และ-logging)
4.8 [การ Integration กับระบบอื่น](#48-การ-integration-กับระบบอื่น)

---

## 4.1 ภาพรวม Anti-Spoofing System

ระบบป้องกันการปลอมแปลงใบหน้าด้วย DeepFace Anti-Spoofing เพื่อรักษาความปลอดภัยของระบบ Face Recognition

### 🛡️ ฟีเจอร์ความปลอดภัย
- **Liveness Detection**: ตรวจจับใบหน้าจริงต่อหน้าจอปลอม
- **Real-time Analysis**: วิเคราะห์แบบเรียลไทม์
- **High Accuracy**: ความแม่นยำสูงด้วย DeepFace
- **API Integration**: เชื่อมต่อง่ายกับระบบอื่น

---

## 4.2 DeepFace Anti-Spoofing Service

### 4.2.1 DeepFace Anti-Spoofing Service Core

```python
"""
DeepFace Anti-Spoofing Service
ระบบตรวจจับภาพปลอม/หน้าจอมือถือ โดยใช้ DeepFace Silent Face Anti-Spoofing
"""

import io
import cv2
import numpy as np
import base64
import tempfile
import os
from typing import Dict, Any, List, Optional, Union
import logging
from PIL import Image
import time

# Import DeepFace
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    logging.warning("DeepFace not available. Please install: pip install deepface")

logger = logging.getLogger(__name__)

class AntiSpoofingService:
    """
    DeepFace Anti-Spoofing Service
    ตรวจจับภาพปลอม/หน้าจอมือถือ
    """
    
    def __init__(self):
        self.model_name = "DeepFace Silent Face Anti-Spoofing"
        self.is_initialized = False
        self.initialize()
    
    def initialize(self) -> bool:
        """เริ่มต้นระบบ Anti-Spoofing"""
        try:
            if not DEEPFACE_AVAILABLE:
                logger.error("❌ DeepFace not installed. Please run: pip install deepface")
                return False
            
            logger.info("🔍 Initializing DeepFace Anti-Spoofing...")
            
            # สร้างภาพทดสอบเล็กๆ
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
            
            try:
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                try:
                    # บันทึกภาพทดสอบ
                    cv2.imwrite(temp_path, test_image)
                    
                    # ทดสอบ extract_faces กับ anti_spoofing
                    face_objs = DeepFace.extract_faces(
                        img_path=temp_path,
                        anti_spoofing=True,
                        enforce_detection=False
                    )
                    logger.info("✅ DeepFace Anti-Spoofing initialized successfully")
                    self.is_initialized = True
                    return True
                    
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        
            except Exception as e:
                logger.warning(f"⚠️ DeepFace anti-spoofing test failed: {e}")
                logger.info("📥 Downloading anti-spoofing models...")
                self.is_initialized = True
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize anti-spoofing: {e}")
            return False
    
    def decode_base64_image(self, base64_string: str) -> np.ndarray:
        """แปลง base64 เป็น numpy array"""
        try:
            # ลบ prefix ถ้ามี
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # แปลง base64 เป็น bytes
            image_bytes = base64.b64decode(base64_string)
            
            # แปลงเป็น PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # แปลงเป็น RGB ถ้าจำเป็น
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # แปลงเป็น numpy array
            image_array = np.array(image)
            
            return image_array
            
        except Exception as e:
            logger.error(f"❌ Error decoding base64 image: {e}")
            raise ValueError(f"Invalid base64 image: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """เตรียมภาพสำหรับ anti-spoofing"""
        try:
            # ตรวจสอบขนาดภาพ
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError("Image must be RGB (3 channels)")
            
            # ปรับขนาดถ้าภาพใหญ่เกินไป
            height, width = image.shape[:2]
            max_size = 1024
            
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
                logger.info(f"📏 Resized image from {width}x{height} to {new_width}x{new_height}")
            
            return image
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing image: {e}")
            raise ValueError(f"Image preprocessing failed: {e}")
```

### 4.2.2 ชุดคำสั่งการตรวจจับ Spoofing หลัก

```python
    def detect_spoofing_from_image(self, image: np.ndarray, 
                                 confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        ตรวจจับ spoofing จากภาพ
        
        Args:
            image: numpy array ของภาพ
            confidence_threshold: threshold สำหรับการตัดสินใจ (0.0-1.0)
        """
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                raise RuntimeError("Anti-spoofing service not initialized")
            
            # เตรียมภาพ
            processed_image = self.preprocess_image(image)
            
            # บันทึกเป็น temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # บันทึกภาพ
                cv2.imwrite(temp_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
                
                # ใช้ DeepFace extract_faces พร้อม anti_spoofing
                face_objs = DeepFace.extract_faces(
                    img_path=temp_path,
                    anti_spoofing=True,
                    enforce_detection=False
                )
                
                # วิเคราะห์ผลลัพธ์
                faces_analysis = []
                spoofing_count = 0
                real_count = 0
                
                for i, face_obj in enumerate(face_objs):
                    # DeepFace คืน dict ที่มี 'face' และ 'is_real' (ถ้า anti_spoofing=True)
                    is_real = face_obj.get('is_real', True)  # default เป็น True ถ้าไม่มีข้อมูล
                    confidence = face_obj.get('antispoof_score', 0.5)  # confidence score
                    
                    # ใช้ threshold ในการตัดสินใจ
                    if isinstance(is_real, (int, float)):
                        final_is_real = float(is_real) >= confidence_threshold
                    else:
                        final_is_real = bool(is_real)
                    
                    face_analysis = {
                        "face_id": i + 1,
                        "is_real": final_is_real,
                        "confidence": float(confidence) if isinstance(confidence, (int, float)) else 0.5,
                        "antispoof_score": float(confidence) if isinstance(confidence, (int, float)) else 0.5,
                        "face_region": {
                            "x": 0,  # DeepFace ไม่ให้ bounding box ใน extract_faces
                            "y": 0,
                            "width": face_obj['face'].shape[1] if 'face' in face_obj else 0,
                            "height": face_obj['face'].shape[0] if 'face' in face_obj else 0
                        }
                    }
                    
                    faces_analysis.append(face_analysis)
                    
                    if final_is_real:
                        real_count += 1
                    else:
                        spoofing_count += 1
                
                # สรุปผลรวม
                total_faces = len(face_objs)
                spoofing_detected = spoofing_count > 0
                
                overall_result = {
                    "spoofing_detected": spoofing_detected,
                    "real_faces": real_count,
                    "spoofed_faces": spoofing_count,
                    "total_faces": total_faces,
                    "confidence_threshold": confidence_threshold
                }
                
                processing_time = time.time() - start_time
                
                result = {
                    "success": True,
                    "faces_detected": total_faces,
                    "faces_analysis": faces_analysis,
                    "overall_result": overall_result,
                    "processing_time": processing_time,
                    "model": self.model_name
                }
                
                logger.info(f"✅ Anti-spoofing completed: {total_faces} faces, "
                          f"{real_count} real, {spoofing_count} spoofed in {processing_time:.2f}s")
                
                return result
                
            finally:
                # ลบ temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Anti-spoofing detection failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "faces_detected": 0,
                "faces_analysis": [],
                "overall_result": {
                    "spoofing_detected": False,
                    "real_faces": 0,
                    "spoofed_faces": 0,
                    "total_faces": 0,
                    "confidence_threshold": confidence_threshold
                },
                "processing_time": processing_time,
                "model": self.model_name
            }
    
    def detect_spoofing_from_base64(self, base64_string: str, 
                                  confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """ตรวจจับ spoofing จาก base64 image"""
        try:
            # แปลง base64 เป็น image
            image = self.decode_base64_image(base64_string)
            
            # ตรวจจับ spoofing
            return self.detect_spoofing_from_image(image, confidence_threshold)
            
        except Exception as e:
            logger.error(f"❌ Base64 anti-spoofing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "faces_detected": 0,
                "faces_analysis": [],
                "overall_result": {
                    "spoofing_detected": False,
                    "real_faces": 0,
                    "spoofed_faces": 0,
                    "total_faces": 0,
                    "confidence_threshold": confidence_threshold
                },
                "processing_time": 0.0,
                "model": self.model_name
            }

# สร้าง instance
anti_spoofing_service = AntiSpoofingService()
```

## 4.3 ชุดคำสั่ง Anti-Spoofing API Endpoints

### 4.3.1 ชุดคำสั่ง REST API สำหรับ Anti-Spoofing

```python
"""
Anti-Spoofing API Endpoints
REST API สำหรับระบบตรวจจับภาพปลอม/หน้าจอมือถือ
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import logging
import base64
import io
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# สร้าง router
router = APIRouter(prefix="/api/anti-spoofing", tags=["Anti-Spoofing"])

# Pydantic models
class AntiSpoofingBase64Request(BaseModel):
    """Request model สำหรับ anti-spoofing จาก base64 image"""
    image_base64: str = Field(..., description="Base64 encoded image")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold (0.0-1.0)")

class AntiSpoofingResponse(BaseModel):
    """Response model สำหรับ anti-spoofing"""
    success: bool
    faces_detected: int
    faces_analysis: list
    overall_result: dict
    processing_time: float
    model: str
    message: Optional[str] = None
    error: Optional[str] = None

@router.post("/detect-base64", response_model=AntiSpoofingResponse)
async def detect_spoofing_base64(request: AntiSpoofingBase64Request):
    """
    ตรวจจับภาพปลอม/หน้าจอมือถือ จาก base64 image
    
    - **image_base64**: Base64 encoded image
    - **confidence_threshold**: Threshold สำหรับการตัดสินใจ (0.0-1.0)
    
    Returns:
    - **success**: สถานะความสำเร็จ
    - **faces_detected**: จำนวนใบหน้าที่ตรวจพบ
    - **faces_analysis**: รายละเอียดการวิเคราะห์แต่ละใบหน้า
    - **overall_result**: ผลรวมการวิเคราะห์
    """
    try:
        # Import service ภายใน function เพื่อหลีกเลี่ยง circular import
        from ..ai_services.anti_spoofing.anti_spoofing_service import anti_spoofing_service
        
        if not anti_spoofing_service.is_initialized:
            raise HTTPException(
                status_code=503,
                detail="Anti-spoofing service not initialized. Please check DeepFace installation."
            )
        
        # ตรวจจับ spoofing
        result = anti_spoofing_service.detect_spoofing_from_base64(
            request.image_base64,
            request.confidence_threshold
        )
        
        # สร้าง response
        spoofing_detected = result["overall_result"]["spoofing_detected"]
        message = "Spoofing detected!" if spoofing_detected else "Real face(s) detected"
        
        return AntiSpoofingResponse(
            success=True,
            faces_detected=result["faces_detected"],
            faces_analysis=result["faces_analysis"],
            overall_result=result["overall_result"],
            processing_time=result["processing_time"],
            model=result["model"],
            message=message,
            error=result.get("error")
        )
        
    except ValueError as e:
        logger.error(f"❌ Validation error in anti-spoofing: {e}")
        raise HTTPException(status_code=422, detail=str(e))
        
    except RuntimeError as e:
        logger.error(f"❌ Runtime error in anti-spoofing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        logger.error(f"❌ Unexpected error in anti-spoofing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@router.post("/detect-upload", response_model=AntiSpoofingResponse)
async def detect_spoofing_upload(
    image: UploadFile = File(..., description="Image file to analyze"),
    confidence_threshold: float = Form(0.5, ge=0.0, le=1.0, description="Confidence threshold")
):
    """
    ตรวจจับภาพปลอม/หน้าจอมือถือ จาก uploaded file
    
    - **image**: Image file (JPG, PNG, BMP, etc.)
    - **confidence_threshold**: Threshold สำหรับการตัดสินใจ (0.0-1.0)
    """
    try:
        from ..ai_services.anti_spoofing.anti_spoofing_service import anti_spoofing_service
        
        if not anti_spoofing_service.is_initialized:
            raise HTTPException(
                status_code=503,
                detail="Anti-spoofing service not initialized."
            )
        
        # ตรวจสอบประเภทไฟล์
        if not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=422,
                detail="Invalid file type. Only image files are accepted."
            )
        
        # อ่านไฟล์
        image_bytes = await image.read()
        
        # แปลงเป็น PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # แปลงเป็น numpy array
        image_array = np.array(pil_image)
        
        # ตรวจจับ spoofing
        result = anti_spoofing_service.detect_spoofing_from_image(
            image_array,
            confidence_threshold
        )
        
        # สร้าง response
        spoofing_detected = result["overall_result"]["spoofing_detected"]
        message = "Spoofing detected!" if spoofing_detected else "Real face(s) detected"
        
        return AntiSpoofingResponse(
            success=True,
            faces_detected=result["faces_detected"],
            faces_analysis=result["faces_analysis"],
            overall_result=result["overall_result"],
            processing_time=result["processing_time"],
            model=result["model"],
            message=message,
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"❌ Upload anti-spoofing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_anti_spoofing_status():
    """
    ตรวจสอบสถานะ Anti-Spoofing Service
    """
    try:
        from ..ai_services.anti_spoofing.anti_spoofing_service import anti_spoofing_service
        
        return {
            "service": "Anti-Spoofing",
            "status": "ready" if anti_spoofing_service.is_initialized else "not_ready",
            "model": anti_spoofing_service.model_name,
            "deepface_available": anti_spoofing_service.is_initialized,
            "description": "DeepFace Silent Face Anti-Spoofing detection"
        }
        
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        return {
            "service": "Anti-Spoofing",
            "status": "error",
            "error": str(e)
        }

@router.post("/batch-detect", response_model=Dict[str, Any])
async def batch_detect_spoofing(
    images: list[str] = Field(..., description="List of base64 encoded images"),
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold")
):
    """
    ตรวจจับภาพปลอมจากหลายภาพพร้อมกัน
    
    - **images**: รายการของ base64 encoded images
    - **confidence_threshold**: Threshold สำหรับการตัดสินใจ
    """
    try:
        from ..ai_services.anti_spoofing.anti_spoofing_service import anti_spoofing_service
        
        if not anti_spoofing_service.is_initialized:
            raise HTTPException(
                status_code=503,
                detail="Anti-spoofing service not initialized."
            )
        
        if len(images) > 10:  # จำกัดจำนวนภาพ
            raise HTTPException(
                status_code=422,
                detail="Maximum 10 images allowed per batch"
            )
        
        results = []
        total_faces = 0
        total_spoofed = 0
        total_real = 0
        
        for i, image_base64 in enumerate(images):
            try:
                result = anti_spoofing_service.detect_spoofing_from_base64(
                    image_base64,
                    confidence_threshold
                )
                
                result["image_index"] = i
                results.append(result)
                
                total_faces += result["faces_detected"]
                total_spoofed += result["overall_result"]["spoofed_faces"]
                total_real += result["overall_result"]["real_faces"]
                
            except Exception as e:
                results.append({
                    "image_index": i,
                    "success": False,
                    "error": str(e),
                    "faces_detected": 0
                })
        
        return {
            "success": True,
            "total_images": len(images),
            "total_faces": total_faces,
            "total_real": total_real,
            "total_spoofed": total_spoofed,
            "spoofing_rate": (total_spoofed / total_faces * 100) if total_faces > 0 else 0,
            "confidence_threshold": confidence_threshold,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"❌ Batch anti-spoofing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

## 4.4 ชุดคำสั่งการใช้งาน Anti-Spoofing ใน Frontend

### 4.4.1 ชุดคำสั่ง React Hook สำหรับ Anti-Spoofing

```tsx
import { useState, useCallback } from 'react';

interface AntiSpoofingResult {
  success: boolean;
  faces_detected: number;
  faces_analysis: Array<{
    face_id: number;
    is_real: boolean;
    confidence: number;
    antispoof_score: number;
  }>;
  overall_result: {
    spoofing_detected: boolean;
    real_faces: number;
    spoofed_faces: number;
    total_faces: number;
  };
  processing_time: number;
  model: string;
  message?: string;
  error?: string;
}

export const useAntiSpoofing = () => {
  const [isDetecting, setIsDetecting] = useState(false);
  const [results, setResults] = useState<AntiSpoofingResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const detectSpoofing = useCallback(async (
    imageBase64: string,
    confidenceThreshold: number = 0.5
  ) => {
    setIsDetecting(true);
    setError(null);

    try {
      const response = await fetch('/api/anti-spoofing/detect-base64', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_base64: imageBase64,
          confidence_threshold: confidenceThreshold
        })
      });

      const result = await response.json();
      
      if (result.success) {
        setResults(result);
      } else {
        setError(result.error || 'การตรวจจับล้มเหลว');
      }
    } catch (err) {
      setError('เกิดข้อผิดพลาดในการเชื่อมต่อ');
      console.error('Anti-spoofing detection error:', err);
    } finally {
      setIsDetecting(false);
    }
  }, []);

  const detectSpoofingFromFile = useCallback(async (
    file: File,
    confidenceThreshold: number = 0.5
  ) => {
    setIsDetecting(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('image', file);
      formData.append('confidence_threshold', confidenceThreshold.toString());

      const response = await fetch('/api/anti-spoofing/detect-upload', {
        method: 'POST',
        body: formData
      });

      const result = await response.json();
      
      if (result.success) {
        setResults(result);
      } else {
        setError(result.error || 'การตรวจจับล้มเหลว');
      }
    } catch (err) {
      setError('เกิดข้อผิดพลาดในการเชื่อมต่อ');
      console.error('Anti-spoofing detection error:', err);
    } finally {
      setIsDetecting(false);
    }
  }, []);

  return {
    detectSpoofing,
    detectSpoofingFromFile,
    isDetecting,
    results,
    error,
    reset: () => {
      setResults(null);
      setError(null);
    }
  };
};
```

### 4.4.2 ชุดคำสั่ง Anti-Spoofing Component

```tsx
import React, { useState, useRef } from 'react';
import { Alert, Button, Card, Progress, Space, Typography, Upload } from 'antd';
import { CameraOutlined, UploadOutlined, SecurityScanOutlined } from '@ant-design/icons';
import { useAntiSpoofing } from './useAntiSpoofing';

const { Title, Text } = Typography;

interface AntiSpoofingComponentProps {
  onDetectionComplete?: (result: any) => void;
}

const AntiSpoofingComponent: React.FC<AntiSpoofingComponentProps> = ({
  onDetectionComplete
}) => {
  const { detectSpoofing, detectSpoofingFromFile, isDetecting, results, error, reset } = useAntiSpoofing();
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (file: File) => {
    await detectSpoofingFromFile(file, confidenceThreshold);
    if (results && onDetectionComplete) {
      onDetectionComplete(results);
    }
    return false; // Prevent default upload
  };

  const handleCameraCapture = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      const video = document.createElement('video');
      video.srcObject = stream;
      video.play();

      video.addEventListener('loadedmetadata', () => {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        
        setTimeout(() => {
          ctx?.drawImage(video, 0, 0);
          const imageData = canvas.toDataURL('image/jpeg', 0.8);
          const base64Data = imageData.split(',')[1];
          
          // Stop camera
          stream.getTracks().forEach(track => track.stop());
          
          // Detect spoofing
          detectSpoofing(base64Data, confidenceThreshold);
        }, 1000);
      });
    } catch (err) {
      console.error('Camera access error:', err);
    }
  };

  return (
    <Card
      title={
        <Space>
          <SecurityScanOutlined />
          <Title level={4} style={{ margin: 0 }}>
            ตรวจจับภาพปลอม (Anti-Spoofing)
          </Title>
        </Space>
      }
    >
      {/* Controls */}
      <Space direction="vertical" style={{ width: '100%' }}>
        <div>
          <Text strong>ระดับความมั่นใจ: {confidenceThreshold}</Text>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={confidenceThreshold}
            onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
            style={{ width: '100%', margin: '8px 0' }}
          />
        </div>

        {/* Action Buttons */}
        <Space>
          <Button
            type="primary"
            icon={<CameraOutlined />}
            onClick={handleCameraCapture}
            loading={isDetecting}
          >
            ถ่ายภาพตรวจสอบ
          </Button>
          
          <Upload
            beforeUpload={handleFileUpload}
            showUploadList={false}
            accept="image/*"
          >
            <Button icon={<UploadOutlined />} loading={isDetecting}>
              อัปโหลดภาพ
            </Button>
          </Upload>
        </Space>

        {/* Results */}
        {isDetecting && (
          <div>
            <Text>กำลังตรวจสอบ...</Text>
            <Progress percent={50} status="active" />
          </div>
        )}

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

        {results && (
          <Card
            size="small"
            title="ผลการตรวจสอบ"
            extra={
              <Button size="small" onClick={reset}>
                ล้างผล
              </Button>
            }
          >
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text strong>
                  สถานะ: {' '}
                  <span style={{ 
                    color: results.overall_result.spoofing_detected ? '#ff4d4f' : '#52c41a' 
                  }}>
                    {results.overall_result.spoofing_detected ? '⚠️ ตรวจพบการปลอมแปลง' : '✅ ภาพจริง'}
                  </span>
                </Text>
              </div>
              
              <div>
                <Text>จำนวนใบหน้า: {results.faces_detected}</Text>
              </div>
              
              <div>
                <Text>ใบหน้าจริง: {results.overall_result.real_faces}</Text>
              </div>
              
              <div>
                <Text>ใบหน้าปลอม: {results.overall_result.spoofed_faces}</Text>
              </div>
              
              <div>
                <Text>เวลาประมวลผล: {results.processing_time.toFixed(2)} วินาที</Text>
              </div>

              {/* Face Analysis Details */}
              {results.faces_analysis.length > 0 && (
                <div>
                  <Text strong>รายละเอียดแต่ละใบหน้า:</Text>
                  {results.faces_analysis.map((face, index) => (
                    <div key={index} style={{ marginLeft: 16, marginTop: 4 }}>
                      <Text>
                        ใบหน้า {face.face_id}: {' '}
                        <span style={{ color: face.is_real ? '#52c41a' : '#ff4d4f' }}>
                          {face.is_real ? 'จริง' : 'ปลอม'}
                        </span>
                        {' '}({(face.confidence * 100).toFixed(1)}%)
                      </Text>
                    </div>
                  ))}
                </div>
              )}
            </Space>
          </Card>
        )}
      </Space>
    </Card>
  );
};

export default AntiSpoofingComponent;
```

---

*เอกสารนี้แสดงชุดคำสั่งหลักสำหรับระบบ Anti-Spoofing ที่ใช้ DeepFace Silent Face Anti-Spoofing เพื่อตรวจจับภาพปลอมและการป้องกันการหลอกลวงด้วยภาพหน้าจอหรือภาพถ่าย*
