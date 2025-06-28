#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

from ..ai_services.anti_spoofing.anti_spoofing_service import anti_spoofing_service

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
    ตรวจจับภาพปลอม/หน้าจอมือถือ จากไฟล์ที่อัปโหลด
    
    - **image**: ไฟล์ภาพ (JPG, PNG, etc.)
    - **confidence_threshold**: Threshold สำหรับการตัดสินใจ (0.0-1.0)
    """
    try:
        if not anti_spoofing_service.is_initialized:
            raise HTTPException(
                status_code=503,
                detail="Anti-spoofing service not initialized. Please check DeepFace installation."
            )
          # ตรวจสอบไฟล์
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=422, detail="File must be an image")
        
        # อ่านไฟล์
        image_bytes = await image.read()
        
        # แปลงเป็น base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # ตรวจจับ spoofing
        result = anti_spoofing_service.detect_spoofing_from_base64(
            image_base64,
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
        
    except ValueError as e:
        logger.error(f"❌ Validation error in anti-spoofing upload: {e}")
        raise HTTPException(status_code=422, detail=str(e))
        
    except RuntimeError as e:
        logger.error(f"❌ Runtime error in anti-spoofing upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        logger.error(f"❌ Unexpected error in anti-spoofing upload: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@router.get("/model-info")
async def get_anti_spoofing_model_info():
    """
    ข้อมูลเกี่ยวกับโมเดล Anti-Spoofing
    
    Returns:
    - ข้อมูลโมเดล DeepFace Silent Face Anti-Spoofing
    - สถานะการเริ่มต้นระบบ
    - คุณสมบัติของโมเดล
    """
    try:
        model_info = anti_spoofing_service.get_model_info()
        
        return {
            "success": True,
            "model_info": model_info,
            "endpoints": {
                "detect_base64": "/api/anti-spoofing/detect-base64",
                "detect_upload": "/api/anti-spoofing/detect-upload",
                "model_info": "/api/anti-spoofing/model-info"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@router.get("/health")
async def anti_spoofing_health_check():
    """
    Health check สำหรับ Anti-Spoofing service
    
    Returns:
    - สถานะการทำงานของ service
    - ข้อมูล DeepFace
    """
    try:
        return {
            "service": "Anti-Spoofing",
            "status": "healthy" if anti_spoofing_service.is_initialized else "not_initialized",
            "deepface_available": anti_spoofing_service.is_initialized,
            "model": anti_spoofing_service.model_name,
            "message": "DeepFace Silent Face Anti-Spoofing ready" if anti_spoofing_service.is_initialized else "DeepFace not available"
        }
        
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return {
            "service": "Anti-Spoofing",
            "status": "error",
            "error": str(e)
        }
