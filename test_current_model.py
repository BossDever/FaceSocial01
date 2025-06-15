#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ทดสอบโมเดลที่ใช้งานอยู่ในปัจจุบัน
"""

import os
import json
import base64
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8080"
TEST_IMAGES_DIR = "test_images"

def image_to_base64(image_path: str) -> str:
    """แปลงรูปภาพเป็น base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def test_current_model():
    """ทดสอบโมเดลปัจจุบัน"""
    # ตรวจสอบสถานะ
    health_response = requests.get(f"{API_BASE_URL}/api/face-recognition/health")
    if health_response.status_code == 200:
        health_data = health_response.json()
        current_model = health_data.get("service_info", {}).get("model_info", {}).get("current_model", "unknown")
        logger.info(f"📊 Current model: {current_model}")
    else:
        logger.error("❌ Cannot get health status")
        return
    
    # ทดสอบด้วยรูปจริง
    test_image_path = os.path.join(TEST_IMAGES_DIR, "boss_01.jpg")
    if not os.path.exists(test_image_path):
        logger.error(f"❌ Test image not found: {test_image_path}")
        return
    
    logger.info(f"📷 Testing with: {test_image_path}")
    
    # ทดสอบ extract embedding
    logger.info("🧠 Testing embedding extraction...")
    image_base64 = image_to_base64(test_image_path)
    
    embedding_data = {
        "face_image_base64": image_base64,
        "model_name": current_model  # ใช้โมเดลปัจจุบัน
    }
    
    embedding_response = requests.post(
        f"{API_BASE_URL}/api/face-recognition/extract-embedding", 
        json=embedding_data, 
        timeout=60
    )
    
    if embedding_response.status_code == 200:
        result = embedding_response.json()
        if "embedding" in result:
            embedding_len = len(result["embedding"])
            logger.info(f"✅ Embedding extracted: {embedding_len} dimensions")
            
            # ทดสอบ recognition
            logger.info("🎯 Testing recognition...")  
            
            # สร้าง gallery จำลอง
            dummy_gallery = {
                "test_person": [{
                    "embedding": result["embedding"],  # ใช้ embedding ที่เพิ่งสร้าง
                    "person_name": "Test Person"
                }]
            }
            
            recognition_data = {
                "face_image_base64": image_base64,
                "gallery": dummy_gallery,
                "model_name": current_model,
                "top_k": 1,
                "similarity_threshold": 0.3
            }
            
            recognition_response = requests.post(
                f"{API_BASE_URL}/api/face-recognition/recognize",
                json=recognition_data,
                timeout=60
            )
            
            if recognition_response.status_code == 200:
                rec_result = recognition_response.json()
                if "matches" in rec_result and rec_result["matches"]:
                    match = rec_result["matches"][0]
                    similarity = match.get("similarity", 0)
                    logger.info(f"✅ Recognition successful: similarity = {similarity:.3f}")
                else:
                    logger.info("⚠️ Recognition successful but no matches found")
            else:
                logger.error(f"❌ Recognition failed: {recognition_response.status_code}")
                logger.error(f"Response: {recognition_response.text}")
        else:
            logger.error("❌ No embedding in response")
            logger.error(f"Response: {result}")
    else:
        logger.error(f"❌ Embedding extraction failed: {embedding_response.status_code}")
        logger.error(f"Response: {embedding_response.text}")

if __name__ == "__main__":
    test_current_model()
