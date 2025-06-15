#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ทดสอบ Framework Models แยกแต่ละตัว
เพื่อตรวจสอบว่าแต่ละโมเดลทำงานได้หรือไม่
"""

import os
import json
import time
import base64
import requests
import logging
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8080"
TEST_IMAGES_DIR = "test_images"

# Framework Models ที่จะทดสอบ
FRAMEWORK_MODELS = [
    "deepface",
    "facenet_pytorch", 
    "dlib",
    "insightface",
    "edgeface"
]

# ONNX Models สำหรับเปรียบเทียบ
ONNX_MODELS = [
    "facenet",
    "adaface", 
    "arcface"
]

def image_to_base64(image_path: str) -> str:
    """แปลงรูปภาพเป็น base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def test_model_available(model_name: str) -> bool:
    """ทดสอบว่าโมเดลพร้อมใช้งานหรือไม่"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/face-recognition/models/available", timeout=10)
        if response.status_code == 200:
            available_models = response.json().get('available_models', [])
            return model_name in available_models
        return False
    except Exception as e:
        logger.error(f"❌ Cannot check model availability: {e}")
        return False

def test_model_recognition(model_name: str, image_path: str) -> Dict[str, Any]:
    """ทดสอบการจดจำด้วยโมเดลเดียว"""
    try:
        # แปลงรูปเป็น base64
        image_base64 = image_to_base64(image_path)
        
        # สร้าง dummy gallery สำหรับทดสอบ
        dummy_gallery = {
            "test_person": [
                {
                    "embedding": [0.1] * 512,  # Dummy embedding
                    "person_name": "Test Person"
                }
            ]
        }
        
        # ส่งคำขอ recognition
        url = f"{API_BASE_URL}/api/face-recognition/recognize"
        data = {
            "face_image_base64": image_base64,
            "gallery": dummy_gallery,
            "model_name": model_name,
            "top_k": 1,
            "similarity_threshold": 0.3
        }
        
        response = requests.post(url, json=data, timeout=60)
        
        return {
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "response": response.json() if response.status_code == 200 else response.text,
            "error": None
        }
        
    except Exception as e:
        return {
            "status_code": None,
            "success": False,
            "response": None,
            "error": str(e)
        }

def test_model_embedding(model_name: str, image_path: str) -> Dict[str, Any]:
    """ทดสอบการสร้าง embedding"""
    try:
        # แปลงรูปเป็น base64
        image_base64 = image_to_base64(image_path)
        
        # ส่งคำขอ extract embedding
        url = f"{API_BASE_URL}/api/face-recognition/extract-embedding"
        data = {
            "face_image_base64": image_base64,
            "model_name": model_name
        }
        
        response = requests.post(url, json=data, timeout=60)
        
        return {
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "response": response.json() if response.status_code == 200 else response.text,
            "error": None
        }
        
    except Exception as e:
        return {
            "status_code": None,
            "success": False,
            "response": None,
            "error": str(e)
        }

def run_framework_models_test():
    """ทดสอบ Framework Models ทั้งหมด"""
    logger.info("🧪 Framework Models Testing")
    logger.info("=" * 60)
    
    # ตรวจสอบ API
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code != 200:
            raise Exception("API server not ready")
        logger.info("✅ API server is ready")
    except Exception as e:
        logger.error(f"❌ Cannot connect to API: {e}")
        return
    
    # เลือกรูปทดสอบ
    test_image_file = "boss_01.jpg"
    test_image_path = os.path.join(TEST_IMAGES_DIR, test_image_file)
    
    if not os.path.exists(test_image_path):
        logger.error(f"❌ Test image not found: {test_image_path}")
        return
    
    logger.info(f"📷 Using test image: {test_image_file}")
    
    # ผลลัพธ์การทดสอบ
    test_results = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "test_image": test_image_file,
        "framework_models": {},
        "onnx_models": {},
        "summary": {}
    }
    
    # ทดสอบ Framework Models
    logger.info("\n🔬 Testing Framework Models:")
    logger.info("-" * 40)
    
    for model_name in FRAMEWORK_MODELS:
        logger.info(f"\n📋 Testing {model_name.upper()}:")
        
        # 1. ตรวจสอบความพร้อมใช้งาน
        is_available = test_model_available(model_name)
        logger.info(f"  🔍 Available: {'✅ Yes' if is_available else '❌ No'}")
        
        if not is_available:
            test_results["framework_models"][model_name] = {
                "available": False,
                "embedding_test": {"success": False, "error": "Model not available"},
                "recognition_test": {"success": False, "error": "Model not available"}
            }
            continue
        
        # 2. ทดสอบ Embedding
        logger.info(f"  🧠 Testing embedding extraction...")
        embedding_result = test_model_embedding(model_name, test_image_path)
        
        if embedding_result["success"]:
            response_data = embedding_result["response"]
            if response_data and "embedding" in response_data:
                embedding_len = len(response_data["embedding"])
                logger.info(f"    ✅ Embedding: {embedding_len} dimensions")
            else:
                logger.info(f"    ⚠️ Embedding: Success but no embedding data")
        else:
            logger.error(f"    ❌ Embedding failed: {embedding_result.get('error', 'Unknown error')}")
        
        # 3. ทดสอบ Recognition
        logger.info(f"  🎯 Testing recognition...")
        recognition_result = test_model_recognition(model_name, test_image_path)
        
        if recognition_result["success"]:
            response_data = recognition_result["response"]
            if response_data and "matches" in response_data:
                matches_count = len(response_data["matches"])
                logger.info(f"    ✅ Recognition: {matches_count} matches found")
            else:
                logger.info(f"    ⚠️ Recognition: Success but no matches")
        else:
            logger.error(f"    ❌ Recognition failed: {recognition_result.get('error', 'Unknown error')}")
        
        # บันทึกผลลัพธ์
        test_results["framework_models"][model_name] = {
            "available": is_available,
            "embedding_test": embedding_result,
            "recognition_test": recognition_result
        }
        
        # รอสักครู่ก่อนทดสอบโมเดลถัดไป
        time.sleep(2)
    
    # ทดสอบ ONNX Models สำหรับเปรียบเทียบ
    logger.info("\n🔬 Testing ONNX Models (for comparison):")
    logger.info("-" * 40)
    
    for model_name in ONNX_MODELS:
        logger.info(f"\n📋 Testing {model_name.upper()}:")
        
        # 1. ตรวจสอบความพร้อมใช้งาน
        is_available = test_model_available(model_name)
        logger.info(f"  🔍 Available: {'✅ Yes' if is_available else '❌ No'}")
        
        if not is_available:
            test_results["onnx_models"][model_name] = {
                "available": False,
                "embedding_test": {"success": False, "error": "Model not available"},
                "recognition_test": {"success": False, "error": "Model not available"}
            }
            continue
        
        # 2. ทดสอบ Embedding
        embedding_result = test_model_embedding(model_name, test_image_path)
        logger.info(f"  🧠 Embedding: {'✅ Success' if embedding_result['success'] else '❌ Failed'}")
        
        # 3. ทดสอบ Recognition  
        recognition_result = test_model_recognition(model_name, test_image_path)
        logger.info(f"  🎯 Recognition: {'✅ Success' if recognition_result['success'] else '❌ Failed'}")
        
        # บันทึกผลลัพธ์
        test_results["onnx_models"][model_name] = {
            "available": is_available,
            "embedding_test": embedding_result,
            "recognition_test": recognition_result
        }
        
        time.sleep(1)
    
    # สรุปผลลัพธ์
    logger.info("\n📊 SUMMARY:")
    logger.info("=" * 60)
    
    # นับผลลัพธ์
    framework_working = 0
    framework_available = 0
    onnx_working = 0
    onnx_available = 0
    
    logger.info("\n🔧 Framework Models:")
    for model_name, result in test_results["framework_models"].items():
        if result["available"]:
            framework_available += 1
            if result["embedding_test"]["success"] and result["recognition_test"]["success"]:
                framework_working += 1
                status = "✅ WORKING"
            else:
                status = "⚠️ PARTIAL"
        else:
            status = "❌ NOT AVAILABLE"
        
        logger.info(f"  {model_name:15} : {status}")
    
    logger.info("\n🎯 ONNX Models:")
    for model_name, result in test_results["onnx_models"].items():
        if result["available"]:
            onnx_available += 1
            if result["embedding_test"]["success"] and result["recognition_test"]["success"]:
                onnx_working += 1
                status = "✅ WORKING"
            else:
                status = "⚠️ PARTIAL"
        else:
            status = "❌ NOT AVAILABLE"
        
        logger.info(f"  {model_name:15} : {status}")
    
    # สรุปรวม
    total_framework = len(FRAMEWORK_MODELS)
    total_onnx = len(ONNX_MODELS)
    
    logger.info(f"\n📋 FINAL SUMMARY:")
    logger.info(f"  Framework Models: {framework_working}/{framework_available}/{total_framework} (Working/Available/Total)")
    logger.info(f"  ONNX Models:      {onnx_working}/{onnx_available}/{total_onnx} (Working/Available/Total)")
    logger.info(f"  Total Working:    {framework_working + onnx_working}/{framework_available + onnx_available} models")
    
    # บันทึกผลลัพธ์
    test_results["summary"] = {
        "framework_models": {
            "total": total_framework,
            "available": framework_available,
            "working": framework_working
        },
        "onnx_models": {
            "total": total_onnx,
            "available": onnx_available,
            "working": onnx_working
        }
    }
    
    # บันทึกรายงาน
    output_file = "framework_models_test_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n📄 Detailed report saved: {output_file}")
    
    # แนะนำขั้นตอนถัดไป
    if framework_working > 0:
        logger.info(f"\n✅ Good news! {framework_working} framework models are working.")
        logger.info("📝 You can now proceed with the real-world test.")
    else:
        logger.warning(f"\n⚠️ Warning: No framework models are working properly.")
        logger.warning("🔧 You may need to check model installations or configurations.")

if __name__ == "__main__":
    try:
        run_framework_models_test()
    except Exception as e:
        logger.error(f"🛑 Test terminated: {e}")
        print(f"\n🛑 TEST TERMINATED DUE TO ERROR: {e}")
        exit(1)
