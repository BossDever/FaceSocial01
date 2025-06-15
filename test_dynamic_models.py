#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ทดสอบระบบ Dynamic Model Switching
ระบบใช้โมเดลเดียวแต่สามารถเปลี่ยนได้ผ่าน request
"""

import os
import base64
import requests
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8080"
TEST_IMAGES_DIR = "test_images"

# รายการโมเดลที่จะทดสอบ
ALL_MODELS = [
    # ONNX Models
    "facenet", "adaface", "arcface",
    # Framework Models  
    "deepface", "facenet_pytorch", "dlib", "insightface", "edgeface"
]

def image_to_base64(image_path: str) -> str:
    """แปลงรูปภาพเป็น base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def test_model_with_dynamic_switching(model_name: str, image_base64: str):
    """ทดสอบโมเดลด้วย dynamic switching"""
    logger.info(f"🧠 Testing {model_name.upper()}...")
    
    # สร้าง dummy gallery สำหรับทดสอบ
    dummy_gallery = {
        "test_person": [{
            "embedding": [0.1] * 512,  # Dummy embedding
            "person_name": "Test Person"
        }]
    }
    
    # ส่งคำขอ recognition พร้อม model_name
    recognition_data = {
        "face_image_base64": image_base64,
        "gallery": dummy_gallery,
        "model_name": model_name,  # ระบุโมเดลที่ต้องการ
        "top_k": 1,
        "similarity_threshold": 0.3
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/face-recognition/recognize",
            json=recognition_data,
            timeout=120  # เพิ่ม timeout เพราะอาจต้องโหลดโมเดล
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"  ✅ {model_name}: SUCCESS")
            
            # ตรวจสอบผลลัพธ์
            if "matches" in result:
                matches_count = len(result["matches"])
                logger.info(f"    📊 Found {matches_count} matches")
            
            if "processing_time" in result:
                proc_time = result["processing_time"]
                logger.info(f"    ⏱️ Processing time: {proc_time:.3f}s")
            
            return {
                "success": True,
                "status_code": 200,
                "response": result,
                "error": None
            }
        else:
            logger.error(f"  ❌ {model_name}: FAILED ({response.status_code})")
            error_text = response.text
            logger.error(f"    💬 Error: {error_text}")
            
            return {
                "success": False,
                "status_code": response.status_code,
                "response": None,
                "error": error_text
            }
            
    except Exception as e:
        logger.error(f"  ❌ {model_name}: EXCEPTION")
        logger.error(f"    💬 Error: {str(e)}")
        
        return {
            "success": False,
            "status_code": None,
            "response": None,
            "error": str(e)
        }

def run_dynamic_model_test():
    """ทดสอบ dynamic model switching"""
    logger.info("🔄 Dynamic Model Switching Test")
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
    image_base64 = image_to_base64(test_image_path)
    
    # ทดสอบทุกโมเดล
    results = {}
    working_models = []
    failed_models = []
    
    logger.info(f"\n🧪 Testing {len(ALL_MODELS)} models...")
    logger.info("-" * 40)
    
    for i, model_name in enumerate(ALL_MODELS, 1):
        logger.info(f"\n[{i}/{len(ALL_MODELS)}] {model_name.upper()}")
        
        result = test_model_with_dynamic_switching(model_name, image_base64)
        results[model_name] = result
        
        if result["success"]:
            working_models.append(model_name)
        else:
            failed_models.append(model_name)
        
        # รอสักครู่ก่อนทดสอบโมเดลถัดไป
        if i < len(ALL_MODELS):
            time.sleep(3)
    
    # สรุปผลลัพธ์
    logger.info("\n📊 SUMMARY:")
    logger.info("=" * 60)
    
    logger.info(f"\n✅ WORKING MODELS ({len(working_models)}/{len(ALL_MODELS)}):")
    if working_models:
        for model in working_models:
            logger.info(f"  🟢 {model}")
    else:
        logger.info("  (None)")
    
    logger.info(f"\n❌ FAILED MODELS ({len(failed_models)}/{len(ALL_MODELS)}):")
    if failed_models:
        for model in failed_models:
            error = results[model]["error"]
            logger.info(f"  🔴 {model}: {error}")
    else:
        logger.info("  (None)")
    
    # สถิติ
    success_rate = len(working_models) / len(ALL_MODELS) * 100
    logger.info(f"\n📈 SUCCESS RATE: {success_rate:.1f}%")
    
    if success_rate >= 50:
        logger.info("✅ Good! More than half of the models are working.")
        logger.info("📝 You can proceed with the real-world test using working models.")
    else:
        logger.warning("⚠️ Warning: Less than half of the models are working.")
        logger.warning("🔧 You may need to check model configurations.")
    
    # บันทึกรายงาน
    import json
    report = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "test_image": test_image_file,
        "total_models": len(ALL_MODELS),
        "working_models": working_models,
        "failed_models": failed_models,
        "success_rate": success_rate,
        "detailed_results": results
    }
    
    output_file = "dynamic_model_test_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n📄 Detailed report saved: {output_file}")
    
    # แนะนำขั้นตอนถัดไป
    if working_models:
        logger.info(f"\n🎯 NEXT STEPS:")
        logger.info(f"1. Update real_world_face_recognition_test_fixed.py")
        logger.info(f"2. Use only working models: {working_models}")
        logger.info(f"3. Remove failed models from ALL_MODELS list")

if __name__ == "__main__":
    try:
        run_dynamic_model_test()
    except Exception as e:
        logger.error(f"🛑 Test terminated: {e}")
        print(f"\n🛑 TEST TERMINATED DUE TO ERROR: {e}")
        exit(1)
