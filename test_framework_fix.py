#!/usr/bin/env python3
"""
Fix Framework Models - แก้ไขปัญหา Framework models
ทดสอบ request format ที่ถูกต้อง
"""

import requests
import json
import logging
import time
import base64
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8080"
TEST_IMAGE = "test_images/boss_01.jpg"

# Framework models
FRAMEWORK_MODELS = [
    "deepface",
    "facenet_pytorch", 
    "dlib",
    "insightface",
    "edgeface"
]

def image_to_base64(image_path):
    """แปลงรูปเป็น base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def test_json_format(model_name):
    """ทดสอบด้วย JSON format (แทนที่จะเป็น multipart/form-data)"""
    logger.info(f"\n🧠 Testing {model_name.upper()} - JSON Format")
    
    try:
        # แปลงรูปเป็น base64
        image_base64 = image_to_base64(TEST_IMAGE)
        
        # ใช้ JSON format
        payload = {
            "image": image_base64,
            "model_name": model_name
        }
        
        logger.info(f"   📤 Sending JSON request...")
        logger.info(f"   🧠 Model: {model_name}")
        logger.info(f"   📊 Image size: {len(image_base64)} chars")
        
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/api/face-recognition/recognize",
            json=payload,  # ใช้ json= แทน files= และ data=
            timeout=120
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"   ⏱️ Processing time: {processing_time:.2f}s")
        logger.info(f"   📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"   ✅ SUCCESS!")
            
            # แสดงข้อมูลสำคัญ
            if result.get("success"):
                matches = result.get("matches", [])
                logger.info(f"   📋 Found {len(matches)} matches")
                if matches:
                    best = matches[0]
                    logger.info(f"   🎯 Best: {best.get('name', 'unknown')} ({best.get('confidence', 0):.3f})")
            else:
                logger.info(f"   📋 No matches found")
            
            return {
                'success': True,
                'processing_time': processing_time,
                'result': result
            }
        else:
            error_text = response.text[:200] + "..." if len(response.text) > 200 else response.text
            logger.error(f"   ❌ FAILED: {response.status_code}")
            logger.error(f"   📝 Error: {error_text}")
            
            return {
                'success': False,
                'processing_time': processing_time,
                'error': error_text,
                'status_code': response.status_code
            }
            
    except requests.exceptions.Timeout:
        logger.error(f"   ⏰ TIMEOUT after 120s")
        return {'success': False, 'error': 'Timeout'}
    except Exception as e:
        logger.error(f"   💥 EXCEPTION: {str(e)}")
        return {'success': False, 'error': str(e)}

def test_multipart_format(model_name):
    """ทดสอบด้วย multipart/form-data format (แบบเดิม)"""
    logger.info(f"\n🧠 Testing {model_name.upper()} - Multipart Format")
    
    try:
        files = {
            'image': open(TEST_IMAGE, 'rb')
        }
        
        data = {
            'model_name': model_name
        }
        
        logger.info(f"   📤 Sending multipart request...")
        
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/api/face-recognition/recognize",
            files=files,
            data=data,
            timeout=120
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"   ⏱️ Processing time: {processing_time:.2f}s")
        logger.info(f"   📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"   ✅ SUCCESS!")
            return {
                'success': True,
                'processing_time': processing_time,
                'result': result
            }
        else:
            error_text = response.text[:200] + "..." if len(response.text) > 200 else response.text
            logger.error(f"   ❌ FAILED: {response.status_code}")
            logger.error(f"   📝 Error: {error_text}")
            return {
                'success': False,
                'processing_time': processing_time,
                'error': error_text
            }
            
    except Exception as e:
        logger.error(f"   💥 EXCEPTION: {str(e)}")
        return {'success': False, 'error': str(e)}
    finally:
        try:
            files['image'].close()
        except:
            pass

def main():
    logger.info("🔧 Framework Models Fix Test")
    logger.info("=" * 50)
    
    # Check server
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Server is running")
        else:
            logger.error(f"❌ Server problem: {response.status_code}")
            return
    except Exception as e:
        logger.error(f"❌ Cannot connect to server: {e}")
        return
    
    # Check test image
    if not Path(TEST_IMAGE).exists():
        logger.error(f"❌ Test image not found: {TEST_IMAGE}")
        return
    
    logger.info(f"📷 Test image: {TEST_IMAGE}")
    
    results = {}
    
    for model in FRAMEWORK_MODELS:
        logger.info(f"\n" + "="*50)
        logger.info(f"🧪 TESTING MODEL: {model.upper()}")
        logger.info("="*50)
        
        # Test 1: JSON format
        json_result = test_json_format(model)
        
        # Test 2: Multipart format
        multipart_result = test_multipart_format(model)
        
        results[model] = {
            'json_format': json_result,
            'multipart_format': multipart_result
        }
        
        # สรุปผล
        logger.info(f"\n📊 {model.upper()} SUMMARY:")
        logger.info(f"   JSON format: {'✅ SUCCESS' if json_result['success'] else '❌ FAILED'}")
        logger.info(f"   Multipart format: {'✅ SUCCESS' if multipart_result['success'] else '❌ FAILED'}")
        
        # พักระหว่างโมเดล
        time.sleep(3)
    
    # Final summary
    logger.info(f"\n" + "="*50)
    logger.info("📊 FINAL SUMMARY")
    logger.info("="*50)
    
    json_working = []
    multipart_working = []
    
    for model, result in results.items():
        if result['json_format']['success']:
            json_working.append(model)
        if result['multipart_format']['success']:
            multipart_working.append(model)
    
    logger.info(f"\n✅ JSON FORMAT WORKING ({len(json_working)}):")
    for model in json_working:
        logger.info(f"   🟢 {model}")
    
    logger.info(f"\n✅ MULTIPART FORMAT WORKING ({len(multipart_working)}):")
    for model in multipart_working:
        logger.info(f"   🟢 {model}")
    
    # Recommendations
    logger.info(f"\n🎯 RECOMMENDATIONS:")
    if len(json_working) > 0:
        logger.info(f"   1. ✅ Use JSON format for framework models")
        logger.info(f"   2. 🔧 Update real_world script to use JSON format")
        logger.info(f"   3. ✅ {len(json_working)}/{len(FRAMEWORK_MODELS)} framework models work with JSON")
    elif len(multipart_working) > 0:
        logger.info(f"   1. ✅ Use multipart format for framework models")
        logger.info(f"   2. 🔧 Keep current format in real_world script")
    else:
        logger.info(f"   1. ❌ Framework models don't work with either format")
        logger.info(f"   2. ✅ Continue using ONNX models only")
    
    # Save results
    with open('framework_fix_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n📄 Report saved: framework_fix_results.json")

if __name__ == "__main__":
    main()
