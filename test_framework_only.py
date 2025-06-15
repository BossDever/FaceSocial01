#!/usr/bin/env python3
"""
Test Framework Models Only - ทดสอบเฉพาะ Framework models
เพื่อแก้ปัญหา timeout และหาวิธีใช้งาน Framework models ให้ได้
"""

import requests
import json
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8080"
TEST_IMAGE = "test_images/boss_01.jpg"

# Framework models ที่มีปัญหา
FRAMEWORK_MODELS = [
    "deepface",
    "facenet_pytorch", 
    "dlib",
    "insightface",
    "edgeface"
]

def test_framework_recognition_only(model_name, timeout=120):
    """ทดสอบ Framework model เฉพาะ Recognition (ไม่ลงทะเบียน)"""
    logger.info(f"\n🧠 Testing {model_name.upper()} - Recognition Only")
    
    try:
        files = {
            'image': open(TEST_IMAGE, 'rb')
        }
        
        data = {
            'model_name': model_name
        }
        
        logger.info(f"   📤 Sending recognition request...")
        logger.info(f"   ⏰ Timeout: {timeout}s")
        
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/api/face-recognition/recognize",
            files=files,
            data=data,
            timeout=timeout
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
          logger.info(f"   ⏱️ Processing time: {processing_time:.2f}s")
        logger.info(f"   📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"   ✅ SUCCESS")
            
            # แสดงเฉพาะข้อมูลสำคัญ ไม่แสดง JSON เต็ม
            if result.get("success") and result.get("matches"):
                matches = result["matches"]
                if matches:
                    best_match = matches[0]
                    name = best_match.get("name", "unknown")
                    confidence = best_match.get("confidence", 0)
                    logger.info(f"   🎯 Best match: {name} (confidence: {confidence:.3f})")
                    logger.info(f"   📊 Total matches: {len(matches)}")
                else:
                    logger.info(f"   📊 No matches found")
            else:
                logger.info(f"   📊 No recognition results")
            
            return {
                'success': True,
                'processing_time': processing_time,
                'result': result,
                'status_code': response.status_code
            }
        else:
            error_text = response.text
            logger.error(f"   ❌ FAILED: {response.status_code}")
            logger.error(f"   📝 Error: {error_text}")
            
            return {
                'success': False,
                'processing_time': processing_time,
                'error': error_text,
                'status_code': response.status_code
            }
            
    except requests.exceptions.Timeout:
        logger.error(f"   ⏰ TIMEOUT after {timeout}s")
        return {
            'success': False,
            'processing_time': timeout,
            'error': 'Timeout',
            'status_code': None
        }
    except Exception as e:
        logger.error(f"   💥 EXCEPTION: {str(e)}")
        return {
            'success': False,
            'processing_time': 0,
            'error': str(e),
            'status_code': None
        }
    finally:
        try:
            files['image'].close()
        except:
            pass

def test_different_timeouts():
    """ทดสอบ Framework models ด้วย timeout ต่างๆ"""
    logger.info("🕒 Testing Framework Models with Different Timeouts")
    logger.info("=" * 60)
    
    timeouts = [30, 60, 120, 180]  # ทดสอบ timeout ต่างๆ
    results = {}
    
    for timeout in timeouts:
        logger.info(f"\n📊 Testing with {timeout}s timeout:")
        logger.info("-" * 40)
        
        timeout_results = {}
        
        for model in FRAMEWORK_MODELS:
            result = test_framework_recognition_only(model, timeout)
            timeout_results[model] = result
            
            # พักระหว่างโมเดล
            time.sleep(3)
        
        results[f"{timeout}s"] = timeout_results
        
        # สรุปผลสำหรับ timeout นี้
        success_count = sum(1 for r in timeout_results.values() if r['success'])
        logger.info(f"\n📈 Timeout {timeout}s Summary: {success_count}/{len(FRAMEWORK_MODELS)} models succeeded")
    
    return results

def test_progressive_loading():
    """ทดสอบการโหลด Framework models ทีละตัว"""
    logger.info("\n🔄 Progressive Loading Test")
    logger.info("=" * 60)
    
    results = {}
    
    for i, model in enumerate(FRAMEWORK_MODELS, 1):
        logger.info(f"\n[{i}/{len(FRAMEWORK_MODELS)}] Testing {model}")
        logger.info("-" * 40)
        
        # ทดสอบหลายครั้งเพื่อดูว่าโหลดครั้งแรกช้าหรือไม่
        model_results = []
        
        for attempt in range(3):
            logger.info(f"   Attempt {attempt + 1}/3:")
            result = test_framework_recognition_only(model, 180)
            model_results.append(result)
            
            if result['success']:
                logger.info(f"   ✅ Attempt {attempt + 1}: SUCCESS in {result['processing_time']:.2f}s")
            else:
                logger.info(f"   ❌ Attempt {attempt + 1}: FAILED")
            
            time.sleep(5)  # พักระหว่าง attempt
        
        results[model] = model_results
        
        # สรุปผลสำหรับโมเดลนี้
        success_attempts = sum(1 for r in model_results if r['success'])
        avg_time = sum(r['processing_time'] for r in model_results if r['success']) / max(success_attempts, 1)
        
        logger.info(f"\n📊 {model} Summary:")
        logger.info(f"   Success rate: {success_attempts}/3")
        logger.info(f"   Average time: {avg_time:.2f}s")
        
        # พักระหว่างโมเดล
        time.sleep(10)
    
    return results

def main():
    logger.info("🔧 Framework Models Troubleshooting")
    logger.info("=" * 60)
    
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
    
    logger.info(f"📷 Using test image: {TEST_IMAGE}")
    
    # Test 1: Different timeouts
    logger.info("\n" + "="*60)
    logger.info("🧪 TEST 1: Different Timeouts")
    logger.info("="*60)
    
    timeout_results = test_different_timeouts()
    
    # Test 2: Progressive loading
    logger.info("\n" + "="*60)
    logger.info("🧪 TEST 2: Progressive Loading")
    logger.info("="*60)
    
    progressive_results = test_progressive_loading()
    
    # Final analysis
    logger.info("\n" + "="*60)
    logger.info("📊 FINAL ANALYSIS")
    logger.info("="*60)
    
    # แยกวิเคราะห์ผลแต่ละ timeout
    for timeout_label, timeout_data in timeout_results.items():
        success_models = [model for model, result in timeout_data.items() if result['success']]
        working_count = len(success_models)
        
        logger.info(f"\n⏰ {timeout_label} Timeout:")
        logger.info(f"   ✅ Working models ({working_count}): {success_models}")
        
        if working_count > 0:
            avg_time = sum(timeout_data[model]['processing_time'] for model in success_models) / working_count
            logger.info(f"   ⏱️ Average processing time: {avg_time:.2f}s")
    
    # แนะนำการแก้ไข
    logger.info(f"\n🎯 RECOMMENDATIONS:")
    
    # หา timeout ที่ดีที่สุด
    best_timeout = None
    best_success_count = 0
    
    for timeout_label, timeout_data in timeout_results.items():
        success_count = sum(1 for result in timeout_data.values() if result['success'])
        if success_count > best_success_count:
            best_success_count = success_count
            best_timeout = timeout_label
    
    if best_success_count > 0:
        logger.info(f"   1. ✅ Use {best_timeout} timeout for framework models")
        logger.info(f"   2. ✅ {best_success_count}/{len(FRAMEWORK_MODELS)} framework models work")
        logger.info(f"   3. 🔧 Update real_world script with longer timeouts")
    else:
        logger.info(f"   1. ❌ Framework models may not be supported")
        logger.info(f"   2. ✅ Stick with ONNX models only")
        logger.info(f"   3. 🔧 Remove framework models from real_world script")
    
    # Save results
    report = {
        'timeout_tests': timeout_results,
        'progressive_tests': progressive_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('framework_troubleshooting_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"\n📄 Detailed report saved: framework_troubleshooting_report.json")

if __name__ == "__main__":
    main()
