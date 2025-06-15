#!/usr/bin/env python3
"""
Test Framework Models Only - ทดสอบเฉพาะ Framework models
แก้ปัญหา timeout และหาวิธีใช้งาน Framework models
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

def test_framework_recognition(model_name, timeout=120):
    """ทดสอบ Framework model เฉพาะ Recognition"""
    logger.info(f"\n🧠 Testing {model_name.upper()}")
    
    try:
        files = {
            'image': open(TEST_IMAGE, 'rb')
        }
        
        data = {
            'model_name': model_name
        }
        
        logger.info(f"   📤 Sending request (timeout: {timeout}s)...")
        
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
        logger.info(f"   📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"   ✅ SUCCESS")
            
            # แสดงเฉพาะสิ่งสำคัญ
            if result.get("success") and result.get("matches"):
                matches = result["matches"]
                if matches:
                    best_match = matches[0]
                    name = best_match.get("name", "unknown")
                    confidence = best_match.get("confidence", 0)
                    logger.info(f"   🎯 Best: {name} ({confidence:.3f})")
                    logger.info(f"   📊 Matches: {len(matches)}")
                else:
                    logger.info(f"   📊 No matches")
            else:
                logger.info(f"   📊 No results")
            
            return {
                'success': True,
                'processing_time': processing_time,
                'result': result,
                'status_code': response.status_code
            }
        else:
            error_text = response.text[:200]  # แสดงแค่ 200 ตัวอักษรแรก
            logger.error(f"   ❌ FAILED: {response.status_code}")
            logger.error(f"   📝 Error: {error_text}...")
            
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
        logger.error(f"   💥 EXCEPTION: {str(e)[:100]}...")
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

def test_multiple_timeouts():
    """ทดสอบด้วย timeout ต่างๆ"""
    logger.info("🕒 Testing Multiple Timeouts")
    logger.info("=" * 50)
    
    timeouts = [30, 60, 120]
    results = {}
    
    for timeout in timeouts:
        logger.info(f"\n📊 Testing with {timeout}s timeout:")
        logger.info("-" * 30)
        
        timeout_results = {}
        
        for model in FRAMEWORK_MODELS:
            result = test_framework_recognition(model, timeout)
            timeout_results[model] = result
            time.sleep(2)  # พักเล็กน้อย
        
        results[f"{timeout}s"] = timeout_results
        
        # สรุปผล
        success_count = sum(1 for r in timeout_results.values() if r['success'])
        logger.info(f"\n📈 {timeout}s Summary: {success_count}/{len(FRAMEWORK_MODELS)} succeeded")
    
    return results

def main():
    logger.info("🔧 Framework Models Test - Clean Output")
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
        logger.error(f"❌ Cannot connect: {str(e)[:50]}...")
        return
    
    # Check test image
    if not Path(TEST_IMAGE).exists():
        logger.error(f"❌ Test image not found: {TEST_IMAGE}")
        return
    
    logger.info(f"📷 Test image: {TEST_IMAGE}")
    
    # Run tests
    results = test_multiple_timeouts()
    
    # Final summary
    logger.info("\n" + "="*50)
    logger.info("📊 FINAL SUMMARY")
    logger.info("="*50)
    
    best_timeout = None
    best_count = 0
    
    for timeout_label, timeout_data in results.items():
        success_models = [model for model, result in timeout_data.items() if result['success']]
        success_count = len(success_models)
        
        logger.info(f"\n⏰ {timeout_label}:")
        logger.info(f"   ✅ Working: {success_count}/{len(FRAMEWORK_MODELS)}")
        if success_models:
            logger.info(f"   🟢 Models: {success_models}")
            avg_time = sum(timeout_data[model]['processing_time'] for model in success_models) / success_count
            logger.info(f"   ⏱️ Avg time: {avg_time:.1f}s")
        
        if success_count > best_count:
            best_count = success_count
            best_timeout = timeout_label
    
    # Recommendations
    logger.info(f"\n🎯 RECOMMENDATIONS:")
    if best_count > 0:
        logger.info(f"   1. ✅ Use {best_timeout} timeout")
        logger.info(f"   2. ✅ {best_count}/{len(FRAMEWORK_MODELS)} models work")
        logger.info(f"   3. 🔧 Update main script with longer timeout")
    else:
        logger.info(f"   1. ❌ Framework models don't work")
        logger.info(f"   2. ✅ Use ONNX models only")
    
    # Save compact report
    compact_report = {}
    for timeout_label, timeout_data in results.items():
        compact_report[timeout_label] = {
            model: {
                'success': result['success'],
                'time': result['processing_time'],
                'error': result.get('error', 'none')[:50] if result.get('error') else 'none'
            }
            for model, result in timeout_data.items()
        }
    
    with open('framework_test_results.json', 'w') as f:
        json.dump(compact_report, f, indent=2)
    
    logger.info(f"\n📄 Report saved: framework_test_results.json")

if __name__ == "__main__":
    main()
