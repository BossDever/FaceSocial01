#!/usr/bin/env python3
"""
Test framework model registration to identify specific issues
"""

import requests
import json
import logging
from pathlib import Path
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8080"
TEST_IMAGE = "test_images/boss_01.jpg"

# Framework models that are failing
FRAMEWORK_MODELS = [
    "deepface",
    "facenet_pytorch", 
    "dlib",
    "insightface",
    "edgeface"
]

def test_registration_detailed(model_name, image_path):
    """Test registration with detailed error reporting"""
    logger.info(f"\n🧪 Testing {model_name.upper()} registration...")
    
    try:
        # Prepare files
        files = {
            'image': open(image_path, 'rb')
        }
        
        # Prepare data
        data = {
            'user_id': f'test_user_{model_name}',
            'model_name': model_name
        }
        
        logger.info(f"   📤 Sending request to /api/face-recognition/add-face-json")
        logger.info(f"   👤 User ID: {data['user_id']}")
        logger.info(f"   🧠 Model: {model_name}")
        
        # Make request
        response = requests.post(
            f"{BASE_URL}/api/face-recognition/add-face-json",
            files=files,
            data=data,
            timeout=30
        )
        
        logger.info(f"   📊 Response Status: {response.status_code}")
        logger.info(f"   📋 Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"   ✅ SUCCESS: {json.dumps(result, indent=2)}")
            return True, result
        else:
            error_text = response.text
            logger.error(f"   ❌ FAILED: {response.status_code}")
            logger.error(f"   📝 Error Response: {error_text}")
            
            # Try to parse JSON error
            try:
                error_json = response.json()
                logger.error(f"   🔍 Detailed Error: {json.dumps(error_json, indent=2)}")
            except:
                logger.error(f"   🔍 Raw Error Text: {error_text}")
                
            return False, error_text
            
    except Exception as e:
        logger.error(f"   💥 EXCEPTION: {str(e)}")
        return False, str(e)
    finally:
        try:
            files['image'].close()
        except:
            pass

def test_model_loading(model_name):
    """Test if model can be loaded via API"""
    logger.info(f"\n🔄 Testing {model_name.upper()} model loading...")
    
    try:
        # Test with a simple recognition request
        files = {
            'image': open(TEST_IMAGE, 'rb')
        }
        
        data = {
            'model_name': model_name
        }
        
        response = requests.post(
            f"{BASE_URL}/api/face-recognition/recognize",
            files=files,
            data=data,
            timeout=30
        )
        
        logger.info(f"   📊 Recognition Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"   ✅ Model loads successfully for recognition")
            logger.info(f"   📋 Recognition result: {json.dumps(result, indent=2)}")
            return True
        else:
            logger.error(f"   ❌ Model loading failed: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"   💥 Exception testing model loading: {str(e)}")
        return False
    finally:
        try:
            files['image'].close()
        except:
            pass

def main():
    logger.info("🔬 Framework Model Registration Debug Test")
    logger.info("=" * 60)
      # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            status = response.json()
            logger.info(f"✅ Server is running: {json.dumps(status, indent=2)}")
        else:
            logger.error(f"❌ Server health check failed: {response.status_code}")
            return
    except Exception as e:
        logger.error(f"❌ Cannot connect to server: {e}")
        return
    
    # Check test image exists
    if not Path(TEST_IMAGE).exists():
        logger.error(f"❌ Test image not found: {TEST_IMAGE}")
        return
    
    logger.info(f"📷 Using test image: {TEST_IMAGE}")
    
    # Test each framework model
    results = {}
    
    for model in FRAMEWORK_MODELS:
        logger.info(f"\n" + "="*60)
        logger.info(f"🧠 TESTING MODEL: {model.upper()}")
        logger.info(f"=" * 60)
        
        # Test 1: Model loading capability
        can_load = test_model_loading(model)
        
        # Test 2: Registration capability  
        can_register, reg_result = test_registration_detailed(model, TEST_IMAGE)
        
        results[model] = {
            'can_load': can_load,
            'can_register': can_register,
            'registration_result': reg_result
        }
        
        logger.info(f"\n📊 {model.upper()} SUMMARY:")
        logger.info(f"   🔄 Can load for recognition: {'✅ YES' if can_load else '❌ NO'}")
        logger.info(f"   📝 Can register faces: {'✅ YES' if can_register else '❌ NO'}")
        
        # Wait between tests
        time.sleep(2)
    
    # Final summary
    logger.info(f"\n" + "="*60)
    logger.info("📊 FINAL SUMMARY")
    logger.info("="*60)
    
    working_for_recognition = []
    working_for_registration = []
    
    for model, result in results.items():
        if result['can_load']:
            working_for_recognition.append(model)
        if result['can_register']:
            working_for_registration.append(model)
    
    logger.info(f"\n✅ MODELS WORKING FOR RECOGNITION ({len(working_for_recognition)}):")
    for model in working_for_recognition:
        logger.info(f"   🟢 {model}")
    
    logger.info(f"\n✅ MODELS WORKING FOR REGISTRATION ({len(working_for_registration)}):")
    for model in working_for_registration:
        logger.info(f"   🟢 {model}")
    
    logger.info(f"\n❌ MODELS FAILING REGISTRATION ({len(FRAMEWORK_MODELS) - len(working_for_registration)}):")
    for model in FRAMEWORK_MODELS:
        if not results[model]['can_register']:
            logger.info(f"   🔴 {model}")
    
    # Save detailed results
    report_file = "framework_registration_debug_report.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n📄 Detailed report saved: {report_file}")
    
    # Recommendations
    logger.info(f"\n🎯 RECOMMENDATIONS:")
    if len(working_for_registration) == 0:
        logger.info("   1. ❌ No framework models work for registration")
        logger.info("   2. ✅ Use ONNX models (facenet, adaface, arcface) for registration")
        logger.info("   3. ✅ Framework models can still be used for recognition")
        logger.info("   4. 🔧 Update script to separate registration and recognition models")
    else:
        logger.info("   1. ✅ Some framework models work for registration")
        logger.info("   2. 🔧 Use working models for both registration and recognition")

if __name__ == "__main__":
    main()
