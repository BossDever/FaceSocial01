#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Framework Integration Test
Test all available models (ONNX + Framework) systematically
"""

import requests
import base64
import time
from pathlib import Path

def encode_image_to_base64(image_path):
    """Encode image file to base64 string"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string

def test_model_registration_recognition(model_name, base_url="http://localhost:8080"):
    """Test registration and recognition for a specific model"""
    print(f"\n{'='*50}")
    print(f"🧪 Testing Model: {model_name}")
    print(f"{'='*50}")
    
    # Test images
    test_image_1 = "test_images/boss_01.jpg"
    test_image_2 = "test_images/boss_02.jpg"
    
    if not Path(test_image_1).exists() or not Path(test_image_2).exists():
        print(f"❌ Test images not found: {test_image_1}, {test_image_2}")
        return False
    
    try:
        # Clear database first
        print("🧹 Clearing database...")
        clear_response = requests.delete(f"{base_url}/api/face-recognition/gallery/clear")
        if clear_response.status_code == 200:
            print("✅ Database cleared successfully")
        else:
            print(f"⚠️ Database clear warning: {clear_response.status_code}")
        
        # Encode images
        image_1_b64 = encode_image_to_base64(test_image_1)
        image_2_b64 = encode_image_to_base64(test_image_2)
        
        # 1. Register face with model
        print(f"📝 Registering face with model: {model_name}")
        register_data = {
            "person_name": f"TestPerson_{model_name}",
            "person_id": f"test_id_{model_name}",
            "face_image_base64": image_1_b64,
            "model_name": model_name        }
        
        register_response = requests.post(
            f"{base_url}/api/face-recognition/add-face-json",
            json=register_data
        )
        
        if register_response.status_code == 200:
            register_result = register_response.json()
            print(f"✅ Registration successful")
            print(f"   Face ID: {register_result.get('face_id', 'N/A')}")
            processing_time = register_result.get('processing_time', 0)
            if isinstance(processing_time, (int, float)):
                print(f"   Processing time: {processing_time:.3f}s")
        else:
            print(f"❌ Registration failed: {register_response.status_code}")
            # Limit error message to avoid base64 spam
            error_text = register_response.text
            if len(error_text) > 200:
                error_text = error_text[:200] + "... [truncated]"
            print(f"   Error: {error_text}")
            return False
        
        # 2. Test recognition with same image
        print(f"🔍 Testing recognition (same image) with model: {model_name}")
        recognize_data = {
            "face_image_base64": image_1_b64,
            "model_name": model_name,
            "top_k": 3,
            "similarity_threshold": 0.4
        }        
        recognize_response = requests.post(
            f"{base_url}/api/face-recognition/recognize",
            json=recognize_data
        )
        
        if recognize_response.status_code == 200:
            recognize_result = recognize_response.json()
            matches = recognize_result.get('matches', recognize_result.get('results', []))
            if matches:
                best_match = matches[0]
                print(f"✅ Recognition successful")
                print(f"   Matched: {best_match.get('person_name', 'N/A')}")
                print(f"   Confidence: {best_match.get('confidence', best_match.get('similarity', 0)):.3f}")
                print(f"   Processing time: {recognize_result.get('processing_time', 0):.3f}s")
            else:
                print(f"⚠️ Recognition returned no matches")
                return False
        else:
            print(f"❌ Recognition failed: {recognize_response.status_code}")
            # Limit error message to avoid base64 spam
            error_text = recognize_response.text
            if len(error_text) > 200:
                error_text = error_text[:200] + "... [truncated]"
            print(f"   Error: {error_text}")
            return False
        
        # 3. Test recognition with different image of same person
        print(f"🔍 Testing recognition (different image) with model: {model_name}")
        recognize_data_2 = {
            "face_image_base64": image_2_b64,
            "model_name": model_name,
            "top_k": 3,
            "similarity_threshold": 0.4
        }        
        recognize_response_2 = requests.post(
            f"{base_url}/api/face-recognition/recognize",
            json=recognize_data_2
        )
        
        if recognize_response_2.status_code == 200:
            recognize_result_2 = recognize_response_2.json()
            matches_2 = recognize_result_2.get('matches', recognize_result_2.get('results', []))
            if matches_2:
                best_match_2 = matches_2[0]
                print(f"✅ Cross-recognition successful")
                print(f"   Matched: {best_match_2.get('person_name', 'N/A')}")
                print(f"   Confidence: {best_match_2.get('confidence', best_match_2.get('similarity', 0)):.3f}")
                print(f"   Processing time: {recognize_result_2.get('processing_time', 0):.3f}s")
            else:
                print(f"⚠️ Cross-recognition returned no matches")
        else:
            print(f"❌ Cross-recognition failed: {recognize_response_2.status_code}")
            # Limit error message to avoid base64 spam
            error_text = recognize_response_2.text
            if len(error_text) > 200:
                error_text = error_text[:200] + "... [truncated]"
            print(f"   Error: {error_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Final Framework Integration Test")
    print("=" * 70)
    
    base_url = "http://localhost:8080"
    
    # Get available models
    print("📋 Fetching available models...")
    try:
        models_response = requests.get(f"{base_url}/api/face-recognition/models/available")
        if models_response.status_code == 200:
            models_data = models_response.json()
            available_frameworks = models_data.get('available_frameworks', [])
            onnx_models = models_data.get('onnx_models', [])
            framework_models = models_data.get('framework_models', [])
            
            print(f"✅ Found {len(available_frameworks)} total models:")
            print(f"   ONNX Models ({len(onnx_models)}): {onnx_models}")
            print(f"   Framework Models ({len(framework_models)}): {framework_models}")
            print(f"   Multi-framework enabled: {models_data.get('multi_framework_enabled', False)}")
        else:
            print(f"❌ Failed to get available models: {models_response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error fetching models: {e}")
        return
    
    # Test results
    results = {}
    
    # Test all models systematically
    for model_name in available_frameworks:
        start_time = time.time()
        success = test_model_registration_recognition(model_name)
        end_time = time.time()
        
        results[model_name] = {
            'success': success,
            'total_time': end_time - start_time
        }
        
        # Wait between tests to avoid overloading
        time.sleep(2)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 70)
    
    successful_models = []
    failed_models = []
    
    for model_name, result in results.items():
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        time_str = f"{result['total_time']:.2f}s"
        print(f"{model_name:<20} | {status:<10} | Time: {time_str}")
        
        if result['success']:
            successful_models.append(model_name)
        else:
            failed_models.append(model_name)
    
    print("\n" + "-" * 70)
    print(f"📈 SUCCESS RATE: {len(successful_models)}/{len(results)} models ({len(successful_models)/len(results)*100:.1f}%)")
    
    if successful_models:
        print(f"✅ WORKING MODELS ({len(successful_models)}): {', '.join(successful_models)}")
    
    if failed_models:
        print(f"❌ FAILED MODELS ({len(failed_models)}): {', '.join(failed_models)}")
    
    print("\n🎯 Framework Integration Test Complete!")

if __name__ == "__main__":
    main()
