#!/usr/bin/env python3
"""Simple test to check if framework models are available"""

import requests
import json

# Configuration
BASE_URL = "http://localhost:8080"

def test_available_models():
    """Test getting available models"""
    print("Testing available models endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/face-recognition/models/available", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Available models: {result.get('available_models', [])}")
            print(f"✅ ONNX models: {result.get('onnx_models', [])}")
            print(f"✅ Framework models: {result.get('framework_models', [])}")
            print(f"✅ Multi-framework enabled: {result.get('multi_framework_enabled', False)}")
            return True
        else:
            print(f"❌ Status: {response.status_code}")
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def test_onnx_model():
    """Test ONNX model registration"""
    print("\nTesting ONNX model (facenet)...")
    
    payload = {
        "person_id": "test_onnx",
        "person_name": "Test ONNX",
        "face_image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
        "model_name": "facenet"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/face-recognition/add-face-json",
            json=payload,
            timeout=15
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result.get('success', False)}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    print("Simple Face Recognition Framework Test")
    print("=" * 50)
    
    # Test available models
    models_ok = test_available_models()
    
    # Test ONNX model
    onnx_ok = test_onnx_model()
    
    print("\n" + "=" * 50)
    print(f"Models endpoint: {'✅' if models_ok else '❌'}")
    print(f"ONNX test: {'✅' if onnx_ok else '❌'}")
    print("Test completed!")
