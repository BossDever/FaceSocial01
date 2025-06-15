#!/usr/bin/env python3
"""Test framework features directly"""

import requests
import json

# Configuration
BASE_URL = "http://localhost:8080"

def test_framework_registration():
    """Test framework model registration"""
    print("Testing framework model (deepface) registration...")
    
    # Simple 1x1 base64 image for testing
    test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    payload = {
        "person_id": "test_framework", 
        "person_name": "Test Framework",
        "face_image_base64": test_image_b64,
        "model_name": "deepface"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/face-recognition/add-face-json",
            json=payload,
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:500]}...")
        
        if response.status_code == 200:
            result = response.json()
            return result.get('success', False)
        else:
            return False
            
    except Exception as e:
        print(f"Exception: {e}")
        return False

def test_available_frameworks():
    """Test if frameworks are available in the service"""
    print("\nTesting service framework availability...")
    
    try:
        # Try to call the available models endpoint
        response = requests.get(f"{BASE_URL}/api/face-recognition/models/available", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Available models: {result.get('available_models', [])}")
            print(f"Multi-framework enabled: {result.get('multi_framework_enabled', False)}")
            
            # Check if any framework models are listed
            framework_models = result.get('framework_models', [])
            print(f"Framework models supported: {framework_models}")
            return len(framework_models) > 0
        else:
            print(f"Failed to get models: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Exception: {e}")
        return False

def test_edgeface_model():
    """Test EdgeFace (should work as it's placeholder)"""
    print("\nTesting EdgeFace model (placeholder)...")
    
    test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    payload = {
        "person_id": "test_edgeface",
        "person_name": "Test EdgeFace", 
        "face_image_base64": test_image_b64,
        "model_name": "edgeface"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/face-recognition/add-face-json",
            json=payload,
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result.get('success', False)}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Exception: {e}")
        return False

if __name__ == "__main__":
    print("Framework Features Test")
    print("=" * 40)
    
    # Test available frameworks
    frameworks_available = test_available_frameworks()
    
    # Test framework registration
    deepface_ok = test_framework_registration()
    
    # Test EdgeFace (placeholder)
    edgeface_ok = test_edgeface_model()
    
    print("\n" + "=" * 40)
    print(f"Frameworks available: {'✅' if frameworks_available else '❌'}")
    print(f"DeepFace test: {'✅' if deepface_ok else '❌'}")
    print(f"EdgeFace test: {'✅' if edgeface_ok else '❌'}")
    print("Test completed!")
