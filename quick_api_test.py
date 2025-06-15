#!/usr/bin/env python3
"""
Quick API Test - Basic functionality test
Created: June 14, 2025
Purpose: Quick test of main API endpoints
"""

import json
import time
import base64
import requests
from datetime import datetime
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8080"
TEST_IMAGES_DIR = Path("test_images")

def encode_image(image_path):
    """Convert image to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def test_endpoint(method, endpoint, **kwargs):
    """Test an API endpoint"""
    url = f"{API_BASE_URL}{endpoint}"
    start_time = time.time()
    
    # Set timeout - None means no timeout limit
    timeout = None if "add-face" in endpoint else 60
    
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
        duration = time.time() - start_time
        
        print(f"  {method} {endpoint}")
        print(f"    Status: {response.status_code}")
        print(f"    Time: {duration:.3f}s")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"    Response: {type(data).__name__}")
                if isinstance(data, dict):
                    for key, value in list(data.items())[:3]:  # Show first 3 keys
                        if isinstance(value, (str, int, float, bool)):
                            print(f"      {key}: {value}")
                        else:
                            print(f"      {key}: {type(value).__name__}")
                print("    ✅ SUCCESS")
            except:
                print(f"    Response: {response.text[:100]}...")
                print("    ✅ SUCCESS")
        else:
            print(f"    Error: {response.text[:100]}")
            print("    ❌ FAILED")
        
        print()
        return response.status_code == 200
        
    except Exception as e:
        print(f"  {method} {endpoint}")
        print(f"    Error: {str(e)}")
        print("    ❌ FAILED")
        print()
        return False

def main():
    """Run quick API tests"""
    print("🚀 Face Recognition System - Quick API Test")
    print("=" * 50)
    print(f"Testing API at: {API_BASE_URL}")
    print(f"Test images: {TEST_IMAGES_DIR}")
    print()
    
    # Test basic endpoints
    print("1. Testing System Health...")
    test_endpoint("GET", "/health")
    
    print("2. Testing Service Health...")
    test_endpoint("GET", "/api/face-detection/health")
    test_endpoint("GET", "/api/face-recognition/health")
    test_endpoint("GET", "/api/face-analysis/health")
    
    print("3. Testing Model Information...")
    test_endpoint("GET", "/api/face-detection/models/available")
    test_endpoint("GET", "/api/face-recognition/models/available")
    test_endpoint("GET", "/api/face-recognition/database-status")
    
    # Test with image if available
    test_image = TEST_IMAGES_DIR / "boss_01.jpg"
    if test_image.exists():
        print("4. Testing Face Detection...")
          # Test file upload
        try:
            with open(test_image, "rb") as f:
                files = {"file": f}
                data = {"model_name": "auto", "conf_threshold": 0.5}
                test_endpoint("POST", "/api/face-detection/detect", files=files, data=data)
        except Exception as e:
            print(f"  File upload test failed: {e}")
        
        # Test base64
        try:
            image_b64 = encode_image(test_image)
            data = {
                "image_base64": image_b64,
                "model_name": "auto",
                "conf_threshold": 0.5
            }
            test_endpoint("POST", "/api/face-detection/detect-base64", json=data)
        except Exception as e:
            print(f"  Base64 test failed: {e}")
        
        print("5. Testing Face Recognition...")
        
        # Add a face
        try:
            data = {
                "person_id": "test_user",
                "person_name": "Test User",
                "face_image_base64": image_b64,
                "model_name": "facenet"
            }
            test_endpoint("POST", "/api/face-recognition/add-face-json", json=data)
        except Exception as e:
            print(f"  Add face test failed: {e}")
        
        # Get gallery
        test_endpoint("GET", "/api/face-recognition/get-gallery")
        
        print("6. Testing Face Analysis...")
        
        # Analyze image
        try:
            data = {
                "image_base64": image_b64,
                "mode": "full_analysis",
                "config": {
                    "detection_model": "auto",
                    "recognition_model": "facenet"
                }
            }
            test_endpoint("POST", "/api/face-analysis/analyze-json", json=data)
        except Exception as e:
            print(f"  Analysis test failed: {e}")
    
    else:
        print("4. Skipping image tests - no test image found")
    
    print("=" * 50)
    print("Quick test completed!")
    print("For comprehensive testing, run: python simple_api_tester.py")

if __name__ == "__main__":
    main()
