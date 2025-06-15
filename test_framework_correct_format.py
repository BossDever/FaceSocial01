#!/usr/bin/env python3
"""Test framework models with correct API format"""

import requests
import json
import base64
import time
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
TEST_IMAGE = "test_images/boss_01.jpg"

# Framework models to test
FRAMEWORK_MODELS = ['deepface', 'facenet_pytorch', 'dlib', 'insightface', 'edgeface']

def encode_image_to_base64(image_path):
    """Encode image to base64 string"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def test_registration_correct_format():
    """Test registration with correct API format"""
    print("Testing framework model registration with correct format...")
    
    # Encode test image
    image_base64 = encode_image_to_base64(TEST_IMAGE)
    
    for model in FRAMEWORK_MODELS:
        print(f"\n--- Testing {model} registration ---")
        
        # Test JSON endpoint (add-face-json)
        payload = {
            "person_id": f"test_person_{model}",
            "person_name": f"Test Person {model}",
            "face_image_base64": image_base64,
            "model_name": model
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
                print(f"Success: {result}")
            else:
                print(f"Error: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"Timeout for {model}")
        except Exception as e:
            print(f"Exception for {model}: {e}")
        
        time.sleep(1)  # Brief pause between requests

def test_recognition_correct_format():
    """Test recognition with correct API format"""
    print("\n\nTesting framework model recognition with correct format...")
    
    # First get the gallery
    try:
        gallery_response = requests.get(f"{BASE_URL}/api/face-recognition/get-gallery")
        if gallery_response.status_code == 200:
            gallery = gallery_response.json()
            print(f"Gallery contains {len(gallery)} faces")
        else:
            print("Failed to get gallery")
            return
    except Exception as e:
        print(f"Error getting gallery: {e}")
        return
    
    # Encode test image
    image_base64 = encode_image_to_base64(TEST_IMAGE)
    
    for model in FRAMEWORK_MODELS:
        print(f"\n--- Testing {model} recognition ---")
        
        payload = {
            "face_image_base64": image_base64,
            "gallery": gallery,
            "model_name": model,
            "top_k": 3
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/face-recognition/recognize",
                json=payload,
                timeout=30
            )
            
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Recognition results: {result}")
            else:
                print(f"Error: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"Timeout for {model}")
        except Exception as e:
            print(f"Exception for {model}: {e}")
        
        time.sleep(1)  # Brief pause between requests

if __name__ == "__main__":
    print("Testing framework models with correct API format")
    print("=" * 60)
    
    # Test registration first
    test_registration_correct_format()
    
    # Test recognition
    test_recognition_correct_format()
    
    print("\n" + "=" * 60)
    print("Test completed!")
