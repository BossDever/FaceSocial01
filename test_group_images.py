#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group Image Recognition Test
ทดสอบการจดจำในภาพกลุ่มเพื่อดูว่าการแก้ไขทำงานถูกต้อง
"""

import os
import json
import time
import base64
import requests
import numpy as np
import cv2
from typing import Dict, List, Any

# Configuration
API_BASE_URL = "http://localhost:8080"
TEST_IMAGES_DIR = "test_images"

def image_to_base64(image_path: str) -> str:
    """แปลงรูปภาพเป็น base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def detect_faces(image_base64: str) -> List[Dict[str, Any]]:
    """ตรวจจับใบหน้าในภาพ"""
    url = f"{API_BASE_URL}/api/face-detection/detect-base64"
    
    data = {
        "image_base64": image_base64,
        "model_name": "yolov11m-face",
        "confidence_threshold": 0.5
    }
    
    try:
        response = requests.post(url, json=data, timeout=60)
        if response.status_code == 200:
            result = response.json()
            return result.get("faces", [])
        else:
            print(f"❌ Face detection failed: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Face detection exception: {e}")
        return []

def recognize_single_model(image_base64: str, model_name: str) -> Dict[str, Any]:
    """ทำการจดจำด้วยโมเดลเดียว"""
    # ดึง gallery
    try:
        gallery_response = requests.get(f"{API_BASE_URL}/api/face-recognition/get-gallery", timeout=30)
        if gallery_response.status_code != 200:
            return {"matches": [], "error": "Cannot get gallery"}
        gallery = gallery_response.json()
    except Exception as e:
        return {"matches": [], "error": str(e)}
    
    if not gallery:
        return {"matches": [], "error": "Gallery is empty"}
    
    url = f"{API_BASE_URL}/api/face-recognition/recognize"
    
    data = {
        "face_image_base64": image_base64,
        "gallery": gallery,
        "model_name": model_name,
        "top_k": 5,
        "similarity_threshold": 0.3
    }
    
    try:
        response = requests.post(url, json=data, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            return {"matches": [], "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"matches": [], "error": str(e)}

def crop_face_from_image(image_base64: str, face: Dict[str, Any]) -> str:
    """ครอปใบหน้าจากภาพ base64"""
    try:
        # Decode base64 to numpy array
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return ""
        
        # Get face coordinates
        bbox = face.get("bbox", {})
        x1 = int(bbox.get("x1", 0))
        y1 = int(bbox.get("y1", 0))
        x2 = int(bbox.get("x2", 0))
        y2 = int(bbox.get("y2", 0))
        
        # Add padding (15% of face size)
        width = x2 - x1
        height = y2 - y1
        padding_x = int(width * 0.15)
        padding_y = int(height * 0.15)
        
        # Expand bounding box with padding
        h, w = image.shape[:2]
        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(w, x2 + padding_x)
        y2 = min(h, y2 + padding_y)
        
        # Crop face
        cropped = image[y1:y2, x1:x2]
        
        # Convert back to base64
        _, encoded = cv2.imencode('.jpg', cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cropped_base64 = base64.b64encode(encoded.tobytes()).decode('utf-8')
        
        return cropped_base64
        
    except Exception as e:
        print(f"Cannot crop face: {e}")
        return ""

def test_group_image(image_file: str):
    """ทดสอบภาพกลุ่มเฉพาะ"""
    print(f"\n🖼️ Testing: {image_file}")
    print("=" * 50)
    
    image_path = os.path.join(TEST_IMAGES_DIR, image_file)
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_file}")
        return
    
    # แปลงเป็น base64
    image_base64 = image_to_base64(image_path)
    
    # 1. ตรวจจับใบหน้า
    faces = detect_faces(image_base64)
    if not faces:
        print("❌ No faces detected")
        return
    
    print(f"👤 Detected {len(faces)} face(s)")
    
    # แสดงขนาดใบหน้าทั้งหมด
    face_areas = []
    for i, face in enumerate(faces):
        bbox = face.get("bbox", {})
        width = bbox.get("x2", 0) - bbox.get("x1", 0)
        height = bbox.get("y2", 0) - bbox.get("y1", 0)
        area = width * height
        face_areas.append(area)
        print(f"  Face {i+1}: {width}x{height} = {area:,} pixels")
    
    # หาใบหน้าที่ใหญ่ที่สุด
    main_face_idx = face_areas.index(max(face_areas))
    main_face = faces[main_face_idx]
    print(f"📏 Largest face: #{main_face_idx+1} (area: {max(face_areas):,})")
    
    # 2. ทดสอบการจดจำ - วิธีเดิม (ภาพเต็ม)
    print(f"\n🔍 Method 1: Full Image Recognition")
    facenet_result = recognize_single_model(image_base64, "facenet")
    matches = facenet_result.get("matches", [])
    if matches:
        best = matches[0]
        print(f"  FACENET (full): {best.get('person_id', 'unknown')} ({best.get('similarity', 0):.3f})")
    else:
        print(f"  FACENET (full): unknown")
    
    # 3. ทดสอบการจดจำ - วิธีใหม่ (ครอปใบหน้าหลัก)
    print(f"\n✂️ Method 2: Cropped Main Face Recognition")
    cropped_base64 = crop_face_from_image(image_base64, main_face)
    if cropped_base64:
        facenet_cropped = recognize_single_model(cropped_base64, "facenet")
        matches_cropped = facenet_cropped.get("matches", [])
        if matches_cropped:
            best_cropped = matches_cropped[0]
            print(f"  FACENET (cropped): {best_cropped.get('person_id', 'unknown')} ({best_cropped.get('similarity', 0):.3f})")
        else:
            print(f"  FACENET (cropped): unknown")
    else:
        print(f"  ❌ Cannot crop main face")
    
    # 4. เปรียบเทียบผล
    print(f"\n📊 Comparison:")
    if matches and matches_cropped:
        full_sim = matches[0].get('similarity', 0)
        crop_sim = matches_cropped[0].get('similarity', 0)
        full_id = matches[0].get('person_id', 'unknown')
        crop_id = matches_cropped[0].get('person_id', 'unknown')
        
        print(f"  Full image:  {full_id} ({full_sim:.3f})")
        print(f"  Cropped:     {crop_id} ({crop_sim:.3f})")
        
        if crop_sim > full_sim:
            print(f"  🎯 Cropping improved similarity by {crop_sim - full_sim:.3f}")
        elif full_sim > crop_sim:
            print(f"  📉 Cropping reduced similarity by {full_sim - crop_sim:.3f}")
        else:
            print(f"  ⚖️ Same similarity")
            
        if full_id != crop_id:
            print(f"  ⚠️ Different person recognition!")
    
def main():
    """ทดสอบภาพกลุ่มทั้งหมด"""
    print("🧪 Group Image Recognition Test")
    print("Testing improved multi-face handling")
    
    # ตรวจสอบ API
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code != 200:
            print("❌ API server not ready")
            return
        print("✅ API server is ready")
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return
    
    # ทดสอบภาพกลุ่ม
    group_images = [
        "boss_group01.jpg",
        "boss_group02.jpg", 
        "boss_group03.jpg",
        "night_group01.jpg",
        "night_group02.jpg",
        "face-swap03.png"  # มี 2 หน้า
    ]
    
    for img in group_images:
        test_group_image(img)
    
    print(f"\n🎉 Group image test completed!")

if __name__ == "__main__":
    main()
