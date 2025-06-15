#!/usr/bin/env python3
"""
Crop all faces from boss_group02.jpg for analysis
"""

import cv2
import requests
import os
from pathlib import Path
import base64

def clean_response_data(data):
    """Recursively clean response data to remove any base64 image data"""
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            if key.lower() in ['image', 'image_data', 'face_image', 'face_images'] and isinstance(value, str):
                # Skip base64 image data
                cleaned[key] = "[base64 data removed]"
            else:
                cleaned[key] = clean_response_data(value)
        return cleaned
    elif isinstance(data, list):
        return [clean_response_data(item) for item in data]
    else:
        return data

def encode_image_to_base64(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def detect_faces_api(image_path, model="auto"):
    """Detect faces using the API"""
    url = "http://localhost:8080/api/face-detection/detect-base64"
      # Encode image
    image_base64 = encode_image_to_base64(image_path)
    
    payload = {
        "image_base64": image_base64,
        "model_name": model,
        "conf_threshold": 0.5
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            # Ensure no base64 data is accidentally printed by removing any image data
            if isinstance(result, dict):
                # Clean response by removing any potential base64 image data
                result = clean_response_data(result)
            return result
        else:
            print(f"API Error: {response.status_code}")
            # Don't print response content to avoid base64 leakage
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
        return None

def crop_face_from_image(image, bbox):
    """Crop face from image using bounding box"""
    x1 = int(bbox['x1'])
    y1 = int(bbox['y1'])
    x2 = int(bbox['x2'])
    y2 = int(bbox['y2'])
    
    # Add some padding
    padding = 20
    height, width = image.shape[:2]
    
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    
    cropped_face = image[y1:y2, x1:x2]
    return cropped_face

def main():
    image_path = "test_images/boss_group02.jpg"
    output_dir = Path("output/cropped_faces_boss_group02")
    output_dir.mkdir(exist_ok=True)
    
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found!")
        return
    
    print(f"🔍 Detecting faces in {image_path}...")
    
    # Detect faces
    detection_result = detect_faces_api(image_path)
    
    if not detection_result:
        print("❌ Failed to detect faces!")
        return
    
    if not detection_result.get('success', False):
        print("❌ Face detection failed!")
        print("Response keys:", list(detection_result.keys()) if detection_result else "None")
        return
    
    faces = detection_result.get('faces', [])
    print(f"✅ Found {len(faces)} faces!")
    
    # Load original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    print(f"📐 Original image size: {original_image.shape[1]}x{original_image.shape[0]}")
    
    # Sort faces by area (largest first)
    faces_with_area = []
    for i, face in enumerate(faces):
        bbox = face.get('bbox', {})
        area = bbox.get('area', 0)
        faces_with_area.append((i, face, area))
    
    faces_with_area.sort(key=lambda x: x[2], reverse=True)
    
    print("\n📊 Face Information (sorted by size):")
    for rank, (original_idx, face, area) in enumerate(faces_with_area, 1):
        bbox = face.get('bbox', {})
        confidence = bbox.get('confidence', 0)
        width = bbox.get('width', 0)
        height = bbox.get('height', 0)
        print(f"  #{rank} (Original #{original_idx+1}): {width}x{height} pixels, Area: {area:.0f}, Confidence: {confidence:.3f}")
    
    # Crop and save all faces
    print(f"\n✂️ Cropping and saving faces to {output_dir}/...")
    
    for rank, (original_idx, face, area) in enumerate(faces_with_area, 1):
        bbox = face.get('bbox', {})
        
        try:
            cropped_face = crop_face_from_image(original_image, bbox)
            
            if cropped_face is not None and cropped_face.size > 0:
                # Save cropped face
                output_filename = f"face_{rank:02d}_original_{original_idx+1:02d}_area_{area:.0f}.jpg"
                output_path = output_dir / output_filename
                
                success = cv2.imwrite(str(output_path), cropped_face)
                if success:
                    print(f"  ✅ Saved: {output_filename} ({cropped_face.shape[1]}x{cropped_face.shape[0]})")
                else:
                    print(f"  ❌ Failed to save: {output_filename}")
            else:
                print(f"  ❌ Invalid cropped face for #{rank}")
                
        except Exception as e:
            print(f"  ❌ Error cropping face #{rank}: {str(e)}")
    
    # Also create a version with all faces marked
    print("\n🎯 Creating annotated version...")
    annotated_image = original_image.copy()
    
    for rank, (original_idx, face, area) in enumerate(faces_with_area, 1):
        bbox = face.get('bbox', {})
        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        
        # Different colors for different ranks
        colors = [
            (0, 255, 0),    # Green for largest
            (255, 0, 0),    # Blue for 2nd
            (0, 0, 255),    # Red for 3rd
            (255, 255, 0),  # Cyan for 4th
            (255, 0, 255),  # Magenta for 5th
        ]
        color = colors[(rank-1) % len(colors)]
        thickness = 3 if rank == 1 else 2
        
        # Draw rectangle
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
        
        # Add label
        label = f"#{rank}"
        font_scale = 0.8
        font_thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        # Background for text
        cv2.rectangle(annotated_image, (x1, y1-text_height-10), (x1+text_width+10, y1), color, -1)
        cv2.putText(annotated_image, label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    
    # Save annotated image
    annotated_path = output_dir / "boss_group02_all_faces_annotated.jpg"
    cv2.imwrite(str(annotated_path), annotated_image)
    print(f"✅ Saved annotated image: {annotated_path}")
    
    print("\n🎉 Complete! Check the output folder for all cropped faces.")
    print("📋 Summary:")
    print(f"  - Total faces found: {len(faces)}")
    if faces_with_area:
        print(f"  - Largest face: #{1} with area {faces_with_area[0][2]:.0f} pixels")
        print(f"  - Smallest face: #{len(faces)} with area {faces_with_area[-1][2]:.0f} pixels")

if __name__ == "__main__":
    main()
