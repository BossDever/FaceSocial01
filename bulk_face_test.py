import requests
import os
import glob
import cv2
import time
from typing import Optional, Dict, Any, cast
import json
import numpy as np
import base64

API_BASE_URL = "http://127.0.0.1:8080/api"
TEST_IMAGES_DIR = r"D:\projec-finals\test_images"
OUTPUT_DIR = r"D:\projec-finals\output\bulk_test_output"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def encode_image_to_base64(image_path: str) -> Optional[str]:
    """Convert image file to base64 string"""
    try:
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            return img_base64
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def register_face(image_path: str, person_id: str, person_name: str) -> Optional[Dict[str, Any]]:
    """Registers a face using the /api/face-recognition/add-face-json endpoint."""
    url = f"{API_BASE_URL}/face-recognition/add-face-json"
    
    # Convert image to base64
    image_base64 = encode_image_to_base64(image_path)
    if not image_base64:
        print(f"Failed to encode image: {image_path}")
        return None
    
    data = {
        'person_id': person_id,
        'person_name': person_name,
        'face_image_base64': image_base64,
        'model_name': 'facenet'
    }
    
    try:
        print(f"Registering {person_id} ({person_name}) with image {os.path.basename(image_path)}...")
        start_time = time.time()
        response = requests.post(url, json=data)
        response.raise_for_status()
        end_time = time.time()
        reg_time = end_time - start_time
        
        result = cast(Dict[str, Any], response.json())
        
        # Clean up embedding preview for better display
        if 'embedding_preview' in result:
            embedding_preview = result['embedding_preview']
            if isinstance(embedding_preview, list) and len(embedding_preview) > 0:
                preview_str = f"[{', '.join(f'{x:.3f}' for x in embedding_preview[:3])}...] ({len(embedding_preview)} dims)"
                result['embedding_preview_clean'] = preview_str
        
        print(f"✅ Registration SUCCESS for {person_id} ({person_name})")
        print(f"   Time: {reg_time:.2f}s")
        print(f"   Model: {result.get('model_used', 'unknown')}")
        if 'embedding_preview_clean' in result:
            print(f"   Embedding: {result['embedding_preview_clean']}")
        print()
        
        return result
        
    except FileNotFoundError:
        print(f"❌ Error: Image file not found at {image_path}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Error registering {person_id} with {os.path.basename(image_path)}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Status Code: {e.response.status_code}")
            try:
                error_json = e.response.json()
                if "detail" in error_json:
                    print("   Error Detail:")
                    print(json.dumps(error_json["detail"], indent=2))
            except ValueError:
                print(f"   Response content: {e.response.text}")
        return None

def draw_faces_on_image(image: np.ndarray, analysis_results: Dict[str, Any]) -> np.ndarray:
    """Draw bounding boxes and labels on image"""
    img_result = image.copy()
    
    if not analysis_results.get('faces'):
        return img_result
    
    # Color scheme for different recognition results
    colors = {
        'recognized': (0, 255, 0),      # Green for recognized faces
        'unknown': (0, 165, 255),       # Orange for unknown faces
        'no_analysis': (0, 0, 255),     # Red for detection only
    }
    
    for i, face in enumerate(analysis_results['faces']):
        # Get bounding box
        bbox = face.get('bbox', {})
        if not bbox:
            continue
            
        x1 = int(bbox.get('x1', 0))
        y1 = int(bbox.get('y1', 0))
        x2 = int(bbox.get('x2', 0))
        y2 = int(bbox.get('y2', 0))
        
        # Determine face status and color
        identity = face.get('identity')
        identity_name = face.get('identity_name')
        recognition_confidence = face.get('recognition_confidence', 0.0)
        
        if identity and identity != "Unknown" and recognition_confidence > 0.6:
            color = colors['recognized']
            label = identity_name if identity_name else identity
            status = f"✓ {label}"
            confidence_text = f"({recognition_confidence:.2f})"
        elif 'best_match' in face and face['best_match']:
            # Check if there's a match but low confidence
            color = colors['unknown']
            status = "? Unknown"
            confidence_text = f"({recognition_confidence:.2f})"
        else:
            color = colors['no_analysis']
            status = "Face"
            confidence_text = f"({bbox.get('confidence', 0.0):.2f})"
        
        # Draw bounding box
        cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 2)
        
        # Prepare text
        main_text = status
        detail_text = confidence_text
        quality_text = f"Q:{face.get('quality_score', 0):.0f}"
        
        # Calculate text position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Get text size for background rectangle
        (text_w, text_h), baseline = cv2.getTextSize(main_text, font, font_scale, thickness)
        
        # Draw text background
        text_bg_y1 = max(0, y1 - text_h - 10)
        text_bg_y2 = y1
        cv2.rectangle(img_result, (x1, text_bg_y1), (x1 + text_w + 10, text_bg_y2), color, -1)
        
        # Draw main text (name or status)
        cv2.putText(img_result, main_text, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness)
        
        # Draw additional info below the box
        info_y = y2 + 20
        cv2.putText(img_result, detail_text, (x1, info_y), font, 0.4, color, 1)
        cv2.putText(img_result, quality_text, (x1, info_y + 15), font, 0.4, color, 1)
        
        # Draw face number
        face_num = f"#{i+1}"
        cv2.putText(img_result, face_num, (x2 - 30, y1 + 15), font, 0.4, (255, 255, 255), 1)
    
    return img_result

def analyze_and_draw_faces(image_path: str, output_dir: str,
                           detection_model: str = "yolov9c",
                           recognition_model: str = "facenet") -> None:
    """Analyzes an image for faces, recognizes them, draws bounding boxes and labels."""
    url = f"{API_BASE_URL}/face-analysis/analyze-json"
    
    try:
        # Read original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"❌ Error: Could not read image {image_path}")
            return
        
        # Convert to base64
        image_base64 = encode_image_to_base64(image_path)
        if not image_base64:
            print(f"❌ Error: Could not encode image {image_path}")
            return
        
        # Prepare request data
        request_data = {
            "image_base64": image_base64,
            "mode": "full_analysis",
            "config": {
                "detection_model": detection_model,
                "recognition_model": recognition_model,
                "confidence_threshold": 0.5,
                "max_faces": 50
            }
        }
        
        print(f"🔍 Analyzing {os.path.basename(image_path)}...")
        start_time = time.time()
        response = requests.post(url, json=request_data)
        response.raise_for_status()
        analysis_time = time.time() - start_time
        
        analysis_results = response.json()
        
        # Print analysis summary
        stats = analysis_results.get('statistics', {})
        print(f"✅ Analysis complete in {analysis_time:.2f}s")
        print(f"   Total faces: {stats.get('total_faces', 0)}")
        print(f"   Usable faces: {stats.get('usable_faces', 0)}")
        print(f"   Identified faces: {stats.get('identified_faces', 0)}")
        print(f"   Recognition rate: {stats.get('recognition_success_rate', 0):.1%}")
        
        # Print face details
        faces = analysis_results.get('faces', [])
        if faces:
            print("   Face details:")
            for i, face in enumerate(faces):
                identity = face.get('identity')
                identity_name = face.get('identity_name')
                confidence = face.get('recognition_confidence', 0.0)
                quality = face.get('quality_score', 0.0)
                
                if identity and identity != "Unknown":
                    name_display = identity_name if identity_name else identity
                    print(f"     Face #{i+1}: {name_display} ({confidence:.2f}) Q:{quality:.0f}")
                else:
                    print(f"     Face #{i+1}: Unknown Q:{quality:.0f}")
        
        # Draw faces on image
        result_image = draw_faces_on_image(original_image, analysis_results)
        
        # Save annotated image
        output_filename = f"analyzed_{os.path.basename(image_path)}"
        output_path = os.path.join(output_dir, output_filename)
        success = cv2.imwrite(output_path, result_image)
        
        if success:
            print(f"💾 Saved annotated image: {output_path}")
        else:
            print(f"❌ Failed to save image: {output_path}")
        
        print()

    except FileNotFoundError:
        print(f"❌ Error: Image file not found at {image_path}")
    except requests.exceptions.RequestException as e:
        print(f"❌ API Error analyzing {os.path.basename(image_path)}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text[:200]}...")
    except Exception as e:
        print(f"❌ Unexpected error analyzing {os.path.basename(image_path)}: {e}")

def print_system_status():
    """Print system status and health check"""
    try:
        health_url = f"{API_BASE_URL.replace('/api', '')}/health"
        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("🏥 System Status:")
            print(f"   Status: {health_data.get('status', 'unknown')}")
            print(f"   Services: {health_data.get('total_services', 0)} active")
            print()
        else:
            print("⚠️ System health check failed")
    except Exception as e:
        print(f"⚠️ Could not connect to system: {e}")

def main() -> None:
    print("🎭 Face Recognition Bulk Test")
    print("=" * 50)
    print()
    
    # Check system status
    print_system_status()
    
    # Check if test images directory exists
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"❌ Test images directory not found: {TEST_IMAGES_DIR}")
        print("Please create the directory and add test images.")
        return
    
    # --- Registration Phase ---
    print("📝 REGISTRATION PHASE")
    print("-" * 30)
    
    boss_images = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, "boss_[0-9][0-9].jpg")))
    night_images = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, "night_[0-9][0-9].jpg")))
    
    if not boss_images and not night_images:
        print("⚠️ No boss_XX.jpg or night_XX.jpg images found")
        print(f"   Looking in: {TEST_IMAGES_DIR}")
        print("   Expected format: boss_01.jpg, boss_02.jpg, night_01.jpg, etc.")
    
    registered_count = 0
    
    if boss_images:
        print(f"\n👔 Registering Boss ({len(boss_images)} images):")
        for img_path in boss_images:
            result = register_face(img_path, "boss", "Boss")
            if result and result.get('success'):
                registered_count += 1
            time.sleep(0.2)  # Small delay between requests
    
    if night_images:
        print(f"\n🌙 Registering Night ({len(night_images)} images):")
        for img_path in night_images:
            result = register_face(img_path, "night", "Night")
            if result and result.get('success'):
                registered_count += 1
            time.sleep(0.2)  # Small delay between requests
    
    print(f"\n✅ Registration complete: {registered_count} faces registered")
    print()
    
    # --- Analysis Phase ---
    print("🔍 ANALYSIS PHASE")
    print("-" * 30)
    
    # Find all test images
    image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    all_test_images = []
    for pattern in image_patterns:
        all_test_images.extend(glob.glob(os.path.join(TEST_IMAGES_DIR, pattern)))
    
    all_test_images = sorted(list(set(all_test_images)))  # Remove duplicates and sort
    
    if not all_test_images:
        print(f"❌ No images found in {TEST_IMAGES_DIR}")
        return
    
    print(f"Found {len(all_test_images)} images to analyze")
    print()
    
    # Analyze each image
    for i, img_path in enumerate(all_test_images, 1):
        print(f"[{i}/{len(all_test_images)}] Processing: {os.path.basename(img_path)}")
        analyze_and_draw_faces(img_path, OUTPUT_DIR)
        time.sleep(0.1)  # Small delay
    
    # --- Summary ---
    print("📊 SUMMARY")
    print("-" * 30)
    print(f"✅ Registration: {registered_count} faces registered")
    print(f"✅ Analysis: {len(all_test_images)} images processed")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print()
    
    # List output files
    output_files = glob.glob(os.path.join(OUTPUT_DIR, "analyzed_*"))
    if output_files:
        print("📄 Generated files:")
        for output_file in sorted(output_files):
            print(f"   {os.path.basename(output_file)}")
    else:
        print("⚠️ No output files generated")
    
    print("\n🎉 Bulk processing complete!")

if __name__ == "__main__":
    main()
