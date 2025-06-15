#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face Detection Test - Process All Images in test_images
ทดสอบการจับใบหน้าทั้งหมดในโฟลเดอร์ test_images และวาดกรอบใบหน้า
"""

import os
import cv2
import requests
import base64
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8080"
TEST_IMAGES_DIR = "test_images"
OUTPUT_DIR = "output/face_detection_results"
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

# Colors for different models (BGR format for OpenCV)
MODEL_COLORS = {
    'auto': (0, 255, 0),      # Green
    'yolov9c': (255, 0, 0),   # Blue  
    'yolov9e': (0, 0, 255),   # Red
    'yolov11m': (255, 255, 0) # Cyan
}

def ensure_output_dir():
    """สร้างโฟลเดอร์ output"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")

def get_image_files() -> List[str]:
    """ดึงรายชื่อไฟล์รูปภาพทั้งหมดจาก test_images"""
    image_files = []
    if not os.path.exists(TEST_IMAGES_DIR):
        logger.error(f"Test images directory not found: {TEST_IMAGES_DIR}")
        return []
    
    for file in os.listdir(TEST_IMAGES_DIR):
        if any(file.lower().endswith(ext) for ext in SUPPORTED_FORMATS):
            image_files.append(file)
    
    logger.info(f"Found {len(image_files)} image files")
    return sorted(image_files)

def image_to_base64(image_path: str) -> str:
    """แปลงรูปภาพเป็น base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def detect_faces_api(image_base64: str, model_name: str = "auto") -> Dict[str, Any]:
    """เรียก Face Detection API"""
    url = f"{API_BASE_URL}/api/face-detection/detect-base64"
    
    data = {
        "image_base64": image_base64,
        "model_name": model_name,
        "conf_threshold": 0.3,  # ลดค่า threshold เพื่อจับใบหน้าได้มากขึ้น
        "iou_threshold": 0.4,
        "max_faces": 50,
        "min_quality_threshold": 20.0  # ลดค่า quality threshold
    }
    
    try:
        response = requests.post(url, json=data, timeout=120)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"API Error {response.status_code}: {response.text}")
            return {}
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return {}

def draw_face_boxes(image: cv2.Mat, faces: List[Dict], model_name: str, processing_time: float) -> cv2.Mat:
    """วาดกรอบใบหน้าบนรูปภาพ"""
    result_image = image.copy()
    color = MODEL_COLORS.get(model_name, (0, 255, 0))
    
    # วาดกรอบใบหน้าแต่ละใบ
    for i, face in enumerate(faces):
        bbox = face.get('bbox', {})
        x1 = int(bbox.get('x1', 0))
        y1 = int(bbox.get('y1', 0))
        x2 = int(bbox.get('x2', 0))
        y2 = int(bbox.get('y2', 0))
        confidence = bbox.get('confidence', 0)
        quality_score = face.get('quality_score', 0)
        
        # วาดกรอบหลัก
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)
        
        # เขียนข้อมูลใบหน้า
        label = f"Face {i+1}: {confidence:.2f} | Q:{quality_score:.1f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # พื้นหลังข้อความ
        cv2.rectangle(result_image, 
                     (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), 
                     color, -1)
        
        # ข้อความ
        cv2.putText(result_image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # เขียนข้อมูลสรุป
    summary_text = [
        f"Model: {model_name.upper()}",
        f"Faces: {len(faces)}",
        f"Time: {processing_time:.1f}ms"
    ]
    
    y_offset = 30
    for text in summary_text:
        cv2.putText(result_image, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y_offset += 35
    
    return result_image

def process_single_image(image_file: str, models: List[str]) -> Dict[str, Any]:
    """ประมวลผลรูปภาพเดียวด้วยทุกโมเดล"""
    image_path = os.path.join(TEST_IMAGES_DIR, image_file)
    logger.info(f"Processing: {image_file}")
    
    # โหลดรูปภาพ
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_file}")
        return {}
    
    # แปลงเป็น base64
    image_base64 = image_to_base64(image_path)
    
    results = {}
    
    # ทดสอบแต่ละโมเดล
    for model_name in models:
        logger.info(f"  Testing model: {model_name}")
        
        start_time = time.time()
        api_result = detect_faces_api(image_base64, model_name)
        processing_time = (time.time() - start_time) * 1000
        
        if api_result and 'faces' in api_result:
            faces = api_result['faces']
            api_processing_time = api_result.get('total_processing_time', 0)
            
            # วาดกรอบใบหน้า
            result_image = draw_face_boxes(image, faces, model_name, api_processing_time)
            
            # บันทึกรูปผลลัพธ์
            base_name = os.path.splitext(image_file)[0]
            output_filename = f"{base_name}_{model_name}_detected.jpg"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            # บันทึกด้วยคุณภาพสูงสุด (ไม่ลดความละเอียด)
            cv2.imwrite(output_path, result_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            results[model_name] = {
                'faces_count': len(faces),
                'faces': faces,
                'processing_time_ms': api_processing_time,
                'total_time_ms': processing_time,
                'output_file': output_filename,
                'success': True
            }
            
            logger.info(f"    {model_name}: {len(faces)} faces detected ({api_processing_time:.1f}ms)")
        else:
            results[model_name] = {
                'success': False,
                'error': 'API call failed or no response'
            }
            logger.error(f"    {model_name}: Failed")
    
    return results

def save_detailed_report(all_results: Dict[str, Dict], models: List[str]):
    """บันทึกรายงานละเอียด"""
    report_path = os.path.join(OUTPUT_DIR, "detection_report.json")
    
    # สร้างสรุปรายงาน
    summary = {
        'total_images': len(all_results),
        'models_tested': models,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': all_results
    }
    
    # คำนวณสถิติ
    model_stats = {}
    for model in models:
        successful_detections = 0
        total_faces = 0
        total_time = 0
        
        for image_file, results in all_results.items():
            if model in results and results[model].get('success'):
                successful_detections += 1
                total_faces += results[model]['faces_count']
                total_time += results[model]['processing_time_ms']
        
        model_stats[model] = {
            'successful_images': successful_detections,
            'success_rate': (successful_detections / len(all_results)) * 100,
            'total_faces_detected': total_faces,
            'average_faces_per_image': total_faces / max(successful_detections, 1),
            'average_processing_time_ms': total_time / max(successful_detections, 1)
        }
    
    summary['model_statistics'] = model_stats
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Detailed report saved: {report_path}")
    return model_stats

def print_summary_table(model_stats: Dict[str, Dict]):
    """แสดงตารางสรุปผลลัพธ์"""
    print("\n" + "="*80)
    print("📊 FACE DETECTION RESULTS SUMMARY")
    print("="*80)
    print(f"{'Model':<12} {'Success%':<10} {'Total Faces':<12} {'Avg Faces':<10} {'Avg Time(ms)':<12}")
    print("-"*80)
    
    for model, stats in model_stats.items():
        print(f"{model:<12} {stats['success_rate']:<9.1f}% {stats['total_faces_detected']:<12} "
              f"{stats['average_faces_per_image']:<9.1f} {stats['average_processing_time_ms']:<12.1f}")
    
    print("="*80)

def main():
    """Main function"""
    print("🎯 Face Detection Test - Processing All Images")
    print("="*60)
    
    # สร้างโฟลเดอร์ output
    ensure_output_dir()
    
    # ดึงรายชื่อไฟล์รูปภาพ
    image_files = get_image_files()
    if not image_files:
        print("❌ No image files found!")
        return
    
    # โมเดลที่จะทดสอบ
    models = ['auto', 'yolov9c', 'yolov9e', 'yolov11m']
    
    print(f"📁 Images to process: {len(image_files)}")
    print(f"🧠 Models to test: {models}")
    print(f"📤 Output directory: {OUTPUT_DIR}")
    print()
    
    # ตรวจสอบ API
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code != 200:
            print("❌ API server is not accessible!")
            return
        print("✅ API server is ready")
    except:
        print("❌ Cannot connect to API server!")
        return
    
    # ประมวลผลทุกรูปภาพ
    all_results = {}
    start_time = time.time()
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_file}")
        results = process_single_image(image_file, models)
        all_results[image_file] = results
    
    total_time = time.time() - start_time
    
    # บันทึกรายงานและแสดงสรุป
    model_stats = save_detailed_report(all_results, models)
    print_summary_table(model_stats)
    
    print(f"\n⏱️  Total processing time: {total_time:.1f} seconds")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print(f"📄 Detailed report: {OUTPUT_DIR}/detection_report.json")
    print("\n🎉 Face detection test completed!")

if __name__ == "__main__":
    main()
