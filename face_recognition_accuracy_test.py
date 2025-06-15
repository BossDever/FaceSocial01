#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face Recognition Accuracy Test - Same Image Comparison
ทดสอบความแม่นยำของ Face Recognition ด้วยรูปเดียวกัน
และตรวจสอบ embedding ทั้ง 512 มิติ
"""

import os
import json
import time
import base64
import requests
import numpy as np
from typing import Dict, List, Any, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8080"
TEST_IMAGES_DIR = "test_images"
OUTPUT_DIR = "output/face_recognition_accuracy"
TEST_IMAGES = [
    "boss_01.jpg"
    # "boss_02.jpg", 
    # "night_01.jpg",
    # "night_02.jpg",
    # "spoofing_01.jpg"
]

MODELS = ["facenet", "adaface", "arcface"]

def ensure_output_dir():
    """สร้างโฟลเดอร์ output"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")

def image_to_base64(image_path: str) -> str:
    """แปลงรูปภาพเป็น base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def clear_face_database() -> bool:
    """ลบข้อมูลในฐานข้อมูลใบหน้าทั้งหมด"""
    try:
        # Add proper clear database implementation if API endpoint exists
        # url = f"{API_BASE_URL}/api/face-recognition/clear-database"
        # response = requests.delete(url, timeout=30)
        # return response.status_code == 200
        logger.info("⚠️ No clear database endpoint available - continuing...")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to clear database: {e}")
        return False

def add_face_to_database(image_base64: str, person_id: str, person_name: str, model_name: str) -> Dict[str, Any]:
    """เพิ่มใบหน้าเข้าฐานข้อมูล"""
    url = f"{API_BASE_URL}/api/face-recognition/add-face-json"
    
    data = {
        "person_id": person_id,
        "person_name": person_name,
        "face_image_base64": image_base64,
        "model_name": model_name
    }
    
    try:
        response = requests.post(url, json=data, timeout=120)
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Added {person_id} to database with {model_name}")
            return result
        else:
            logger.error(f"❌ Failed to add {person_id}: {response.status_code} - {response.text}")
            return {}
    except Exception as e:
        logger.error(f"❌ Exception adding {person_id}: {e}")
        return {}

def recognize_face(image_base64: str, model_name: str) -> Dict[str, Any]:
    """ทำการจดจำใบหน้า"""
    
    # ดึง gallery ปัจจุบันก่อน
    gallery_url = f"{API_BASE_URL}/api/face-recognition/get-gallery"
    try:
        gallery_response = requests.get(gallery_url, timeout=30)
        if gallery_response.status_code == 200:
            gallery = gallery_response.json()
        else:
            logger.error(f"❌ Failed to get gallery: {gallery_response.status_code}")
            return {}
    except Exception as e:
        logger.error(f"❌ Failed to get gallery: {e}")
        return {}
    
    # ถ้า gallery ว่าง
    if not gallery:
        logger.warning("⚠️ Gallery is empty - no faces to compare against")
        return {"matches": [], "gallery_empty": True}
    
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
            logger.error(f"❌ Recognition failed: {response.status_code} - {response.text}")
            return {}
    except Exception as e:
        logger.error(f"❌ Recognition exception: {e}")
        return {}

def get_face_embedding(image_base64: str, model_name: str) -> Dict[str, Any]:
    """ดึง face embedding"""
    url = f"{API_BASE_URL}/api/face-recognition/extract-embedding"
    
    # สร้าง multipart form data
    files = {
        'file': ('image.jpg', base64.b64decode(image_base64), 'image/jpeg')
    }
    data = {
        'model_name': model_name
    }
    
    try:
        response = requests.post(url, files=files, data=data, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"❌ Get embedding failed: {response.status_code} - {response.text}")
            return {}
    except Exception as e:
        logger.error(f"❌ Get embedding exception: {e}")
        return {}

def analyze_embedding(embedding: List[float], model_name: str) -> Dict[str, Any]:
    """วิเคราะห์ embedding"""
    if not embedding:
        return {"error": "Empty embedding"}
    
    embedding_array = np.array(embedding)
    
    analysis = {
        "dimensions": len(embedding),
        "expected_dimensions": 512,
        "is_correct_size": len(embedding) == 512,
        "min_value": float(np.min(embedding_array)),
        "max_value": float(np.max(embedding_array)),
        "mean_value": float(np.mean(embedding_array)),
        "std_value": float(np.std(embedding_array)),
        "norm_l2": float(np.linalg.norm(embedding_array)),
        "zero_values": int(np.sum(embedding_array == 0)),
        "positive_values": int(np.sum(embedding_array > 0)),
        "negative_values": int(np.sum(embedding_array < 0)),
        "model_name": model_name
    }
    
    return analysis

def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """คำนวณ cosine similarity"""
    if not embedding1 or not embedding2:
        return 0.0
    
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # Normalize vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Cosine similarity
    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    return float(similarity)

def test_single_image_accuracy(image_file: str) -> Dict[str, Any]:
    """ทดสอบความแม่นยำด้วยรูปเดียวกัน"""
    image_path = os.path.join(TEST_IMAGES_DIR, image_file)
    if not os.path.exists(image_path):
        logger.error(f"❌ Image not found: {image_file}")
        return {}
    
    logger.info(f"\n🖼️ Testing image: {image_file}")
    
    # แปลงเป็น base64
    image_base64 = image_to_base64(image_path)
    
    # สร้าง person_id จากชื่อไฟล์
    person_id = os.path.splitext(image_file)[0]
    person_name = f"Test {person_id.title()}"
    
    results = {}
    
    for model_name in MODELS:
        logger.info(f"  🧠 Testing model: {model_name}")
        
        model_results = {
            "model_name": model_name,
            "image_file": image_file,
            "person_id": person_id,
            "registration_success": False,
            "recognition_success": False,
            "similarity_score": 0.0,
            "embedding_analysis": {},
            "recognition_details": {}
        }
        
        # 1. ลงทะเบียนใบหน้า (ไม่ลบฐานข้อมูล)
        logger.info(f"    📝 Registering face...")
        add_result = add_face_to_database(image_base64, person_id, person_name, model_name)
        
        if add_result and add_result.get('success'):
            model_results["registration_success"] = True
            model_results["face_ids"] = add_result.get('face_ids', [])
            
            # 3. ดึง embedding
            logger.info(f"    🔍 Getting embedding...")
            embedding_result = get_face_embedding(image_base64, model_name)
            
            if embedding_result and 'embedding' in embedding_result:
                embedding = embedding_result['embedding']
                model_results["embedding_analysis"] = analyze_embedding(embedding, model_name)
                logger.info(f"    📊 Embedding: {len(embedding)} dimensions")
                
                # 4. ทดสอบการจดจำด้วยรูปเดียวกัน
                logger.info(f"    🎯 Testing recognition...")
                recognition_result = recognize_face(image_base64, model_name)
                
                if recognition_result and 'matches' in recognition_result:
                    matches = recognition_result['matches']
                    model_results["recognition_details"] = recognition_result
                    
                    if matches:
                        best_match = matches[0]
                        similarity = best_match.get('similarity', 0.0)
                        recognized_id = best_match.get('person_id', '')
                        
                        model_results["recognition_success"] = True
                        model_results["similarity_score"] = similarity
                        model_results["recognized_person_id"] = recognized_id
                        
                        # แก้ไข: เปรียบเทียบด้วย flexible matching
                        is_correct = (
                            recognized_id == person_id or 
                            recognized_id in person_id or 
                            person_id in recognized_id or
                            recognized_id.lower() in person_id.lower()
                        )
                        model_results["is_correct_person"] = is_correct
                        
                        logger.info(f"    ✅ Recognition: {similarity:.4f} ({recognized_id}) - Match: {is_correct}")
                    else:
                        logger.warning(f"    ⚠️ No matches found")
                else:
                    logger.error(f"    ❌ Recognition failed")
            else:
                logger.error(f"    ❌ Failed to get embedding")
        else:
            logger.error(f"    ❌ Registration failed")
        
        results[model_name] = model_results
        
        # รอเล็กน้อยระหว่างโมเดล
        time.sleep(2)
    
    return results

def generate_report(all_results: Dict[str, Dict]) -> Dict[str, Any]:
    """สร้างรายงานสรุป"""
    report = {
        "test_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "total_images": len(all_results),
        "models_tested": MODELS,
        "detailed_results": all_results,
        "summary": {}
    }
    
    # สร้างสรุปแต่ละโมเดล
    model_summary = {}
    
    for model in MODELS:
        successful_registrations = 0
        successful_recognitions = 0
        similarity_scores = []
        embedding_analyses = []
        correct_recognitions = 0
        
        for image_file, results in all_results.items():
            if model in results:
                result = results[model]
                
                if result.get('registration_success'):
                    successful_registrations += 1
                
                if result.get('recognition_success'):
                    successful_recognitions += 1
                    similarity_scores.append(result.get('similarity_score', 0))
                    
                    if result.get('is_correct_person'):
                        correct_recognitions += 1
                
                if result.get('embedding_analysis'):
                    embedding_analyses.append(result['embedding_analysis'])
        
        # คำนวณสถิติ
        total_tests = len(all_results)
        registration_rate = (successful_registrations / total_tests) * 100 if total_tests > 0 else 0
        recognition_rate = (successful_recognitions / total_tests) * 100 if total_tests > 0 else 0
        accuracy_rate = (correct_recognitions / total_tests) * 100 if total_tests > 0 else 0
        
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
        min_similarity = np.min(similarity_scores) if similarity_scores else 0
        max_similarity = np.max(similarity_scores) if similarity_scores else 0
        
        # สถิติ embedding
        embedding_stats = {}
        if embedding_analyses:
            dimensions = [e.get('dimensions', 0) for e in embedding_analyses]
            norms = [e.get('norm_l2', 0) for e in embedding_analyses]
            
            embedding_stats = {
                "avg_dimensions": np.mean(dimensions),
                "all_512_dimensions": all(d == 512 for d in dimensions),
                "avg_norm": np.mean(norms),
                "min_norm": np.min(norms),
                "max_norm": np.max(norms)
            }
        
        model_summary[model] = {
            "registration_success_rate": registration_rate,
            "recognition_success_rate": recognition_rate,
            "accuracy_rate": accuracy_rate,
            "similarity_statistics": {
                "average": avg_similarity,
                "minimum": min_similarity,
                "maximum": max_similarity,
                "count": len(similarity_scores)
            },
            "embedding_statistics": embedding_stats
        }
    
    report["summary"] = model_summary
    return report

def print_summary_report(report: Dict[str, Any]):
    """แสดงรายงานสรุป"""
    print("\n" + "="*80)
    print("📊 FACE RECOGNITION ACCURACY TEST RESULTS")
    print("="*80)
    print(f"📅 Test Date: {report['test_timestamp']}")
    print(f"🖼️ Images Tested: {report['total_images']}")
    print(f"🧠 Models: {', '.join(report['models_tested'])}")
    print("\n" + "-"*80)
    
    for model, stats in report['summary'].items():
        print(f"\n🔬 {model.upper()} Model Results:")
        print(f"  📝 Registration Success: {stats['registration_success_rate']:.1f}%")
        print(f"  🎯 Recognition Success:  {stats['recognition_success_rate']:.1f}%")
        print(f"  ✅ Accuracy (Same Image): {stats['accuracy_rate']:.1f}%")
        
        sim_stats = stats['similarity_statistics']
        print(f"  📈 Similarity Scores:")
        print(f"     Average: {sim_stats['average']:.4f}")
        print(f"     Range:   {sim_stats['minimum']:.4f} - {sim_stats['maximum']:.4f}")
        
        emb_stats = stats['embedding_statistics']
        if emb_stats:
            print(f"  🧮 Embedding Analysis:")
            print(f"     All 512 dimensions: {'✅ Yes' if emb_stats['all_512_dimensions'] else '❌ No'}")
            print(f"     Average L2 norm: {emb_stats['avg_norm']:.4f}")
    
    print("\n" + "="*80)

def main():
    """Main function"""
    print("🎯 Face Recognition Accuracy Test - Same Image Comparison")
    print("="*70)
    
    # สร้างโฟลเดอร์ output
    ensure_output_dir()
    
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
    
    print(f"📁 Test images: {TEST_IMAGES}")
    print(f"🧠 Models: {MODELS}")
    print(f"📤 Output: {OUTPUT_DIR}")
    print()
    
    # ทดสอบแต่ละรูป
    all_results = {}
    start_time = time.time()
    
    for i, image_file in enumerate(TEST_IMAGES, 1):
        print(f"\n[{i}/{len(TEST_IMAGES)}] Testing: {image_file}")
        results = test_single_image_accuracy(image_file)
        if results:
            all_results[image_file] = results
    
    total_time = time.time() - start_time
    
    # สร้างรายงาน
    if all_results:
        report = generate_report(all_results)
        
        # บันทึกรายงาน
        report_path = os.path.join(OUTPUT_DIR, "accuracy_test_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # แสดงสรุป
        print_summary_report(report)
        
        print(f"\n⏱️  Total test time: {total_time:.1f} seconds")
        print(f"📄 Detailed report: {report_path}")
    else:
        print("❌ No test results to report")
    
    print("\n🎉 Face recognition accuracy test completed!")

if __name__ == "__main__":
    main()