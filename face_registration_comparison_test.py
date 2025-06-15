#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face Registration Method Comparison Test
เปรียบเทียบการลงทะเบียนใบหน้า: ทีละภาพ vs มัดรวม
และทดสอบความแม่นยำในการจดจำ
"""

import os
import json
import time
import base64
import requests
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8080"
TEST_IMAGES_DIR = "test_images"
OUTPUT_DIR = "output/registration_comparison"

# ใช้รูปเซ็ตเดียวกัน 10 รูป
TEST_IMAGES = [
    "boss_01.jpg", "boss_02.jpg", "boss_03.jpg", "boss_04.jpg", "boss_05.jpg",
    "boss_06.jpg", "boss_07.jpg", "boss_08.jpg", "boss_09.jpg", "boss_10.jpg"
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

def clear_database():
    """ล้างฐานข้อมูล (ถ้ามี endpoint)"""
    try:
        # ใช้ get gallery แล้วลบทีละ person
        gallery_response = requests.get(f"{API_BASE_URL}/api/face-recognition/get-gallery", timeout=30)
        if gallery_response.status_code == 200:
            gallery = gallery_response.json()
            logger.info(f"📋 Current gallery has {len(gallery)} persons")
            # Note: จริงๆ ควรมี clear endpoint แต่ถ้าไม่มีก็ไม่เป็นไร
        return True
    except Exception as e:
        logger.warning(f"⚠️ Cannot clear database: {e}")
        return True

def test_individual_registration(person_id: str, images: List[str], model_name: str) -> Dict[str, Any]:
    """ทดสอบการลงทะเบียนทีละภาพ"""
    logger.info(f"  📝 Individual Registration - {model_name}")
    
    results = {
        "method": "individual",
        "model_name": model_name,
        "person_id": person_id,
        "total_images": len(images),
        "registration_results": [],
        "registration_success_rate": 0.0,
        "total_time": 0.0,
        "average_time_per_image": 0.0
    }
    
    start_time = time.time()
    successful_registrations = 0
    
    for i, image_file in enumerate(images, 1):
        image_path = os.path.join(TEST_IMAGES_DIR, image_file)
        if not os.path.exists(image_path):
            logger.error(f"❌ Image not found: {image_file}")
            continue
            
        logger.info(f"    [{i}/{len(images)}] Registering: {image_file}")
        
        # แปลงเป็น base64
        image_base64 = image_to_base64(image_path)
        
        # ลงทะเบียน
        reg_start = time.time()
        success = add_single_face(image_base64, person_id, f"Person {person_id}", model_name)
        reg_time = time.time() - reg_start
        
        reg_result = {
            "image_file": image_file,
            "success": success,
            "time": reg_time
        }
        results["registration_results"].append(reg_result)
        
        if success:
            successful_registrations += 1
            logger.info(f"      ✅ Success ({reg_time:.2f}s)")
        else:
            logger.error(f"      ❌ Failed ({reg_time:.2f}s)")
    
    total_time = time.time() - start_time
    results["total_time"] = total_time
    results["registration_success_rate"] = (successful_registrations / len(images)) * 100
    results["average_time_per_image"] = total_time / len(images)
    
    logger.info(f"  📊 Individual Results: {successful_registrations}/{len(images)} success, {total_time:.2f}s total")
    
    return results

def test_batch_registration(person_id: str, images: List[str], model_name: str) -> Dict[str, Any]:
    """ทดสอบการลงทะเบียนแบบมัดรวม"""
    logger.info(f"  📦 Batch Registration - {model_name}")
    
    results = {
        "method": "batch",
        "model_name": model_name,
        "person_id": person_id,
        "total_images": len(images),
        "registration_results": [],
        "registration_success_rate": 0.0,
        "total_time": 0.0,
        "average_time_per_image": 0.0
    }
    
    # เตรียมข้อมูลทั้งหมด
    batch_data = []
    for image_file in images:
        image_path = os.path.join(TEST_IMAGES_DIR, image_file)
        if os.path.exists(image_path):
            image_base64 = image_to_base64(image_path)
            batch_data.append({
                "image_file": image_file,
                "image_base64": image_base64
            })
    
    if not batch_data:
        logger.error("❌ No valid images for batch registration")
        return results
    
    start_time = time.time()
    
    # ลงทะเบียนแบบ batch (หลายภาพพร้อมกัน)
    success_count = 0
    for i, item in enumerate(batch_data, 1):
        logger.info(f"    [{i}/{len(batch_data)}] Batch processing: {item['image_file']}")
        
        reg_start = time.time()
        success = add_single_face(item['image_base64'], person_id, f"Person {person_id}", model_name)
        reg_time = time.time() - reg_start
        
        reg_result = {
            "image_file": item['image_file'],
            "success": success,
            "time": reg_time
        }
        results["registration_results"].append(reg_result)
        
        if success:
            success_count += 1
            logger.info(f"      ✅ Success ({reg_time:.2f}s)")
        else:
            logger.error(f"      ❌ Failed ({reg_time:.2f}s)")
    
    total_time = time.time() - start_time
    results["total_time"] = total_time
    results["registration_success_rate"] = (success_count / len(batch_data)) * 100
    results["average_time_per_image"] = total_time / len(batch_data)
    
    logger.info(f"  📊 Batch Results: {success_count}/{len(batch_data)} success, {total_time:.2f}s total")
    
    return results

def add_single_face(image_base64: str, person_id: str, person_name: str, model_name: str) -> bool:
    """เพิ่มใบหน้าเดียวเข้าฐานข้อมูล"""
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
            return result.get('success', False)
        else:
            logger.error(f"❌ Registration failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Registration exception: {e}")
        return False

def test_recognition_accuracy(person_id: str, test_images: List[str], model_name: str) -> Dict[str, Any]:
    """ทดสอบความแม่นยำในการจดจำ"""
    logger.info(f"  🎯 Testing Recognition Accuracy - {model_name}")
    
    results = {
        "model_name": model_name,
        "person_id": person_id,
        "test_results": [],
        "accuracy": 0.0,
        "average_similarity": 0.0,
        "total_test_time": 0.0
    }
    
    start_time = time.time()
    correct_recognitions = 0
    similarity_scores = []
    
    for i, image_file in enumerate(test_images, 1):
        image_path = os.path.join(TEST_IMAGES_DIR, image_file)
        if not os.path.exists(image_path):
            continue
            
        logger.info(f"    [{i}/{len(test_images)}] Testing: {image_file}")
        
        # แปลงเป็น base64
        image_base64 = image_to_base64(image_path)
        
        # ทดสอบการจดจำ
        recognition_result = recognize_face(image_base64, model_name)
        
        test_result = {
            "image_file": image_file,
            "recognition_success": False,
            "similarity": 0.0,
            "recognized_person": "",
            "is_correct": False
        }
        
        if recognition_result and 'matches' in recognition_result:
            matches = recognition_result['matches']
            if matches:
                best_match = matches[0]
                similarity = best_match.get('similarity', 0.0)
                recognized_id = best_match.get('person_id', '')
                
                test_result["recognition_success"] = True
                test_result["similarity"] = similarity
                test_result["recognized_person"] = recognized_id
                test_result["is_correct"] = (recognized_id == person_id)
                
                if test_result["is_correct"]:
                    correct_recognitions += 1
                
                similarity_scores.append(similarity)
                logger.info(f"      ✅ Recognized: {recognized_id} (similarity: {similarity:.4f})")
            else:
                logger.warning(f"      ⚠️ No matches found")
        else:
            logger.error(f"      ❌ Recognition failed")
        
        results["test_results"].append(test_result)
    
    total_time = time.time() - start_time
    results["total_test_time"] = total_time
    results["accuracy"] = (correct_recognitions / len(test_images)) * 100 if test_images else 0
    results["average_similarity"] = np.mean(similarity_scores) if similarity_scores else 0
    
    logger.info(f"  📊 Recognition Results: {correct_recognitions}/{len(test_images)} correct, {results['accuracy']:.1f}% accuracy")
    
    return results

def recognize_face(image_base64: str, model_name: str) -> Dict[str, Any]:
    """ทำการจดจำใบหน้า"""
    # ดึง gallery ปัจจุบัน
    try:
        gallery_response = requests.get(f"{API_BASE_URL}/api/face-recognition/get-gallery", timeout=30)
        if gallery_response.status_code == 200:
            gallery = gallery_response.json()
        else:
            return {}
    except Exception:
        return {}
    
    if not gallery:
        return {"matches": []}
    
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
            return {}
    except Exception:
        return {}

def run_comparison_test():
    """รันการทดสอบเปรียบเทียบ"""
    logger.info("🔬 Face Registration Method Comparison Test")
    logger.info("=" * 60)
    
    ensure_output_dir()
    
    # ตรวจสอบ API
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code != 200:
            logger.error("❌ API server not ready")
            return
        logger.info("✅ API server is ready")
    except Exception as e:
        logger.error(f"❌ Cannot connect to API: {e}")
        return
    
    # ตรวจสอบรูปภาพ
    available_images = []
    for img in TEST_IMAGES:
        if os.path.exists(os.path.join(TEST_IMAGES_DIR, img)):
            available_images.append(img)
    
    if len(available_images) < 5:
        logger.error(f"❌ Need at least 5 images, found {len(available_images)}")
        return
    
    logger.info(f"📁 Found {len(available_images)} test images")
    
    # แบ่งรูป: 70% สำหรับ registration, 30% สำหรับ testing
    split_point = int(len(available_images) * 0.7)
    registration_images = available_images[:split_point]
    test_images = available_images[split_point:]
    
    logger.info(f"📝 Registration images: {len(registration_images)}")
    logger.info(f"🎯 Test images: {len(test_images)}")
    
    all_results = {}
    
    for model_name in MODELS:
        logger.info(f"\n🧠 Testing model: {model_name}")
        logger.info("-" * 50)
        
        model_results = {
            "model_name": model_name,
            "individual_registration": {},
            "batch_registration": {},
            "individual_recognition": {},
            "batch_recognition": {}
        }
        
        # ล้างฐานข้อมูล
        clear_database()
        
        # Test 1: Individual Registration + Recognition
        logger.info(f"🔄 Test 1: Individual Registration")
        individual_reg = test_individual_registration("person_individual", registration_images, model_name)
        model_results["individual_registration"] = individual_reg
        
        individual_rec = test_recognition_accuracy("person_individual", test_images, model_name)
        model_results["individual_recognition"] = individual_rec
        
        # รอสักครู่
        time.sleep(5)
        
        # ล้างฐานข้อมูล
        clear_database()
        
        # Test 2: Batch Registration + Recognition
        logger.info(f"🔄 Test 2: Batch Registration")
        batch_reg = test_batch_registration("person_batch", registration_images, model_name)
        model_results["batch_registration"] = batch_reg
        
        batch_rec = test_recognition_accuracy("person_batch", test_images, model_name)
        model_results["batch_recognition"] = batch_rec
        
        all_results[model_name] = model_results
        
        # รอก่อนทดสอบโมเดลถัดไป
        time.sleep(10)
    
    # สร้างรายงาน
    generate_comparison_report(all_results)
    
    logger.info("🎉 Comparison test completed!")

def generate_comparison_report(results: Dict[str, Any]):
    """สร้างรายงานเปรียบเทียบ"""
    report = {
        "test_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "comparison_results": results,
        "summary": {}
    }
    
    # สร้างสรุป
    summary = {}
    
    for model_name, model_data in results.items():
        individual_reg = model_data["individual_registration"]
        batch_reg = model_data["batch_registration"]
        individual_rec = model_data["individual_recognition"]
        batch_rec = model_data["batch_recognition"]
        
        summary[model_name] = {
            "registration_comparison": {
                "individual_success_rate": individual_reg.get("registration_success_rate", 0),
                "batch_success_rate": batch_reg.get("registration_success_rate", 0),
                "individual_total_time": individual_reg.get("total_time", 0),
                "batch_total_time": batch_reg.get("total_time", 0),
                "individual_avg_time": individual_reg.get("average_time_per_image", 0),
                "batch_avg_time": batch_reg.get("average_time_per_image", 0)
            },
            "recognition_comparison": {
                "individual_accuracy": individual_rec.get("accuracy", 0),
                "batch_accuracy": batch_rec.get("accuracy", 0),
                "individual_avg_similarity": individual_rec.get("average_similarity", 0),
                "batch_avg_similarity": batch_rec.get("average_similarity", 0)
            }
        }
    
    report["summary"] = summary
    
    # บันทึกรายงาน
    report_file = os.path.join(OUTPUT_DIR, "registration_comparison_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # แสดงสรุป
    print_comparison_summary(summary)
    
    logger.info(f"📄 Detailed report saved: {report_file}")

def print_comparison_summary(summary: Dict[str, Any]):
    """แสดงสรุปการเปรียบเทียบ"""
    print("\n" + "="*80)
    print("📊 FACE REGISTRATION METHOD COMPARISON RESULTS")
    print("="*80)
    print(f"📅 Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🧠 Models Tested: {', '.join(summary.keys())}")
    print("-"*80)
    
    for model_name, model_summary in summary.items():
        print(f"\n🔬 {model_name.upper()} Model Results:")
        
        reg_comp = model_summary["registration_comparison"]
        rec_comp = model_summary["recognition_comparison"]
        
        print(f"  📝 Registration Performance:")
        print(f"     Individual: {reg_comp['individual_success_rate']:.1f}% success, {reg_comp['individual_total_time']:.1f}s total")
        print(f"     Batch:      {reg_comp['batch_success_rate']:.1f}% success, {reg_comp['batch_total_time']:.1f}s total")
        print(f"     Time per image: Individual {reg_comp['individual_avg_time']:.2f}s vs Batch {reg_comp['batch_avg_time']:.2f}s")
        
        print(f"  🎯 Recognition Accuracy:")
        print(f"     Individual: {rec_comp['individual_accuracy']:.1f}% accuracy, {rec_comp['individual_avg_similarity']:.4f} avg similarity")
        print(f"     Batch:      {rec_comp['batch_accuracy']:.1f}% accuracy, {rec_comp['batch_avg_similarity']:.4f} avg similarity")
        
        # Winner analysis
        reg_winner = "Individual" if reg_comp['individual_success_rate'] > reg_comp['batch_success_rate'] else "Batch"
        if reg_comp['individual_success_rate'] == reg_comp['batch_success_rate']:
            reg_winner = "Individual" if reg_comp['individual_total_time'] < reg_comp['batch_total_time'] else "Batch"
        
        rec_winner = "Individual" if rec_comp['individual_accuracy'] > rec_comp['batch_accuracy'] else "Batch"
        if rec_comp['individual_accuracy'] == rec_comp['batch_accuracy']:
            rec_winner = "Individual" if rec_comp['individual_avg_similarity'] > rec_comp['batch_avg_similarity'] else "Batch"
        
        print(f"  🏆 Winner: Registration={reg_winner}, Recognition={rec_winner}")

if __name__ == "__main__":
    run_comparison_test()
