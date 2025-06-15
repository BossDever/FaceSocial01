#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-world Face Recognition Test - Fixed Version
ทดสอบระบบ Face Recognition แบบจริงที่แก้ไข timeout และปรับปรุงประสิทธิภาพ
"""

import os
import json
import time
import base64
import requests
import numpy as np
import cv2
from typing import Dict, List, Any, Tuple, Optional
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration - เพิ่ม timeout และปรับการตั้งค่า
API_BASE_URL = "http://localhost:8080"
TEST_IMAGES_DIR = "test_images"
OUTPUT_DIR = "output/real_world_recognition_fixed"

# Timeout settings - เพิ่ม timeout สำหรับการประมวลผลที่ใช้เวลานาน
TIMEOUTS = {
    "health_check": 30,
    "registration": 300,    # 5 นาที สำหรับการลงทะเบียน
    "recognition": 180,     # 3 นาที สำหรับการจดจำ
    "detection": 120,       # 2 นาที สำหรับการตรวจจับ
    "clear_database": 60,   # 1 นาที สำหรับการล้างฐานข้อมูล
    "warm_up": 120         # 2 นาที สำหรับ warm up
}

# Users และรูปสำหรับลงทะเบียน - ลดจำนวนภาพเพื่อลดเวลา
USERS = {
    "boss": [f"boss_{i:02d}.jpg" for i in range(1, 6)],  # ลดเหลือ 5 ภาพ
    "night": [f"night_{i:02d}.jpg" for i in range(1, 6)]  # ลดเหลือ 5 ภาพ
}

# Model Configuration - เลือกเฉพาะ models ที่สำคัญ
PRIORITY_MODELS = ["facenet", "adaface"]  # ลดเหลือ 2 models หลัก
OPTIONAL_MODELS = ["arcface"]  # models เสริม
ALL_MODELS = PRIORITY_MODELS + OPTIONAL_MODELS

def ensure_output_dir():
    """สร้างโฟลเดอร์ output"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for model in ALL_MODELS + ["ensemble"]:
        model_dir = os.path.join(OUTPUT_DIR, model)
        os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")

def image_to_base64(image_path: str) -> str:
    """แปลงรูปภาพเป็น base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def wait_for_api_ready(max_attempts: int = 10) -> bool:
    """รอให้ API พร้อมใช้งาน"""
    logger.info("🔄 Waiting for API to be ready...")
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(
                f"{API_BASE_URL}/health", 
                timeout=TIMEOUTS["health_check"]
            )
            if response.status_code == 200:
                logger.info("✅ API is ready!")
                return True
                
        except Exception as e:
            logger.info(f"⏳ Attempt {attempt + 1}/{max_attempts}: API not ready yet ({e})")
            if attempt < max_attempts - 1:
                time.sleep(10)  # รอ 10 วินาที
    
    logger.error("❌ API not ready after maximum attempts")
    return False

def clear_database():
    """ล้างฐานข้อมูลก่อนเริ่มทดสอบ"""
    try:
        logger.info("🧹 Clearing database...")
        clear_response = requests.post(
            f"{API_BASE_URL}/api/face-recognition/clear-gallery", 
            timeout=TIMEOUTS["clear_database"]
        )
        if clear_response.status_code == 200:
            result = clear_response.json()
            logger.info(f"✅ Database cleared successfully: {result.get('message', 'Cleared')}")
            return True
        else:
            logger.warning(f"⚠️ Clear database failed: {clear_response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Cannot clear database: {e}")
        return False

def warm_up_models():
    """Warm up models โดยการเรียกใช้ทุก model ก่อน"""
    logger.info("🔥 Warming up models...")
    
    # สร้าง dummy image เล็กๆ สำหรับ warm up
    dummy_image = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
    
    for i, model_name in enumerate(ALL_MODELS):
        try:
            logger.info(f"  🔥 Warming up {model_name} ({i+1}/{len(ALL_MODELS)})...")
            url = f"{API_BASE_URL}/api/face-recognition/recognize"
            data = {
                "face_image_base64": dummy_image,
                "model_name": model_name,
                "top_k": 1,
                "similarity_threshold": 0.9
            }
            response = requests.post(url, json=data, timeout=TIMEOUTS["warm_up"])
            if response.status_code == 200:
                logger.info(f"    ✅ {model_name} warmed up successfully")
            else:
                logger.warning(f"    ⚠️ {model_name} warm up failed: {response.status_code}")
        except Exception as e:
            logger.warning(f"    ⚠️ {model_name} warm up error: {e}")
        
        # รอระหว่าง models เพื่อไม่ให้ overload
        time.sleep(5)
    
    logger.info("🔥 Model warm up completed")

def register_user_sequential(user_id: str, image_files: List[str]) -> Dict[str, int]:
    """ลงทะเบียนผู้ใช้หนึ่งคนแบบทีละโมเดล"""
    logger.info(f"👤 Registering user: {user_id}")
    
    # ตรวจสอบว่ารูปมีอยู่จริง
    available_images = []
    for img_file in image_files:
        img_path = os.path.join(TEST_IMAGES_DIR, img_file)
        if os.path.exists(img_path):
            available_images.append(img_path)
        else:
            logger.warning(f"⚠️ Image not found: {img_file}")
    
    if len(available_images) < 3:
        logger.error(f"❌ Not enough images for {user_id}: {len(available_images)}")
        return {}
    
    logger.info(f"📁 Found {len(available_images)} images for {user_id}")
    
    model_results = {}
    
    # ลงทะเบียนทีละโมเดล (sequential)
    for model_name in ALL_MODELS:
        logger.info(f"  🧠 Registering with {model_name} model...")
        success_count = 0
        
        for i, img_path in enumerate(available_images, 1):
            logger.info(f"    [{i}/{len(available_images)}] Processing: {os.path.basename(img_path)}")
            
            try:
                # แปลงเป็น base64
                image_base64 = image_to_base64(img_path)
                
                # ลงทะเบียน
                success = add_single_face_with_retry(image_base64, user_id, f"User {user_id.title()}", model_name)
                
                if success:
                    success_count += 1
                    logger.info(f"      ✅ Success")
                else:
                    logger.error(f"      ❌ Failed")
                
                # รอระหว่างการลงทะเบียนแต่ละภาพ
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"      ❌ Exception: {e}")
        
        model_results[model_name] = success_count
        logger.info(f"  📊 {model_name}: {success_count}/{len(available_images)} success")
        
        # รอระหว่างโมเดล
        time.sleep(3)
    
    return model_results

def add_single_face_with_retry(image_base64: str, person_id: str, person_name: str, model_name: str, max_retries: int = 3) -> bool:
    """เพิ่มใบหน้าเดียวเข้าฐานข้อมูลพร้อม retry"""
    url = f"{API_BASE_URL}/api/face-recognition/add-face-json"
    
    data = {
        "person_id": person_id,
        "person_name": person_name,
        "face_image_base64": image_base64,
        "model_name": model_name
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data, timeout=TIMEOUTS["registration"])
            if response.status_code == 200:
                result = response.json()
                return result.get('success', False)
            else:
                logger.warning(f"      ⚠️ Attempt {attempt + 1}: Status {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    
        except Exception as e:
            logger.warning(f"      ⚠️ Attempt {attempt + 1}: Exception {e}")
            if attempt < max_retries - 1:
                time.sleep(10)
    
    logger.error(f"      ❌ Failed after {max_retries} attempts")
    return False

def detect_faces_in_image(image_base64: str) -> List[Dict[str, Any]]:
    """ตรวจจับใบหน้าในภาพ"""
    url = f"{API_BASE_URL}/api/face-detection/detect-base64"
    
    data = {
        "image_base64": image_base64,
        "model_name": "auto",
        "conf_threshold": 0.5,
        "iou_threshold": 0.4,
        "max_faces": 10
    }
    
    try:
        logger.info(f"    🔍 Face detection...")
        response = requests.post(url, json=data, timeout=TIMEOUTS["detection"])
        if response.status_code == 200:
            result = response.json()
            faces = result.get("faces", [])
            logger.info(f"    ✅ Detected {len(faces)} faces")
            return faces
        else:
            logger.warning(f"    ⚠️ Face detection failed: {response.status_code}")
            
    except Exception as e:
        logger.warning(f"    ⚠️ Face detection error: {e}")
    
    return []

def recognize_face_sequential(image_base64: str) -> Dict[str, Any]:
    """ทำการจดจำแบบทีละโมเดล"""
    results = {}
    
    logger.info("    🗄️ Using internal database for recognition")
    
    # ทดสอบทีละโมเดล (sequential)
    for model_name in ALL_MODELS:
        try:
            logger.info(f"    🧠 Testing {model_name}")
            
            result = recognize_single_model_with_retry(image_base64, model_name)
            results[model_name] = result
            
            # Log ผลลัพธ์
            if result.get("matches"):
                best_match = result["matches"][0]
                similarity = best_match.get("similarity", best_match.get("confidence", 0))
                person_name = best_match.get("person_name", "unknown")
                logger.info(f"      ✅ {model_name}: {person_name} ({similarity:.3f})")
            else:
                logger.info(f"      ❌ {model_name}: No matches found")
            
            # รอระหว่างโมเดล
            time.sleep(2)
                
        except Exception as e:
            logger.error(f"❌ Recognition failed for {model_name}: {e}")
            results[model_name] = {"matches": [], "error": str(e)}
    
    return results

def recognize_single_model_with_retry(image_base64: str, model_name: str, max_retries: int = 2) -> Dict[str, Any]:
    """ทำการจดจำด้วยโมเดลเดียวพร้อม retry"""
    url = f"{API_BASE_URL}/api/face-recognition/recognize"
    
    data = {
        "face_image_base64": image_base64,
        "model_name": model_name,
        "top_k": 5,
        "similarity_threshold": 0.3
    }
    
    for attempt in range(max_retries):
        try:
            logger.info(f"      🔄 Recognition attempt {attempt + 1}/{max_retries} for {model_name}")
            response = requests.post(url, json=data, timeout=TIMEOUTS["recognition"])
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                logger.warning(f"      ⚠️ {model_name} failed: {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(15)
                    
        except Exception as e:
            logger.warning(f"      ⚠️ {model_name} error (attempt {attempt + 1}): {str(e)[:100]}")
            if attempt < max_retries - 1:
                time.sleep(15)
    
    logger.error(f"      ❌ {model_name} failed after {max_retries} attempts")
    return {"matches": [], "error": f"Failed after {max_retries} attempts"}

def get_all_test_images() -> List[str]:
    """ดึงรายการรูปทั้งหมดใน test_images"""
    if not os.path.exists(TEST_IMAGES_DIR):
        logger.error(f"❌ Test images directory not found: {TEST_IMAGES_DIR}")
        return []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    test_images = []
    
    for file in os.listdir(TEST_IMAGES_DIR):
        if os.path.splitext(file.lower())[1] in image_extensions:
            test_images.append(file)
    
    test_images.sort()
    logger.info(f"📁 Found {len(test_images)} test images")
    
    return test_images

def create_ensemble_prediction(model_results: Dict[str, Any]) -> Dict[str, Any]:
    """สร้างการทำนายแบบ Ensemble (แบบง่าย)"""
    if not model_results:
        return {"matches": [], "method": "ensemble"}
    
    # รวบรวมผลจากทุกโมเดล
    person_scores = defaultdict(list)
    
    for model_name, result in model_results.items():
        if "matches" in result and result["matches"]:
            for match in result["matches"]:
                person_name = match.get("person_name", "")
                similarity = match.get("similarity", match.get("confidence", 0))
                person_scores[person_name].append(similarity)
    
    # คำนวณคะแนนเฉลี่ย
    ensemble_matches = []
    for person_name, scores in person_scores.items():
        avg_similarity = sum(scores) / len(scores)
        ensemble_matches.append({
            "person_name": person_name,
            "similarity": avg_similarity,
            "confidence": avg_similarity,
            "model_count": len(scores)
        })
    
    # เรียงตามคะแนน
    ensemble_matches.sort(key=lambda x: x["similarity"], reverse=True)
    
    return {
        "matches": ensemble_matches,
        "method": "ensemble"
    }

def run_real_world_test():
    """รันการทดสอบแบบจริง - Fixed Version"""
    logger.info("🚀 Real-world Face Recognition Test - Fixed Version")
    logger.info("=" * 70)
    
    ensure_output_dir()
    
    # 1. ตรวจสอบ API และรอให้พร้อม
    logger.info("\n⏳ Step 1: Waiting for API to be ready...")
    if not wait_for_api_ready():
        logger.error("❌ API not ready, stopping test")
        return
    
    # 2. ล้างฐานข้อมูล
    logger.info("\n🧹 Step 2: Clearing database...")
    if not clear_database():
        logger.warning("⚠️ Failed to clear database, continuing anyway...")
    
    # 3. Warm up models
    logger.info("\n🔥 Step 3: Warming up models...")
    warm_up_models()
    
    # 4. ลงทะเบียนผู้ใช้
    logger.info("\n👥 Step 4: Registering users...")
    registration_results = {}
    
    for user_id, image_files in USERS.items():
        result = register_user_sequential(user_id, image_files)
        registration_results[user_id] = result
        time.sleep(5)  # รอระหว่าง users
    
    # สรุปผลการลงทะเบียน
    logger.info("\n📊 Registration Summary:")
    for user_id, results in registration_results.items():
        logger.info(f"  👤 {user_id}:")
        for model_name, success_count in results.items():
            total_images = len(USERS[user_id])
            logger.info(f"    🧠 {model_name}: {success_count}/{total_images} images")
    
    # 5. ทดสอบการจดจำ (เลือกภาพสำคัญ)
    logger.info("\n🔍 Step 5: Testing recognition...")
    test_images = get_all_test_images()
    
    # เลือกภาพสำคัญเท่านั้น
    important_images = []
    for img in test_images:
        if any(keyword in img.lower() for keyword in ['boss_01', 'boss_02', 'night_01', 'night_02']):
            important_images.append(img)
    
    # หากไม่มีภาพสำคัญ ให้เลือกภาพแรกๆ
    if not important_images:
        important_images = test_images[:5]  # เลือกแค่ 5 ภาพแรก
    
    logger.info(f"📸 Selected {len(important_images)} important images")
    
    all_results = {}
    for i, image_file in enumerate(important_images, 1):
        logger.info(f"\n📸 [{i}/{len(important_images)}] Processing: {image_file}")
        
        image_path = os.path.join(TEST_IMAGES_DIR, image_file)
        image_base64 = image_to_base64(image_path)
        
        # 5.1 ตรวจจับใบหน้า
        faces = detect_faces_in_image(image_base64)
        logger.info(f"  👥 Detected {len(faces)} faces")
        
        # 5.2 ทำการจดจำ
        recognition_results = recognize_face_sequential(image_base64)
        
        # 5.3 สร้าง Ensemble prediction
        ensemble_result = create_ensemble_prediction(recognition_results)
        recognition_results["ensemble"] = ensemble_result
        
        # 5.4 บันทึกผลลัพธ์
        all_results[image_file] = {
            "faces": faces,
            "recognition_results": recognition_results,
            "image_path": image_path
        }
        
        # รอระหว่างภาพ
        time.sleep(3)
    
    # 6. สร้างรายงาน
    logger.info("\n📋 Step 6: Generating report...")
    generate_final_report(all_results)
    
    logger.info("\n✅ Real-world test completed successfully!")
    logger.info(f"📁 Results saved in: {OUTPUT_DIR}")

def generate_final_report(results: Dict[str, Any]):
    """สร้างรายงานสรุปผล"""
    report = {
        "test_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "total_images_processed": len(results),
        "models_tested": ALL_MODELS + ["ensemble"],
        "timeout_settings": TIMEOUTS,
        "results_summary": {},
        "detailed_results": {}
    }
    
    # สถิติสำหรับแต่ละโมเดล
    model_stats = {}
    for model_name in ALL_MODELS + ["ensemble"]:
        matches_found = 0
        total_similarity = 0
        person_counts = defaultdict(int)
        
        for image_file, data in results.items():
            recognition_result = data["recognition_results"].get(model_name, {})
            if recognition_result.get("matches"):
                matches_found += 1
                best_match = recognition_result["matches"][0]
                similarity = best_match.get("similarity", best_match.get("confidence", 0))
                total_similarity += similarity
                
                person_name = best_match.get("person_name", "unknown")
                person_counts[person_name] += 1
        
        model_stats[model_name] = {
            "matches_found": matches_found,
            "match_rate": (matches_found / len(results)) * 100 if results else 0,
            "average_similarity": (total_similarity / matches_found) if matches_found > 0 else 0,
            "person_distribution": dict(person_counts)
        }
    
    report["results_summary"] = model_stats
    
    # บันทึกรายงาน
    report_path = os.path.join(OUTPUT_DIR, "test_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 Report saved: {report_path}")
    
    # แสดงสรุปในคอนโซล
    print_final_summary(model_stats)

def print_final_summary(stats: Dict[str, Any]):
    """แสดงสรุปผลการทดสอบ"""
    print("\n" + "="*80)
    print("📊 REAL-WORLD FACE RECOGNITION TEST RESULTS - FIXED VERSION")
    print("="*80)
    print(f"🧠 Models tested: {len(ALL_MODELS)} models + ensemble")
    print("👥 Users: boss, night (5 images each)")
    print("📸 Test images: Selected important images")
    print("⚖️ Ensemble: Average from all models")
    print("\n" + "-"*80)
    
    # เรียงตาม match rate
    sorted_models = sorted(stats.items(), key=lambda x: x[1]["match_rate"], reverse=True)
    
    for model_name, data in sorted_models:
        match_rate = data["match_rate"]
        avg_similarity = data["average_similarity"]
        matches_found = data["matches_found"]
        person_dist = data["person_distribution"]
        
        status = "🟢" if match_rate >= 80 else "🟡" if match_rate >= 50 else "🔴"
        
        print(f"{status} {model_name:15} | {match_rate:5.1f}% | {avg_similarity:.3f} | {matches_found:3d} matches")
        
        # แสดงการกระจายของ person
        if person_dist:
            person_summary = ", ".join([f"{name}: {count}" for name, count in person_dist.items()])
            print(f"   📊 Distribution: {person_summary}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    run_real_world_test()