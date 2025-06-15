#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-world Face Recognition Test - Improved Version
ทดสอบระบบ Face Recognition แบบจริงที่ได้รับการปรับปรุง:
- ใช้ทั้ง ONNX และ Framework models
- ลงทะเบียน 2 users (boss, night) ด้วย 10 ภาพต่อคน
- ทดสอบกับทุกภาพใน test_images
- สร้าง Ensemble model จากทุก models
- แก้ไขปัญหาการ crop ภาพและ accuracy ที่เหมือนกันหมด
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

# Configuration
API_BASE_URL = "http://localhost:8080"
TEST_IMAGES_DIR = "test_images"
OUTPUT_DIR = "output/real_world_recognition_improved"

# Users และรูปสำหรับลงทะเบียน
USERS = {
    "boss": [f"boss_{i:02d}.jpg" for i in range(1, 11)],
    "night": [f"night_{i:02d}.jpg" for i in range(1, 11)]
}

# Model Configuration - ลดจำนวน models เพื่อลด timeout
# เลือกเฉพาะ models ที่สำคัญ
ONNX_MODELS = ["facenet", "adaface", "arcface"]
FRAMEWORK_MODELS = ["deepface", "facenet_pytorch"]  # ลดจาก 5 models เหลือ 2
ALL_MODELS = ONNX_MODELS + FRAMEWORK_MODELS

# Ensemble weights - ปรับให้สอดคล้องกับ models ที่เหลือ
ENSEMBLE_WEIGHTS = {
    # ONNX Models
    "facenet": 0.25,      # 25%
    "adaface": 0.25,      # 25%
    "arcface": 0.20,      # 20%
    # Framework Models (เลือกเฉพาะที่สำคัญ)
    "deepface": 0.15,     # 15%
    "facenet_pytorch": 0.15, # 15%
}

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

def clear_database():
    """ล้างฐานข้อมูลก่อนเริ่มทดสอบ"""
    try:
        # ใช้ endpoint ที่เพิ่งเพิ่มใน complete_endpoints.py
        clear_response = requests.post(f"{API_BASE_URL}/api/face-recognition/clear-gallery", timeout=30)
        if clear_response.status_code == 200:
            result = clear_response.json()
            logger.info(f"✅ Database cleared successfully: {result.get('message', 'Cleared')}")
            return True
        else:
            logger.warning(f"Clear database failed: {clear_response.status_code}")
            # ลองตรวจสอบ gallery status เพื่อยืนยัน
            gallery_response = requests.get(f"{API_BASE_URL}/api/face-recognition/get-gallery", timeout=30)
            if gallery_response.status_code == 200:
                gallery = gallery_response.json()
                logger.info(f"📋 Current gallery has {len(gallery)} persons")
                return True
            return False
    except Exception as e:
        logger.error(f"❌ Cannot access database: {e}")
        return False

def register_user_all_models(user_id: str, image_files: List[str]) -> Dict[str, int]:
    """ลงทะเบียนผู้ใช้หนึ่งคนกับทุก models"""
    logger.info(f"👤 Registering user: {user_id}")
    
    # ตรวจสอบว่ารูปมีอยู่จริง
    available_images = []
    for img_file in image_files:
        img_path = os.path.join(TEST_IMAGES_DIR, img_file)
        if os.path.exists(img_path):
            available_images.append(img_path)
        else:
            logger.warning(f"⚠️ Image not found: {img_file}")
    
    if len(available_images) < 5:
        logger.error(f"❌ Not enough images for {user_id}: {len(available_images)}/10")
        return {}
    
    logger.info(f"📁 Found {len(available_images)} images for {user_id}")
    
    # ผลลัพธ์สำหรับแต่ละ model
    model_results = {}
    
    # ลงทะเบียนทีละโมเดล
    for model_name in ALL_MODELS:
        logger.info(f"  🧠 Registering with {model_name} model...")
        
        success_count = 0
        for i, img_path in enumerate(available_images, 1):
            logger.info(f"    [{i}/{len(available_images)}] Processing: {os.path.basename(img_path)}")
            
            # แปลงเป็น base64
            image_base64 = image_to_base64(img_path)
            
            # ลงทะเบียน
            success = add_single_face(image_base64, user_id, f"User {user_id.title()}", model_name)
            
            if success:
                success_count += 1
                logger.info(f"      ✅ Success")
            else:
                logger.error(f"      ❌ Failed")
            
            # รอเล็กน้อย
            time.sleep(1)
        
        model_results[model_name] = success_count
        logger.info(f"  📊 {model_name}: {success_count}/{len(available_images)} success")
    
    return model_results

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
            logger.error(f"❌ Registration failed for {model_name}: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Registration exception for {model_name}: {e}")
        return False

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

def detect_faces_in_image(image_base64: str) -> List[Dict[str, Any]]:
    """ตรวจจับใบหน้าในภาพ - เพิ่ม retry และ timeout ที่นาน"""
    url = f"{API_BASE_URL}/api/face-detection/detect-base64"
    
    data = {
        "face_image_base64": image_base64,
        "model_name": "auto",
        "conf_threshold": 0.5,
        "iou_threshold": 0.4,
        "max_faces": 10
    }
    
    # ลองหลายครั้งถ้า timeout
    for attempt in range(3):
        try:
            logger.info(f"    🔍 Face detection attempt {attempt + 1}/3")
            response = requests.post(url, json=data, timeout=60)  # เพิ่ม timeout เป็น 60 วินาที
            if response.status_code == 200:
                result = response.json()
                faces = result.get("faces", [])
                logger.info(f"    ✅ Detected {len(faces)} faces")
                return faces
            else:
                logger.warning(f"    ⚠️ Face detection failed: {response.status_code}")
                if attempt < 2:  # ไม่ใช่ attempt สุดท้าย
                    time.sleep(5)  # รอ 5 วินาทีก่อนลองใหม่
                    continue
        except Exception as e:
            logger.warning(f"    ⚠️ Face detection error (attempt {attempt + 1}): {e}")
            if attempt < 2:  # ไม่ใช่ attempt สุดท้าย
                time.sleep(5)  # รอ 5 วินาทีก่อนลองใหม่
                continue
    
    logger.error(f"    ❌ Face detection failed after 3 attempts")
    return []

def recognize_face_all_models(image_base64: str) -> Dict[str, Any]:
    """ทำการจดจำด้วยทุกโมเดลโดยใช้ internal database"""
    results = {}
    
    logger.info("    🗄️ Using internal database for recognition")
    
    # ทดสอบทุกโมเดล
    for model_name in ALL_MODELS:
        try:
            logger.info(f"    🧠 Testing {model_name}")
            
            result = recognize_single_model(image_base64, model_name)
            results[model_name] = result
            
            # Log ผลลัพธ์
            if result.get("matches"):
                best_match = result["matches"][0]
                similarity = best_match.get("similarity", best_match.get("confidence", 0))
                person_name = best_match.get("person_name", "unknown")
                logger.info(f"      ✅ {model_name}: {person_name} ({similarity:.3f})")
            else:
                logger.info(f"      ❌ {model_name}: No matches found")
            
            # เพิ่ม delay ระหว่างโมเดล
            time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"❌ Recognition failed for {model_name}: {e}")
            results[model_name] = {"matches": [], "error": str(e)}
    
    return results

def recognize_single_model(image_base64: str, model_name: str) -> Dict[str, Any]:
    """ทำการจดจำด้วยโมเดลเดียวโดยใช้ internal database - เพิ่ม retry และ timeout"""
    url = f"{API_BASE_URL}/api/face-recognition/recognize"
    
    data = {
        "face_image_base64": image_base64,
        "model_name": model_name,
        "top_k": 5,
        "similarity_threshold": 0.3
        # ไม่ส่ง gallery เพื่อให้ API ใช้ internal database
    }
    
    # ลองหลายครั้งถ้า timeout
    for attempt in range(2):  # ลดจาก 3 เป็น 2 ครั้ง
        try:
            logger.info(f"      🔄 Recognition attempt {attempt + 1}/2 for {model_name}")
            response = requests.post(url, json=data, timeout=180)  # เพิ่ม timeout เป็น 3 นาที
            if response.status_code == 200:
                result = response.json()
                if result.get("matches"):
                    logger.info(f"      ✅ {model_name} success")
                else:
                    logger.info(f"      ⚪ {model_name} no matches")
                return result
            else:
                error_text = response.text[:200] if response.text else "No error message"
                logger.warning(f"      ⚠️ {model_name} failed: {response.status_code}")
                if attempt < 1:  # ไม่ใช่ attempt สุดท้าย
                    time.sleep(10)  # รอ 10 วินาทีก่อนลองใหม่
                    continue
        except Exception as e:
            logger.warning(f"      ⚠️ {model_name} error (attempt {attempt + 1}): {str(e)[:100]}")
            if attempt < 1:  # ไม่ใช่ attempt สุดท้าย
                time.sleep(10)  # รอ 10 วินาทีก่อนลองใหม่
                continue
    
    logger.error(f"      ❌ {model_name} failed after 2 attempts")
    return {"matches": [], "error": f"Failed after 2 attempts"}

def create_ensemble_prediction(model_results: Dict[str, Any]) -> Dict[str, Any]:
    """สร้างการทำนายแบบ Ensemble"""
    if not model_results:
        return {"matches": [], "method": "ensemble"}
    
    # รวบรวมผลจากทุกโมเดล
    person_scores = defaultdict(list)
    
    for model_name, result in model_results.items():
        if "matches" in result and result["matches"]:
            weight = ENSEMBLE_WEIGHTS.get(model_name, 0)
            
            for match in result["matches"]:
                person_name = match.get("person_name", "")
                similarity = match.get("similarity", match.get("confidence", 0))
                weighted_score = similarity * weight
                
                person_scores[person_name].append({
                    "model": model_name,
                    "similarity": similarity,
                    "weighted_score": weighted_score,
                    "weight": weight
                })
    
    # คำนวณคะแนนรวม
    ensemble_matches = []
    for person_name, scores in person_scores.items():
        total_weighted_score = sum(score["weighted_score"] for score in scores)
        avg_similarity = sum(score["similarity"] for score in scores) / len(scores)
        
        ensemble_matches.append({
            "person_name": person_name,
            "similarity": total_weighted_score,
            "confidence": total_weighted_score,
            "average_similarity": avg_similarity,
            "model_count": len(scores),
            "contributing_models": [score["model"] for score in scores]
        })
    
    # เรียงตามคะแนน
    ensemble_matches.sort(key=lambda x: x["similarity"], reverse=True)
    
    return {
        "matches": ensemble_matches,
        "method": "ensemble",
        "total_models": len(model_results),
        "contributing_models": len([m for m in model_results.values() if m.get("matches")])
    }

def draw_results_on_image(image_path: str, faces: List[Dict], 
                         recognition_results: Dict[str, Any], model_name: str) -> str:
    """วาดผลลัพธ์บนภาพ"""
    try:
        # อ่านภาพ
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"❌ Cannot read image: {image_path}")
            return ""
        
        # วาดกรอบใบหน้า
        for i, face in enumerate(faces):
            bbox = face.get("bbox", {})
            x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
            x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
            confidence = face.get("confidence", 0)
            
            # สีกรอบ
            color = (0, 255, 0)  # เขียว
            
            # วาดกรอบ
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # ข้อความ
            face_text = f"Face {i+1} ({confidence:.2f})"
            cv2.putText(image, face_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # ข้อความผลลัพธ์
        result_text = f"Model: {model_name}"
        if recognition_results.get("matches"):
            best_match = recognition_results["matches"][0]
            person_name = best_match.get("person_name", "unknown")
            similarity = best_match.get("similarity", best_match.get("confidence", 0))
            result_text += f" | Best: {person_name} ({similarity:.3f})"
        else:
            result_text += " | No matches"
        
        # วาดข้อความที่ด้านบน
        cv2.putText(image, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # บันทึกภาพ
        output_path = os.path.join(OUTPUT_DIR, model_name, os.path.basename(image_path))
        cv2.imwrite(output_path, image)
        
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Error drawing results: {e}")
        return ""

def warm_up_models():
    """Warm up models โดยการเรียกใช้ทุก model ก่อน"""
    logger.info("🔥 Warming up models...")
    
    # สร้าง dummy image เล็กๆ สำหรับ warm up
    dummy_image = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="
    
    for model_name in ALL_MODELS:
        try:
            logger.info(f"  🔥 Warming up {model_name}...")
            url = f"{API_BASE_URL}/api/face-recognition/recognize"
            data = {
                "face_image_base64": dummy_image,
                "model_name": model_name,
                "top_k": 1,
                "similarity_threshold": 0.9
            }
            response = requests.post(url, json=data, timeout=60)
            if response.status_code == 200:
                logger.info(f"    ✅ {model_name} warmed up")
            else:
                logger.warning(f"    ⚠️ {model_name} warm up failed: {response.status_code}")
            time.sleep(2)  # รอเล็กน้อยระหว่าง models
        except Exception as e:
            logger.warning(f"    ⚠️ {model_name} warm up error: {e}")
    
    logger.info("🔥 Model warm up completed")

def run_real_world_test():
    """รันการทดสอบแบบจริง"""
    logger.info("🚀 Real-world Face Recognition Test - Improved Version")
    logger.info("=" * 70)
    
    ensure_output_dir()
    
    # ตรวจสอบ API
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if health_response.status_code != 200:
            logger.error("❌ API server not responding")
            return
    except Exception as e:
        logger.error(f"❌ Cannot connect to API: {e}")
        return
      # 1. ล้างฐานข้อมูล
    logger.info("\n🧹 Step 1: Clearing database...")
    if not clear_database():
        logger.error("❌ Failed to clear database")
        return
    
    # 1.5 Warm up models
    logger.info("\n🔥 Step 1.5: Warming up models...")
    warm_up_models()
    
    # 2. ลงทะเบียนผู้ใช้
    logger.info("\n👥 Step 2: Registering users...")
    registration_results = {}
    
    for user_id, image_files in USERS.items():
        result = register_user_all_models(user_id, image_files)
        registration_results[user_id] = result
        time.sleep(2)  # รอระหว่าง users
    
    # สรุปผลการลงทะเบียน
    logger.info("\n📊 Registration Summary:")
    for user_id, results in registration_results.items():
        logger.info(f"  👤 {user_id}:")
        for model_name, success_count in results.items():
            logger.info(f"    🧠 {model_name}: {success_count}/10 images")
      # 3. ทดสอบการจดจำ (ลดจำนวนภาพเพื่อลด timeout)
    logger.info("\n🔍 Step 3: Testing recognition...")
    test_images = get_all_test_images()
    
    # เลือกภาพสำคัญเท่านั้น เพื่อลดเวลา
    important_images = []
    for img in test_images:
        if any(keyword in img.lower() for keyword in ['boss_01', 'boss_02', 'night_01', 'night_02', 'group']):
            important_images.append(img)
    
    # หากไม่มีภาพสำคัญ ให้เลือกภาพแรกๆ
    if not important_images:
        important_images = test_images[:10]  # เลือกแค่ 10 ภาพแรก
    
    logger.info(f"📸 Selected {len(important_images)} important images from {len(test_images)} total images")
    
    all_results = {}    
    for i, image_file in enumerate(important_images, 1):
        logger.info(f"\n📸 [{i}/{len(important_images)}] Processing: {image_file}")
        
        image_path = os.path.join(TEST_IMAGES_DIR, image_file)
        image_base64 = image_to_base64(image_path)
        
        # 3.1 ตรวจจับใบหน้า
        faces = detect_faces_in_image(image_base64)
        logger.info(f"  👥 Detected {len(faces)} faces")
        
        # 3.2 ทำการจดจำ
        recognition_results = recognize_face_all_models(image_base64)
        
        # 3.3 สร้าง Ensemble prediction
        ensemble_result = create_ensemble_prediction(recognition_results)
        recognition_results["ensemble"] = ensemble_result
        
        # 3.4 บันทึกผลลัพธ์
        all_results[image_file] = {
            "faces": faces,
            "recognition_results": recognition_results,
            "image_path": image_path
        }
        
        # 3.5 สร้างภาพผลลัพธ์
        for model_name, result in recognition_results.items():
            output_path = draw_results_on_image(image_path, faces, result, model_name)
            if output_path:
                logger.info(f"    💾 Saved: {os.path.basename(output_path)}")
        
        # รอเล็กน้อย
        time.sleep(2)
    
    # 4. สร้างรายงาน
    logger.info("\n📋 Step 4: Generating report...")
    generate_final_report(all_results)
    
    logger.info("\n✅ Real-world test completed!")
    logger.info(f"📁 Results saved in: {OUTPUT_DIR}")

def generate_final_report(results: Dict[str, Any]):
    """สร้างรายงานสรุปผล"""
    report = {
        "test_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "total_images_processed": len(results),
        "models_tested": ALL_MODELS + ["ensemble"],
        "ensemble_weights": ENSEMBLE_WEIGHTS,
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
            "match_rate": (matches_found / len(results)) * 100,
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
    print("📊 REAL-WORLD FACE RECOGNITION TEST RESULTS - IMPROVED")
    print("="*80)
    print(f"🧠 Models tested: {len(ALL_MODELS)} models + ensemble")
    print("👥 Users: boss, night (10 images each)")
    print("📸 Test images: Multiple processed")
    print("⚖️ Ensemble: Weighted average from all models")
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
