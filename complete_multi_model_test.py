#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Multi-Model Face Recognition Test
ทดสอบระบบ Face Recognition กับทุกโมเดลที่มี:
- ONNX Models: FaceNet, AdaFace, ArcFace (โมเดลของคุณ)
- Framework Models: DeepFace, FaceNet-PyTorch, Dlib, InsightFace
- เปรียบเทียบประสิทธิภาพของทุกโมเดล
- สร้าง Ensemble prediction จากทุกโมเดล
"""

import os
import json
import time
import base64
import requests
import numpy as np
import cv2
from typing import Dict, List, Any, Tuple
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8080"
TEST_IMAGES_DIR = "test_images"
OUTPUT_DIR = "output/complete_multi_model_test"

# Users และรูปสำหรับลงทะเบียน
USERS = {
    "boss": [f"boss_{i:02d}.jpg" for i in range(1, 11)],
    "night": [f"night_{i:02d}.jpg" for i in range(1, 11)]
}

# โมเดล ONNX ของคุณ (Custom Models)
ONNX_MODELS = ["facenet", "adaface", "arcface"]

# Multi-Framework Libraries - ตรวจสอบว่าเซิฟเวอร์รองรับหรือไม่
FRAMEWORK_MODELS = []

# รวมทุกโมเดล (จะถูกตั้งค่าใน initialize_available_models)
ALL_MODELS = []

# Ensemble weights - จะถูกปรับตามโมเดลที่มี
ENSEMBLE_WEIGHTS = {}

def check_server_availability():
    """ตรวจสอบว่าเซิฟเวอร์พร้อมใช้งาน"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            logger.info("✅ API server is ready")
            return True
        else:
            logger.error(f"❌ API server responded with {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Cannot connect to API server: {e}")
        return False

def get_available_models():
    """ดึงรายการโมเดลที่เซิฟเวอร์รองรับ"""
    try:
        # ลองเรียก API เพื่อดูว่ามีโมเดลไหนบ้าง
        response = requests.get(f"{API_BASE_URL}/api/face-recognition/models/info", timeout=10)
        if response.status_code == 200:
            models_info = response.json()
            available_models = models_info.get("available_models", ONNX_MODELS)
            logger.info(f"✅ Server supports models: {available_models}")
            return available_models
        else:
            logger.warning(f"⚠️ Cannot get models info, using default ONNX models")
            return ONNX_MODELS
    except Exception as e:
        logger.warning(f"⚠️ Cannot get models info: {e}, using default ONNX models")
        return ONNX_MODELS

def test_model_recognition(test_image_path: str, model_name: str):
    """ทดสอบการทำงานของโมเดลหนึ่งๆ"""
    try:
        # แปลงรูปเป็น base64
        image_base64 = image_to_base64(test_image_path)
        
        # สร้าง gallery ง่ายๆ เพื่อทดสอบ
        test_gallery = {
            "test_person": [
                {
                    "embedding": [0.1] * 512,  # dummy embedding
                    "person_name": "Test Person"
                }
            ]
        }
        
        # ทดสอบ recognition
        url = f"{API_BASE_URL}/api/face-recognition/recognize"
        data = {
            "face_image_base64": image_base64,
            "gallery": test_gallery,
            "model_name": model_name,
            "top_k": 1,
            "similarity_threshold": 0.1
        }
        
        response = requests.post(url, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if "matches" in result:
                logger.info(f"  ✅ {model_name}: Working")
                return True
            else:
                logger.warning(f"  ⚠️ {model_name}: No matches returned")
                return False
        else:
            logger.error(f"  ❌ {model_name}: HTTP {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"  ❌ {model_name}: Error - {e}")
        return False

def initialize_available_models():
    """เริ่มต้นและตรวจสอบโมเดลที่ใช้งานได้"""
    global ALL_MODELS, ENSEMBLE_WEIGHTS, FRAMEWORK_MODELS
    
    logger.info("🔍 Checking available models...")
    
    # ดึงโมเดลจากเซิฟเวอร์
    server_models = get_available_models()
    
    # ตรวจสอบ ONNX models ของคุณ
    available_onnx = []
    for model in ONNX_MODELS:
        if model in server_models:
            available_onnx.append(model)
            logger.info(f"  ✅ ONNX Model: {model}")
        else:
            logger.warning(f"  ❌ ONNX Model not available: {model}")
    
    # ตรวจสอบ framework models
    potential_frameworks = ["deepface", "facenet_pytorch", "dlib", "insightface", "edge_face"]
    available_frameworks = []
    
    for framework in potential_frameworks:
        if framework in server_models:
            available_frameworks.append(framework)
            logger.info(f"  ✅ Framework Model: {framework}")
        else:
            logger.info(f"  ➖ Framework Model not available: {framework}")
    
    # รวมโมเดลที่ใช้งานได้
    ALL_MODELS = available_onnx + available_frameworks
    FRAMEWORK_MODELS = available_frameworks
    
    if not ALL_MODELS:
        raise Exception("No models available for testing!")
    
    # ตั้งค่า Ensemble weights
    total_models = len(ALL_MODELS)
    
    # ONNX models ได้น้ำหนักรวม 60%
    onnx_weight = 0.6 / len(available_onnx) if available_onnx else 0
    
    # Framework models ได้น้ำหนักรวม 40%
    framework_weight = 0.4 / len(available_frameworks) if available_frameworks else 0
    
    ENSEMBLE_WEIGHTS = {}
    for model in available_onnx:
        ENSEMBLE_WEIGHTS[model] = onnx_weight
    for model in available_frameworks:
        ENSEMBLE_WEIGHTS[model] = framework_weight
    
    logger.info(f"📊 Total models to test: {len(ALL_MODELS)}")
    logger.info(f"   ONNX Models: {available_onnx}")
    logger.info(f"   Framework Models: {available_frameworks}")
    logger.info(f"   Ensemble Weights: {ENSEMBLE_WEIGHTS}")
    
    return ALL_MODELS

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
        gallery_response = requests.get(f"{API_BASE_URL}/api/face-recognition/get-gallery", timeout=30)
        if gallery_response.status_code == 200:
            gallery = gallery_response.json()
            logger.info(f"📋 Current gallery has {len(gallery)} persons")
        return True
    except Exception as e:
        logger.error(f"❌ Cannot access database: {e}")
        return False

def register_all_users():
    """ลงทะเบียนผู้ใช้ทั้งหมดด้วยทุกโมเดล"""
    logger.info("🔄 Starting Multi-Model Registration for all users...")
    
    total_registrations = 0
    success_count = 0
    
    for user_id, image_files in USERS.items():
        logger.info(f"\n👤 Registering user: {user_id}")
        
        # ตรวจสอบว่ารูปมีอยู่จริง
        available_images = []
        for img_file in image_files:
            img_path = os.path.join(TEST_IMAGES_DIR, img_file)
            if os.path.exists(img_path):
                available_images.append(img_file)
            else:
                logger.warning(f"⚠️ Image not found: {img_file}")
        
        if len(available_images) < 5:
            logger.error(f"❌ Not enough images for {user_id}: {len(available_images)}/10")
            raise Exception(f"Insufficient images for user {user_id}")
        
        logger.info(f"📁 Found {len(available_images)} images for {user_id}")
        
        # ลงทะเบียนกับทุกโมเดล
        for model_name in ALL_MODELS:
            logger.info(f"  🧠 Registering with {model_name} model...")
            
            model_success = 0
            for i, img_file in enumerate(available_images, 1):
                img_path = os.path.join(TEST_IMAGES_DIR, img_file)
                
                logger.info(f"    [{i}/{len(available_images)}] Processing: {img_file}")
                
                # แปลงเป็น base64
                image_base64 = image_to_base64(img_path)
                
                # ลงทะเบียน
                success = add_single_face(image_base64, user_id, f"User {user_id.title()}", model_name)
                
                total_registrations += 1
                if success:
                    success_count += 1
                    model_success += 1
                    logger.info(f"      ✅ Success")
                else:
                    logger.error(f"      ❌ Failed")
                    raise Exception(f"Registration failed for {user_id}/{img_file}/{model_name}")
                
                # รอเล็กน้อย
                time.sleep(0.5)
            
            logger.info(f"  📊 {model_name}: {model_success}/{len(available_images)} success")
    
    success_rate = (success_count / total_registrations) * 100 if total_registrations > 0 else 0
    logger.info(f"\n✅ Registration completed: {success_count}/{total_registrations} ({success_rate:.1f}%)")
    
    if success_rate < 90:
        raise Exception(f"Registration success rate too low: {success_rate:.1f}%")
    
    return success_count, total_registrations

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
            logger.error(f"❌ Registration failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Registration exception: {e}")
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
    
    test_images.sort()  # เรียงตามชื่อ
    logger.info(f"📁 Found {len(test_images)} test images")
    
    return test_images

def recognize_face_all_models(image_base64: str, faces: List[Dict] = None) -> Dict[str, Any]:
    """ทำการจดจำด้วยทุกโมเดล"""
    results = {}
    
    # ดึง gallery
    try:
        gallery_response = requests.get(f"{API_BASE_URL}/api/face-recognition/get-gallery", timeout=30)
        if gallery_response.status_code != 200:
            logger.error("❌ Cannot get gallery")
            return {}
        gallery = gallery_response.json()
    except Exception as e:
        logger.error(f"❌ Gallery error: {e}")
        return {}
    
    if not gallery:
        logger.error("❌ Gallery is empty")
        return {}
    
    # สำหรับภาพที่มีหลายใบหน้า: ใช้ใบหน้าที่ใหญ่ที่สุด
    if faces and len(faces) > 1:
        face_areas = []
        for face in faces:
            bbox = face.get("bbox", {})
            width = bbox.get("x2", 0) - bbox.get("x1", 0)
            height = bbox.get("y2", 0) - bbox.get("y1", 0)
            area = width * height
            face_areas.append(area)
        
        main_face_idx = face_areas.index(max(face_areas))
        logger.info(f"    📏 Multiple faces detected: Using largest face (#{main_face_idx+1}/{len(faces)})")
        
        # พยายามครอปใบหน้าหลัก
        main_face = faces[main_face_idx]
        try:
            cropped_base64 = crop_face_from_image(image_base64, main_face)
            if cropped_base64:
                image_base64 = cropped_base64
                logger.info(f"    ✂️ Using cropped main face for recognition")
        except Exception as e:
            logger.warning(f"    ⚠️ Cannot crop face, using full image: {e}")
    
    # ทดสอบทุกโมเดล
    for model_name in ALL_MODELS:
        try:
            result = recognize_single_model(image_base64, model_name, gallery)
            results[model_name] = result
        except Exception as e:
            logger.error(f"❌ Recognition failed for {model_name}: {e}")
            results[model_name] = {"matches": [], "error": str(e)}
    
    return results

def recognize_single_model(image_base64: str, model_name: str, gallery: Dict[str, Any]) -> Dict[str, Any]:
    """ทำการจดจำด้วยโมเดลเดียว"""
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
            logger.error(f"❌ Recognition failed: {response.status_code}")
            return {"matches": [], "error": f"HTTP {response.status_code}"}
    except Exception as e:
        logger.error(f"❌ Recognition exception: {e}")
        return {"matches": [], "error": str(e)}

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
                person_id = match.get("person_id", "")
                similarity = match.get("similarity", 0.0)
                weighted_score = similarity * weight
                
                person_scores[person_id].append({
                    "model": model_name,
                    "similarity": similarity,
                    "weighted_score": weighted_score,
                    "weight": weight
                })
    
    # คำนวณคะแนนรวม
    ensemble_matches = []
    for person_id, scores in person_scores.items():
        total_weighted_score = sum(score["weighted_score"] for score in scores)
        total_weight = sum(score["weight"] for score in scores)
        
        if total_weight > 0:
            final_score = total_weighted_score / total_weight
            
            ensemble_matches.append({
                "person_id": person_id,
                "similarity": final_score,
                "contributing_models": len(scores),
                "model_details": scores
            })
    
    # เรียงตามคะแนน
    ensemble_matches.sort(key=lambda x: x["similarity"], reverse=True)
    
    return {
        "matches": ensemble_matches,
        "method": "ensemble",
        "weights_used": ENSEMBLE_WEIGHTS
    }

def detect_faces(image_base64: str) -> List[Dict[str, Any]]:
    """ตรวจจับใบหน้าในภาพ"""
    url = f"{API_BASE_URL}/api/face-detection/detect-base64"
    
    data = {
        "image_base64": image_base64,
        "model_name": "auto",
        "conf_threshold": 0.5
    }
    
    try:
        response = requests.post(url, json=data, timeout=60)
        if response.status_code == 200:
            result = response.json()
            if result.get('success', False):
                return result.get("faces", [])
            else:
                logger.error(f"❌ Face detection failed: {result}")
                return []
        else:
            logger.error(f"❌ Face detection failed: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"❌ Face detection exception: {e}")
        return []

def draw_results_on_image(image_path: str, faces: List[Dict], recognition_results: Dict[str, Any], model_name: str) -> str:
    """วาดผลการจดจำบนภาพ"""
    # อ่านภาพ
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"❌ Cannot read image: {image_path}")
        return ""
    
    # ใช้ผลการจดจำจากโมเดลที่ระบุ
    matches = recognition_results.get("matches", [])
    recognized_person = matches[0].get("person_id", "unknown") if matches else "unknown"
    similarity = matches[0].get("similarity", 0.0) if matches else 0.0
    
    # หาใบหน้าหลักที่ใหญ่ที่สุด
    main_face_idx = 0
    if len(faces) > 1:
        face_areas = []
        for face in faces:
            bbox = face.get("bbox", {})
            width = bbox.get("x2", 0) - bbox.get("x1", 0)
            height = bbox.get("y2", 0) - bbox.get("y1", 0)
            area = width * height
            face_areas.append(area)
        main_face_idx = face_areas.index(max(face_areas))
    
    # วาดกรอบและข้อความ
    for i, face in enumerate(faces):
        bbox = face.get("bbox", {})
        x1 = int(bbox.get("x1", 0))
        y1 = int(bbox.get("y1", 0))
        x2 = int(bbox.get("x2", 0))
        y2 = int(bbox.get("y2", 0))
        
        # กำหนดสีและข้อความตามว่าเป็นใบหน้าหลักหรือไม่
        if i == main_face_idx:
            is_main = True
            color = (0, 255, 0) if recognized_person != "unknown" else (0, 0, 255)
            label = f"MAIN: {recognized_person} ({similarity:.3f})" if recognized_person != "unknown" else "MAIN: Unknown"
            thickness = 4
        else:
            is_main = False
            color = (128, 128, 128)
            label = f"Other #{i+1}"
            thickness = 2
        
        # วาดกรอบ
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # ข้อความ
        model_text = f"Model: {model_name.upper()}"
        
        # พื้นหลังและข้อความ
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6 if is_main else 0.5
        text_thickness = 2 if is_main else 1
        
        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
        (model_w, model_h), _ = cv2.getTextSize(model_text, font, font_scale, text_thickness)
        
        text_bg_color = color if is_main else (100, 100, 100)
        cv2.rectangle(image, (x1, y1 - label_h - model_h - 10), 
                     (x1 + max(label_w, model_w) + 10, y1), text_bg_color, -1)
        
        text_color = (255, 255, 255)
        cv2.putText(image, label, (x1 + 5, y1 - model_h - 5), 
                   font, font_scale, text_color, text_thickness)
        cv2.putText(image, model_text, (x1 + 5, y1 - 5), 
                   font, font_scale, text_color, text_thickness)
    
    # บันทึกภาพ
    output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{model_name}.jpg"
    output_path = os.path.join(OUTPUT_DIR, model_name, output_filename)
    
    cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return output_path

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
        
        # Add padding
        width = x2 - x1
        height = y2 - y1
        padding_x = int(width * 0.1)
        padding_y = int(height * 0.1)
        
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
        logger.warning(f"Cannot crop face: {e}")
        return ""

def generate_performance_report(results: Dict[str, Any]):
    """สร้างรายงานประสิทธิภาพแต่ละโมเดล"""
    report = {
        "test_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "total_images_processed": len(results),
        "models_tested": ALL_MODELS,
        "ensemble_weights": ENSEMBLE_WEIGHTS,
        "detailed_results": results,
        "performance_summary": {}
    }
    
    # วิเคราะห์ประสิทธิภาพ
    model_stats = {}
    
    for model_name in ALL_MODELS + ["ensemble"]:
        correct_boss = 0
        correct_night = 0
        wrong_predictions = 0
        unknown_count = 0
        total_processed = 0
        confidence_scores = []
        
        for image_file, image_results in results.items():
            model_result = image_results["model_results"].get(model_name, {})
            matches = model_result.get("matches", [])
            
            if matches:
                recognized = matches[0].get("person_id", "unknown")
                confidence = matches[0].get("similarity", 0.0)
                confidence_scores.append(confidence)
                
                # ตรวจสอบความถูกต้อง
                if "boss" in image_file.lower():
                    if recognized == "boss":
                        correct_boss += 1
                    elif recognized == "night":
                        wrong_predictions += 1
                    else:
                        unknown_count += 1
                elif "night" in image_file.lower():
                    if recognized == "night":
                        correct_night += 1
                    elif recognized == "boss":
                        wrong_predictions += 1
                    else:
                        unknown_count += 1
                else:
                    # สำหรับรูปอื่นๆ (group, glass, etc.)
                    if recognized in ["boss", "night"]:
                        # ถือว่าถูกถ้าจดจำได้คนที่ลงทะเบียนไว้
                        if "boss" in image_file.lower():
                            correct_boss += 1
                        elif "night" in image_file.lower():
                            correct_night += 1
            else:
                unknown_count += 1
            
            total_processed += 1
        
        # คำนวณสถิติ
        accuracy = ((correct_boss + correct_night) / total_processed * 100) if total_processed > 0 else 0
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        model_stats[model_name] = {
            "model_type": "ONNX" if model_name in ONNX_MODELS else "Framework" if model_name in FRAMEWORK_MODELS else "Ensemble",
            "correct_boss": correct_boss,
            "correct_night": correct_night,
            "wrong_predictions": wrong_predictions,
            "unknown_count": unknown_count,
            "total_processed": total_processed,
            "accuracy": accuracy,
            "average_confidence": avg_confidence,
            "confidence_scores": confidence_scores
        }
    
    report["performance_summary"] = model_stats
    
    # บันทึกรายงาน
    report_file = os.path.join(OUTPUT_DIR, "complete_performance_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # แสดงสรุป
    print_performance_summary(model_stats)
    
    logger.info(f"📄 Performance report saved: {report_file}")

def print_performance_summary(stats: Dict[str, Any]):
    """แสดงสรุปประสิทธิภาพ"""
    print("\n" + "="*100)
    print("🏆 COMPLETE MULTI-MODEL FACE RECOGNITION PERFORMANCE TEST")
    print("="*100)
    print(f"📅 Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👥 Users: boss, night (10 images each)")
    print(f"🧠 Total Models Tested: {len(ALL_MODELS)}")
    print(f"   • ONNX Models: {ONNX_MODELS}")
    print(f"   • Framework Models: {FRAMEWORK_MODELS}")
    print(f"⚖️ Ensemble Weights: {ENSEMBLE_WEIGHTS}")
    print("-"*100)
    
    # เรียงตามประสิทธิภาพ
    sorted_models = sorted(stats.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print(f"\n📊 PERFORMANCE RANKING:")
    print("-"*100)
    print(f"{'Rank':<4} {'Model':<20} {'Type':<10} {'Accuracy':<10} {'Confidence':<12} {'Boss':<6} {'Night':<6} {'Wrong':<6} {'Unknown':<8}")
    print("-"*100)
    
    for rank, (model_name, model_stats) in enumerate(sorted_models, 1):
        print(f"{rank:<4} {model_name:<20} {model_stats['model_type']:<10} "
              f"{model_stats['accuracy']:<10.1f}% {model_stats['average_confidence']:<12.3f} "
              f"{model_stats['correct_boss']:<6} {model_stats['correct_night']:<6} "
              f"{model_stats['wrong_predictions']:<6} {model_stats['unknown_count']:<8}")
    
    # แสดงผลวิเคราะห์
    best_model = sorted_models[0]
    print(f"\n🏆 BEST MODEL: {best_model[0].upper()}")
    print(f"   • Type: {best_model[1]['model_type']}")
    print(f"   • Accuracy: {best_model[1]['accuracy']:.1f}%")
    print(f"   • Average Confidence: {best_model[1]['average_confidence']:.3f}")
    
    # เปรียบเทียบ ONNX vs Framework
    onnx_scores = [stats[m]['accuracy'] for m in ONNX_MODELS if m in stats]
    framework_scores = [stats[m]['accuracy'] for m in FRAMEWORK_MODELS if m in stats]
    
    if onnx_scores and framework_scores:
        avg_onnx = sum(onnx_scores) / len(onnx_scores)
        avg_framework = sum(framework_scores) / len(framework_scores)
        print(f"\n📈 MODEL TYPE COMPARISON:")
        print(f"   • ONNX Models Average: {avg_onnx:.1f}%")
        print(f"   • Framework Models Average: {avg_framework:.1f}%")
        
        if avg_onnx > avg_framework:
            print(f"   🎯 YOUR CUSTOM ONNX MODELS PERFORM BETTER! (+{avg_onnx - avg_framework:.1f}%)")
        else:
            print(f"   🎯 Framework models perform better (+{avg_framework - avg_onnx:.1f}%)")
    
    print(f"\n📁 Output images saved in: {OUTPUT_DIR}")
    print("   - Each model has its own folder with annotated images")
    print("   - Face boxes and recognition results are drawn on images")

def run_complete_test():
    """รันการทดสอบแบบครบถ้วน"""
    logger.info("🚀 Complete Multi-Model Face Recognition Test")
    logger.info("=" * 80)
    
    # ตรวจสอบเซิฟเวอร์
    if not check_server_availability():
        return
    
    # เริ่มต้นโมเดลที่ใช้งานได้
    try:
        initialize_available_models()
        ensure_output_dir()
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return
    
    # ตรวจสอบฐานข้อมูล
    if not clear_database():
        logger.error("❌ Cannot access database")
        return
    
    try:
        # 1. ลงทะเบียนผู้ใช้
        logger.info("\n📝 Phase 1: Multi-Model User Registration")
        success_count, total_count = register_all_users()
        logger.info(f"✅ Registration completed: {success_count}/{total_count}")
        
        # 2. ดึงรายการรูปทดสอบ
        test_images = get_all_test_images()
        if not test_images:
            raise Exception("No test images found")
        
        logger.info(f"\n🎯 Phase 2: Multi-Model Recognition Testing ({len(test_images)} images)")
        
        # 3. ทดสอบการจดจำ
        test_results = {}
        error_count = 0
        
        for i, image_file in enumerate(test_images, 1):
            image_path = os.path.join(TEST_IMAGES_DIR, image_file)
            logger.info(f"\n[{i}/{len(test_images)}] Processing: {image_file}")
            
            try:
                # แปลงเป็น base64
                image_base64 = image_to_base64(image_path)
                
                # ตรวจจับใบหน้า
                faces = detect_faces(image_base64)
                if not faces:
                    logger.error(f"❌ No faces detected in {image_file}")
                    error_count += 1
                    if error_count >= 3:
                        raise Exception("Too many face detection failures")
                    continue
                
                logger.info(f"  👤 Detected {len(faces)} face(s)")
                
                # จดจำด้วยทุกโมเดล
                model_results = recognize_face_all_models(image_base64, faces)
                if not model_results:
                    logger.error(f"❌ Recognition failed for {image_file}")
                    error_count += 1
                    continue
                
                # สร้าง Ensemble prediction
                ensemble_result = create_ensemble_prediction(model_results)
                model_results["ensemble"] = ensemble_result
                
                # วาดผลลัพธ์บนภาพสำหรับทุกโมเดล
                for model_name in ALL_MODELS + ["ensemble"]:
                    if model_name in model_results:
                        output_path = draw_results_on_image(image_path, faces, model_results[model_name], model_name)
                        if output_path:
                            logger.info(f"  📸 {model_name}: {os.path.basename(output_path)}")
                
                # บันทึกผลลัพธ์
                test_results[image_file] = {
                    "faces_detected": len(faces),
                    "model_results": model_results,
                    "processing_time": time.time()
                }
                
                # แสดงผลลัพธ์
                for model_name in ALL_MODELS:
                    if model_name in model_results:
                        matches = model_results[model_name].get("matches", [])
                        if matches:
                            best_match = matches[0]
                            logger.info(f"    🧠 {model_name}: {best_match.get('person_id', 'unknown')} ({best_match.get('similarity', 0):.3f})")
                        else:
                            logger.info(f"    🧠 {model_name}: unknown")
                
                # Ensemble result
                ensemble_matches = ensemble_result.get("matches", [])
                if ensemble_matches:
                    best_ensemble = ensemble_matches[0]
                    logger.info(f"    🏆 Ensemble: {best_ensemble.get('person_id', 'unknown')} ({best_ensemble.get('similarity', 0):.3f})")
                else:
                    logger.info(f"    🏆 Ensemble: unknown")
                
            except Exception as e:
                logger.error(f"❌ Error processing {image_file}: {e}")
                error_count += 1
                if error_count >= 5:
                    raise
        
        # 4. สร้างรายงานประสิทธิภาพ
        logger.info("\n📊 Generating performance report...")
        generate_performance_report(test_results)
        
        logger.info("🎉 Complete multi-model test finished successfully!")
        
    except Exception as e:
        logger.error(f"🛑 Test stopped due to error: {e}")
        raise

if __name__ == "__main__":
    try:
        run_complete_test()
    except Exception as e:
        logger.error(f"🛑 Test terminated: {e}")
        print(f"\n🛑 TEST TERMINATED DUE TO ERROR: {e}")
        exit(1)
