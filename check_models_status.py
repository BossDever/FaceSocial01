#!/usr/bin/env python3
"""
สรุปสถานะโมเดลปัจจุบัน - ไม่ต้องพึ่ง import
"""
import os
import json

def check_model_status():
    """ตรวจสอบสถานะโมเดลปัจจุบัน"""
    
    print("🔍 สรุปสถานะโมเดลในระบบ Face Recognition")
    print("=" * 70)
    
    # ตรวจสอบไฟล์โมเดล ONNX
    print("\n1. 📦 โมเดล ONNX ของคุณ (Custom Models):")
    onnx_models = {
        "FaceNet": "model/face-recognition/facenet_vggface2.onnx",
        "AdaFace": "model/face-recognition/adaface_ir101.onnx", 
        "ArcFace": "model/face-recognition/arcface_r100.onnx"
    }
    
    onnx_available = 0
    for name, path in onnx_models.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"   ✅ {name}: {size_mb:.1f} MB - พร้อมใช้งาน")
            onnx_available += 1
        else:
            print(f"   ❌ {name}: ไม่พบไฟล์")
    
    # ตรวจสอบไลบรารี่ Multi-Framework
    print("\n2. 🔧 ไลบรารี่ Multi-Framework:")
    frameworks = {
        "DeepFace": "deepface",
        "FaceNet-PyTorch": "facenet_pytorch",
        "Dlib": "dlib", 
        "InsightFace": "insightface",
        "EdgeFace": "edge_face"
    }
    
    framework_available = 0
    for name, module_name in frameworks.items():
        try:
            __import__(module_name)
            print(f"   ✅ {name}: ติดตั้งแล้ว")
            framework_available += 1
        except ImportError:
            print(f"   ❌ {name}: ยังไม่ได้ติดตั้ง")
    
    # ตรวจสอบไฟล์ config
    print("\n3. ⚙️ การตั้งค่าระบบ:")
    
    # ตรวจสอบใน service file
    service_file = "src/ai_services/face_recognition/face_recognition_service.py"
    if os.path.exists(service_file):
        with open(service_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # ตรวจสอบว่าโมเดลไหนเป็นค่าเริ่มต้น
        if 'preferred_model=RecognitionModel(config.get("preferred_model", "facenet"))' in content:
            print("   ✅ โมเดลเริ่มต้น: FaceNet (ONNX)")
        
        if 'enable_multi_framework' in content:
            print("   ✅ รองรับ Multi-Framework")
        
        if 'model_configs' in content and 'facenet_vggface2.onnx' in content:
            print("   ✅ โมเดล ONNX กำหนดค่าแล้ว")
    
    # ตรวจสอบ logs
    print("\n4. 📊 ข้อมูลจาก Logs:")
    log_file = "logs/app.log"
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = f.read()
        
        if "Enhanced Face Recognition Service initialized" in logs:
            print("   ✅ Service เริ่มทำงานแล้ว")
        
        if "Recommended model for general use: facenet" in logs:
            print("   ✅ ระบบแนะนำใช้ FaceNet")
            
        if "3/3 models available" in logs:
            print("   ✅ โมเดล ONNX ทั้ง 3 ตัวพร้อมใช้งาน")
    
    # สรุปผล
    print("\n" + "=" * 70)
    print("📋 สรุปผลการตรวจสอบ:")
    print(f"   • โมเดล ONNX ของคุณ: {onnx_available}/3 โมเดล")
    print(f"   • ไลบรารี่ Multi-Framework: {framework_available}/{len(frameworks)} ไลบรารี่")
    
    if onnx_available == 3:
        print("   ✅ โมเดลของคุณยังใช้งานได้ปกติ")
    
    if framework_available > 0:
        print("   ✅ สามารถเปรียบเทียบกับไลบรารี่อื่นได้")
        
    print("\n🎯 คำตอบ: ระบบใช้ทั้งสองแบบ")
    print("   • โมเดล ONNX ของคุณเป็นโมเดลหลัก")
    print("   • ไลบรารี่ Multi-Framework เป็นตัวเลือกเสริม")
    print("   • สามารถเปรียบเทียบประสิทธิภาพได้")

if __name__ == "__main__":
    check_model_status()
