#!/usr/bin/env python3
"""
Face Recognition System - Comprehensive Testing Script
ทดสอบระบบจดจำใบหน้าอย่างครอบคลุม
"""

import requests
import os
import json
import base64
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

class FaceRecognitionTester:
    def __init__(self, api_base_url: str = "http://127.0.0.1:8080/api"):
        self.api_base_url = api_base_url
        self.test_images_dir = Path("test_images")
        self.output_dir = Path("output/test_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def encode_image_to_base64(self, image_path: Path) -> Optional[str]:
        """Convert image to base64"""
        try:
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            self.log(f"Error encoding {image_path}: {e}", "ERROR")
            return None
    
    def test_health_check(self) -> bool:
        """Test API health"""
        try:
            response = requests.get(f"{self.api_base_url.replace('/api', '')}/health", timeout=10)
            if response.status_code == 200:
                self.log("✅ Health check passed")
                return True
            else:
                self.log(f"❌ Health check failed: {response.status_code}", "ERROR")
                return False
        except Exception as e:
            self.log(f"❌ Health check error: {e}", "ERROR")
            return False
    
    def test_face_detection(self, image_path: Path) -> Optional[Dict]:
        """Test face detection"""
        try:
            image_base64 = self.encode_image_to_base64(image_path)
            if not image_base64:
                return None
                
            response = requests.post(
                f"{self.api_base_url}/face-detection/detect-base64",
                json={
                    "image_base64": image_base64,
                    "model_name": "auto",
                    "conf_threshold": 0.5,
                    "max_faces": 10
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                faces_count = len(result.get("faces", []))
                self.log(f"✅ Detection success: {faces_count} faces in {image_path.name}")
                return result
            else:
                self.log(f"❌ Detection failed for {image_path.name}: {response.status_code}", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"❌ Detection error for {image_path.name}: {e}", "ERROR")
            return None
    
    def register_face(self, image_path: Path, person_name: str, person_id: str) -> bool:
        """Register a face"""
        try:
            image_base64 = self.encode_image_to_base64(image_path)
            if not image_base64:
                return False
                
            response = requests.post(
                f"{self.api_base_url}/face-recognition/add-face-json",
                json={
                    "person_name": person_name,
                    "person_id": person_id,
                    "face_image_base64": image_base64,
                    "model_name": "facenet"
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                self.log(f"✅ Registered {person_name} successfully")
                return True
            else:
                error_detail = response.json().get("detail", "Unknown error")
                self.log(f"❌ Registration failed for {person_name}: {error_detail}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ Registration error for {person_name}: {e}", "ERROR")
            return False
    
    def get_gallery(self) -> Optional[Dict]:
        """Get current gallery"""
        try:
            response = requests.get(f"{self.api_base_url}/face-recognition/get-gallery", timeout=30)
            if response.status_code == 200:
                gallery = response.json()
                count = len(gallery)
                self.log(f"✅ Gallery retrieved: {count} people")
                return gallery
            else:
                self.log(f"❌ Failed to get gallery: {response.status_code}", "ERROR")
                return None
        except Exception as e:
            self.log(f"❌ Gallery error: {e}", "ERROR")
            return None
    
    def recognize_face(self, image_path: Path, gallery: Dict) -> Optional[Dict]:
        """Recognize face against gallery"""
        try:
            image_base64 = self.encode_image_to_base64(image_path)
            if not image_base64:
                return None
                
            response = requests.post(
                f"{self.api_base_url}/face-recognition/recognize",
                json={
                    "face_image_base64": image_base64,
                    "gallery": gallery,
                    "model_name": "facenet",
                    "top_k": 5,
                    "similarity_threshold": 0.6
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                self.log(f"❌ Recognition failed for {image_path.name}: {response.status_code}", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"❌ Recognition error for {image_path.name}: {e}", "ERROR")
            return None
    
    def analyze_face_full(self, image_path: Path, gallery: Dict) -> Optional[Dict]:
        """Full face analysis"""
        try:
            image_base64 = self.encode_image_to_base64(image_path)
            if not image_base64:
                return None
                
            response = requests.post(
                f"{self.api_base_url}/face-analysis/analyze-json",
                json={
                    "image_base64": image_base64,
                    "mode": "full_analysis",
                    "gallery": gallery,
                    "config": {
                        "detection_model": "auto",
                        "recognition_model": "facenet",
                        "similarity_threshold": 0.6,
                        "enable_gallery_matching": True
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                self.log(f"❌ Analysis failed for {image_path.name}: {response.status_code}", "ERROR")
                return None
                
        except Exception as e:
            self.log(f"❌ Analysis error for {image_path.name}: {e}", "ERROR")
            return None
    
    def clear_gallery(self) -> bool:
        """Clear gallery (if endpoint exists)"""
        try:
            response = requests.delete(f"{self.api_base_url}/face-recognition/clear-gallery", timeout=30)
            if response.status_code == 200:
                self.log("✅ Gallery cleared")
                return True
            else:
                self.log("⚠️ Clear gallery not supported or failed")
                return False
        except Exception as e:
            self.log("⚠️ Clear gallery not supported")
            return False
    
    def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        self.log("🚀 Starting comprehensive face recognition test")
        self.log("=" * 60)
        
        # 1. Health check
        self.log("1️⃣ Testing API health...")
        if not self.test_health_check():
            self.log("❌ Health check failed. Stopping tests.", "ERROR")
            return
        
        # 2. Clear gallery (optional)
        self.log("\n2️⃣ Clearing gallery...")
        self.clear_gallery()
        
        # 3. Test face detection
        self.log("\n3️⃣ Testing face detection...")
        test_images = list(self.test_images_dir.glob("*.jpg"))[:5]  # Test first 5 images
        
        detection_results = {}
        for img_path in test_images:
            result = self.test_face_detection(img_path)
            if result:
                detection_results[img_path.name] = result
        
        self.log(f"Detection test completed: {len(detection_results)}/{len(test_images)} successful")
        
        # 4. Register test faces
        self.log("\n4️⃣ Registering test faces...")
        
        # Register Boss
        boss_images = [p for p in self.test_images_dir.glob("boss_*.jpg")][:3]
        if boss_images:
            if self.register_face(boss_images[0], "Boss", "boss"):
                self.log("✅ Boss registered successfully")
            else:
                self.log("❌ Boss registration failed", "ERROR")
        
        # Register Night  
        night_images = [p for p in self.test_images_dir.glob("night_*.jpg")][:3]
        if night_images:
            if self.register_face(night_images[0], "Night", "night"):
                self.log("✅ Night registered successfully")
            else:
                self.log("❌ Night registration failed", "ERROR")
        
        # 5. Get gallery
        self.log("\n5️⃣ Getting gallery...")
        gallery = self.get_gallery()
        if not gallery:
            self.log("❌ Failed to get gallery. Stopping recognition tests.", "ERROR")
            return
        
        # 6. Test recognition
        self.log("\n6️⃣ Testing recognition...")
        
        recognition_results = {}
        test_recognition_images = [
            *boss_images[1:3],  # Test other boss images
            *night_images[1:3]  # Test other night images
        ]
        
        for img_path in test_recognition_images:
            self.log(f"\n🔍 Testing recognition for {img_path.name}...")
            
            # Test direct recognition
            rec_result = self.recognize_face(img_path, gallery)
            if rec_result:
                best_match = rec_result.get("best_match")
                confidence = rec_result.get("confidence", 0)
                
                if best_match:
                    person_id = best_match.get("person_id", "unknown")
                    similarity = best_match.get("similarity", 0)
                    self.log(f"   📋 Direct Recognition: {person_id} (similarity: {similarity:.3f}, confidence: {confidence:.3f})")
                else:
                    self.log(f"   📋 Direct Recognition: No match (confidence: {confidence:.3f})")
            
            # Test full analysis
            analysis_result = self.analyze_face_full(img_path, gallery)
            if analysis_result:
                faces = analysis_result.get("faces", [])
                identified_count = analysis_result.get("identified_count", 0)
                total_faces = analysis_result.get("total_faces", 0)
                
                self.log(f"   🔬 Full Analysis: {identified_count}/{total_faces} faces identified")
                
                for i, face in enumerate(faces):
                    best_match = face.get("best_match")
                    if best_match:
                        person_id = best_match.get("person_id", "unknown")
                        similarity = best_match.get("similarity", 0)
                        self.log(f"      Face {i+1}: {person_id} (similarity: {similarity:.3f})")
                    else:
                        self.log(f"      Face {i+1}: Unknown")
            
            recognition_results[img_path.name] = {
                "direct_recognition": rec_result,
                "full_analysis": analysis_result
            }
        
        # 7. Test threshold sensitivity
        self.log("\n7️⃣ Testing threshold sensitivity...")
        if boss_images:
            test_img = boss_images[1]
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            
            for threshold in thresholds:
                try:
                    image_base64 = self.encode_image_to_base64(test_img)
                    response = requests.post(
                        f"{self.api_base_url}/face-recognition/recognize",
                        json={
                            "face_image_base64": image_base64,
                            "gallery": gallery,
                            "similarity_threshold": threshold
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        best_match = result.get("best_match")
                        if best_match:
                            similarity = best_match.get("similarity", 0)
                            person_id = best_match.get("person_id", "unknown")
                            self.log(f"   Threshold {threshold}: {person_id} (sim: {similarity:.3f})")
                        else:
                            self.log(f"   Threshold {threshold}: No match")
                            
                except Exception as e:
                    self.log(f"   Threshold {threshold}: Error - {e}")
        
        # 8. Save results
        self.log("\n8️⃣ Saving results...")
        results_file = self.output_dir / f"test_results_{int(time.time())}.json"
        
        final_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "detection_results": detection_results,
            "recognition_results": recognition_results,
            "gallery": gallery,
            "summary": {
                "total_detection_tests": len(test_images),
                "successful_detections": len(detection_results),
                "total_recognition_tests": len(test_recognition_images),
                "successful_recognitions": len([r for r in recognition_results.values() if r["direct_recognition"]])
            }
        }
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            self.log(f"✅ Results saved to {results_file}")
        except Exception as e:
            self.log(f"❌ Failed to save results: {e}", "ERROR")
        
        # 9. Summary
        self.log("\n" + "=" * 60)
        self.log("📊 TEST SUMMARY")
        self.log("=" * 60)
        
        summary = final_results["summary"]
        self.log(f"Detection Success Rate: {summary['successful_detections']}/{summary['total_detection_tests']}")
        self.log(f"Recognition Tests: {summary['total_recognition_tests']}")
        self.log(f"Gallery People: {len(gallery) if gallery else 0}")
        
        # Check for potential issues
        if summary['successful_detections'] < summary['total_detection_tests']:
            self.log("⚠️ Some detection tests failed", "WARNING")
        
        if gallery and len(gallery) < 2:
            self.log("⚠️ Gallery has fewer than 2 people registered", "WARNING")
        
        self.log("🏁 Test completed!")

def main():
    """Main function"""
    print("🎭 Face Recognition System - Comprehensive Test")
    print("=" * 50)
    
    # Check if server is running
    tester = FaceRecognitionTester()
    
    if not tester.test_health_check():
        print("\n❌ Server is not running or not responding!")
        print("Please start the server first:")
        print("  python start.py")
        return
    
    # Run tests
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()
