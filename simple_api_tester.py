#!/usr/bin/env python3
"""
Simple API Testing Script for Face Recognition System
Created: June 14, 2025
Purpose: Test all API endpoints with test images and generate comprehensive report
"""

import json
import time
import base64
import requests
from datetime import datetime
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('api_test_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8080"
TEST_IMAGES_DIR = Path("test_images")
OUTPUT_DIR = Path("output/api_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class SimpleAPITester:
    def __init__(self):
        logger.info("🚀 Initializing API Tester...")
        self.session = requests.Session()
        self.session.timeout = 30
        self.start_time = datetime.now()
        
        # Test results storage
        self.results = {
            "test_info": {
                "start_time": self.start_time.isoformat(),
                "api_url": API_BASE_URL
            },
            "tests": {},
            "summary": {},
            "errors": []
        }
        
        # Test images for different scenarios
        self.test_cases = {
            "high_quality_boss": ["boss_01.jpg", "boss_02.jpg"],
            "boss_with_glasses": ["boss_glass01.jpg", "boss_glass02.jpg"],
            "low_light_night": ["night_01.jpg", "night_02.jpg"],
            "group_photos": ["boss_group01.jpg", "night_group01.jpg"],
            "spoofing_images": ["spoofing_01.jpg", "spoofing_02.jpg"],
            "face_swap": ["face-swap01.png"]
        }
        
        # Models to test
        self.detection_models = ["auto", "yolov9c", "yolov9e", "yolov11m"]
        self.recognition_models = ["facenet", "adaface", "arcface"]
        
        logger.info(f"📁 Test images directory: {TEST_IMAGES_DIR}")
        logger.info(f"📊 Output directory: {OUTPUT_DIR}")
    
    def encode_image(self, image_path):
        """Convert image to base64"""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"❌ Failed to encode {image_path}: {e}")
            return None
    
    def api_call(self, method, endpoint, **kwargs):
        """Make API call with error handling"""
        url = f"{API_BASE_URL}{endpoint}"
        start = time.time()
        
        try:
            response = self.session.request(method, url, **kwargs)
            duration = time.time() - start
            
            result = {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "duration": round(duration, 3),
                "endpoint": endpoint,
                "method": method
            }
            
            if response.status_code == 200:
                try:
                    result["data"] = response.json()
                except:
                    result["data"] = {"response": response.text}
            else:
                result["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "duration": time.time() - start,
                "endpoint": endpoint,
                "method": method
            }
    
    def test_system_health(self):
        """Test system health"""
        logger.info("🔧 Testing System Health...")
        
        result = self.api_call("GET", "/health")
        self.results["tests"]["system_health"] = result
        
        if result["success"]:
            data = result["data"]
            logger.info("✅ System Health: OK")
            logger.info(f"   Services: {data.get('active_services', 'N/A')}/{data.get('total_services', 'N/A')}")
            return True
        else:
            logger.error(f"❌ System Health Failed: {result.get('error', 'Unknown error')}")
            self.results["errors"].append("System health check failed")
            return False
    
    def test_face_detection(self):
        """Test face detection APIs"""
        logger.info("👤 Testing Face Detection APIs...")
        
        detection_results = {"health": {}, "models": {}}
        
        # Health check
        health = self.api_call("GET", "/api/face-detection/health")
        detection_results["health"] = health
        
        if health["success"]:
            logger.info("✅ Face Detection Health: OK")
        else:
            logger.error(f"❌ Face Detection Health Failed: {health.get('error')}")
            self.results["tests"]["face_detection"] = detection_results
            return
        
        # Test each detection model
        for model in self.detection_models:
            logger.info(f"   🧠 Testing model: {model}")
            model_results = {}
            
            # Test with different image types
            for category, images in self.test_cases.items():
                category_results = {}
                
                for image_file in images[:1]:  # Test 1 image per category
                    image_path = TEST_IMAGES_DIR / image_file
                    if not image_path.exists():
                        continue
                    
                    # Test file upload
                    try:
                        with open(image_path, "rb") as f:
                            files = {"image": f}
                            data = {"model_name": model, "conf_threshold": 0.5}
                            
                            result = self.api_call("POST", "/api/face-detection/detect", 
                                                 files=files, data=data)
                            
                            if result["success"]:
                                faces = len(result["data"].get("faces", []))
                                logger.info(f"     ✅ {image_file}: {faces} faces detected")
                                category_results[f"{image_file}_file"] = {
                                    "success": True, "faces": faces, "time": result["duration"]
                                }
                            else:
                                logger.error(f"     ❌ {image_file}: {result.get('error')}")
                                category_results[f"{image_file}_file"] = {
                                    "success": False, "error": result.get("error")
                                }
                    except Exception as e:
                        logger.error(f"     ❌ {image_file} (file): {e}")
                    
                    # Test base64
                    try:
                        image_b64 = self.encode_image(image_path)
                        if image_b64:
                            data = {
                                "image_base64": image_b64,
                                "model_name": model,
                                "conf_threshold": 0.5
                            }
                            
                            result = self.api_call("POST", "/api/face-detection/detect-base64", 
                                                 json=data)
                            
                            if result["success"]:
                                faces = len(result["data"].get("faces", []))
                                logger.info(f"     ✅ {image_file} (B64): {faces} faces")
                                category_results[f"{image_file}_base64"] = {
                                    "success": True, "faces": faces, "time": result["duration"]
                                }
                            else:
                                logger.error(f"     ❌ {image_file} (B64): {result.get('error')}")
                                category_results[f"{image_file}_base64"] = {
                                    "success": False, "error": result.get("error")
                                }
                    except Exception as e:
                        logger.error(f"     ❌ {image_file} (base64): {e}")
                
                if category_results:
                    model_results[category] = category_results
            
            detection_results["models"][model] = model_results
        
        self.results["tests"]["face_detection"] = detection_results
    
    def test_face_recognition(self):
        """Test face recognition APIs"""
        logger.info("🔍 Testing Face Recognition APIs...")
        
        recognition_results = {"health": {}, "add_faces": {}, "recognition": {}, "gallery": {}}
        
        # Health check
        health = self.api_call("GET", "/api/face-recognition/health")
        recognition_results["health"] = health
        
        if not health["success"]:
            logger.error(f"❌ Face Recognition Health Failed: {health.get('error')}")
            self.results["tests"]["face_recognition"] = recognition_results
            return
        
        logger.info("✅ Face Recognition Health: OK")
        
        # Add faces to database
        logger.info("   📝 Adding faces to database...")
        add_results = {}
        
        # Add boss faces
        boss_images = ["boss_01.jpg", "boss_02.jpg"]
        for img in boss_images:
            img_path = TEST_IMAGES_DIR / img
            if img_path.exists():
                b64 = self.encode_image(img_path)
                if b64:
                    data = {
                        "person_id": "boss",
                        "person_name": "Boss User",
                        "face_image_base64": b64,
                        "model_name": "facenet"
                    }
                    
                    result = self.api_call("POST", "/api/face-recognition/add-face-json", json=data)
                    add_results[img] = result
                    
                    if result["success"]:
                        logger.info(f"     ✅ Added {img} for boss")
                    else:
                        logger.error(f"     ❌ Failed to add {img}: {result.get('error')}")
        
        # Add night faces
        night_images = ["night_01.jpg", "night_02.jpg"]
        for img in night_images:
            img_path = TEST_IMAGES_DIR / img
            if img_path.exists():
                b64 = self.encode_image(img_path)
                if b64:
                    data = {
                        "person_id": "night",
                        "person_name": "Night User",
                        "face_image_base64": b64,
                        "model_name": "facenet"
                    }
                    
                    result = self.api_call("POST", "/api/face-recognition/add-face-json", json=data)
                    add_results[img] = result
                    
                    if result["success"]:
                        logger.info(f"     ✅ Added {img} for night")
                    else:
                        logger.error(f"     ❌ Failed to add {img}: {result.get('error')}")
        
        recognition_results["add_faces"] = add_results
        
        # Get gallery
        gallery_result = self.api_call("GET", "/api/face-recognition/get-gallery")
        recognition_results["gallery"] = gallery_result
        
        gallery = {}
        if gallery_result["success"]:
            gallery = gallery_result["data"].get("gallery", {})
            logger.info(f"   ✅ Gallery retrieved: {len(gallery)} persons")
        else:
            logger.error(f"   ❌ Failed to get gallery: {gallery_result.get('error')}")
        
        # Test recognition with different models
        logger.info("   🎯 Testing recognition...")
        test_cases = [
            ("boss", "boss_03.jpg"),  # Should recognize as boss
            ("boss", "boss_glass01.jpg"),  # Boss with glasses
            ("night", "night_03.jpg"),  # Should recognize as night
            ("unknown", "spoofing_01.jpg")  # Should not recognize
        ]
        
        recog_results = {}
        for model in self.recognition_models:
            logger.info(f"     🧠 Model: {model}")
            model_results = {}
            
            for expected, img_file in test_cases:
                img_path = TEST_IMAGES_DIR / img_file
                if not img_path.exists():
                    continue
                
                b64 = self.encode_image(img_path)
                if b64:
                    data = {
                        "face_image_base64": b64,
                        "gallery": gallery,
                        "model_name": model,
                        "top_k": 3
                    }
                    
                    result = self.api_call("POST", "/api/face-recognition/recognize", json=data)
                    
                    if result["success"]:
                        matches = result["data"].get("matches", [])
                        if matches:
                            top = matches[0]
                            person = top.get("person_id", "unknown")
                            confidence = top.get("confidence", 0)
                            correct = person == expected
                            
                            logger.info(f"       {'✅' if correct else '❌'} {img_file}: {person} ({confidence:.3f})")
                            model_results[img_file] = {
                                "success": True,
                                "predicted": person,
                                "expected": expected,
                                "confidence": confidence,
                                "correct": correct
                            }
                        else:
                            logger.info(f"       ✅ {img_file}: No matches (expected for unknown)")
                            model_results[img_file] = {
                                "success": True,
                                "predicted": "no_match",
                                "expected": expected,
                                "correct": expected == "unknown"
                            }
                    else:
                        logger.error(f"       ❌ {img_file}: {result.get('error')}")
                        model_results[img_file] = {
                            "success": False,
                            "error": result.get("error")
                        }
            
            recog_results[model] = model_results
        
        recognition_results["recognition"] = recog_results
        self.results["tests"]["face_recognition"] = recognition_results
    
    def test_face_analysis(self):
        """Test face analysis APIs"""
        logger.info("📊 Testing Face Analysis APIs...")
        
        analysis_results = {"health": {}, "analysis": {}}
        
        # Health check
        health = self.api_call("GET", "/api/face-analysis/health")
        analysis_results["health"] = health
        
        if not health["success"]:
            logger.error(f"❌ Face Analysis Health Failed: {health.get('error')}")
            self.results["tests"]["face_analysis"] = analysis_results
            return
        
        logger.info("✅ Face Analysis Health: OK")
        
        # Test analysis modes
        modes = ["full_analysis", "detection_only", "recognition_only"]
        test_images = ["boss_01.jpg", "boss_group01.jpg", "night_01.jpg"]
        
        mode_results = {}
        for mode in modes:
            logger.info(f"   🔍 Mode: {mode}")
            mode_data = {}
            
            for img_file in test_images:
                img_path = TEST_IMAGES_DIR / img_file
                if not img_path.exists():
                    continue
                
                b64 = self.encode_image(img_path)
                if b64:
                    data = {
                        "image_base64": b64,
                        "mode": mode,
                        "config": {
                            "detection_model": "auto",
                            "recognition_model": "facenet"
                        }
                    }
                    
                    # Add gallery for recognition modes
                    if mode in ["full_analysis", "recognition_only"]:
                        gallery_result = self.api_call("GET", "/api/face-recognition/get-gallery")
                        if gallery_result["success"]:
                            data["gallery"] = gallery_result["data"].get("gallery", {})
                    
                    result = self.api_call("POST", "/api/face-analysis/analyze-json", json=data)
                    
                    if result["success"]:
                        faces = len(result["data"].get("faces", []))
                        logger.info(f"     ✅ {img_file}: {faces} faces analyzed")
                        mode_data[img_file] = {
                            "success": True,
                            "faces": faces,
                            "time": result["duration"]
                        }
                    else:
                        logger.error(f"     ❌ {img_file}: {result.get('error')}")
                        mode_data[img_file] = {
                            "success": False,
                            "error": result.get("error")
                        }
            
            mode_results[mode] = mode_data
        
        analysis_results["analysis"] = mode_results
        self.results["tests"]["face_analysis"] = analysis_results
    
    def test_info_endpoints(self):
        """Test information and performance endpoints"""
        logger.info("ℹ️ Testing Information Endpoints...")
        
        endpoints = [
            ("/api/face-detection/models/available", "Detection Models"),
            ("/api/face-detection/models/status", "Detection Status"),
            ("/api/face-recognition/models/available", "Recognition Models"),
            ("/api/face-recognition/database-status", "Database Status"),
            ("/api/face-detection/performance/stats", "Detection Performance"),
            ("/api/face-recognition/performance/stats", "Recognition Performance")
        ]
        
        info_results = {}
        for endpoint, name in endpoints:
            result = self.api_call("GET", endpoint)
            info_results[endpoint] = result
            
            if result["success"]:
                logger.info(f"   ✅ {name}: OK")
            else:
                logger.error(f"   ❌ {name}: {result.get('error')}")
        
        self.results["tests"]["info_endpoints"] = info_results
    
    def run_all_tests(self):
        """Run all tests"""
        logger.info("🚀 Starting Complete API Test Suite...")
        logger.info("=" * 50)
        
        try:
            # Check API availability
            if not self.test_system_health():
                logger.error("❌ System health check failed. Stopping tests.")
                return False
            
            # Run all test suites
            self.test_face_detection()
            self.test_face_recognition()
            self.test_face_analysis()
            self.test_info_endpoints()
            
            # Generate report
            self.generate_report()
            return True
            
        except Exception as e:
            logger.error(f"❌ Testing failed: {e}")
            self.results["errors"].append(f"Testing error: {str(e)}")
            return False
    
    def generate_report(self):
        """Generate final test report"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Calculate statistics
        total_tests = 0
        successful_tests = 0
        
        def count_tests(data):
            nonlocal total_tests, successful_tests
            if isinstance(data, dict):
                if "success" in data:
                    total_tests += 1
                    if data["success"]:
                        successful_tests += 1
                else:
                    for value in data.values():
                        if isinstance(value, dict):
                            count_tests(value)
        
        count_tests(self.results["tests"])
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Update results with summary
        self.results["test_info"]["end_time"] = end_time.isoformat()
        self.results["test_info"]["duration_seconds"] = duration
        self.results["summary"] = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "success_rate_percent": round(success_rate, 2)
        }
        
        # Add recommendations
        self.results["recommendations"] = self.get_recommendations()
        
        # Save to file
        timestamp = int(time.time())
        report_file = OUTPUT_DIR / f"api_test_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # Print summary
        logger.info("=" * 50)
        logger.info("📊 TEST RESULTS SUMMARY")
        logger.info("=" * 50)
        logger.info(f"   🕐 Duration: {duration:.1f} seconds")
        logger.info(f"   📋 Total Tests: {total_tests}")
        logger.info(f"   ✅ Successful: {successful_tests}")
        logger.info(f"   ❌ Failed: {total_tests - successful_tests}")
        logger.info(f"   📈 Success Rate: {success_rate:.1f}%")
        logger.info(f"   📄 Report saved: {report_file}")
        
        # Print key findings
        self.print_key_findings()
        
        return self.results
    
    def get_recommendations(self):
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check overall health
        system_health = self.results["tests"].get("system_health", {})
        if not system_health.get("success", False):
            recommendations.append("System health check failed - verify API server status")
        
        # Check service health
        for service in ["face_detection", "face_recognition", "face_analysis"]:
            service_data = self.results["tests"].get(service, {})
            health = service_data.get("health", {})
            if not health.get("success", False):
                recommendations.append(f"{service.replace('_', ' ').title()} service is unhealthy")
        
        # Check model performance
        detection_models = self.results["tests"].get("face_detection", {}).get("models", {})
        for model, results in detection_models.items():
            success_count = 0
            total_count = 0
            for category in results.values():
                for test in category.values():
                    total_count += 1
                    if test.get("success", False):
                        success_count += 1
            
            if total_count > 0:
                success_rate = (success_count / total_count) * 100
                if success_rate < 80:
                    recommendations.append(f"Detection model '{model}' has low success rate ({success_rate:.1f}%)")
        
        # Check recognition accuracy
        recognition_data = self.results["tests"].get("face_recognition", {}).get("recognition", {})
        for model, results in recognition_data.items():
            correct = sum(1 for r in results.values() if r.get("correct", False))
            total = len([r for r in results.values() if r.get("success", False)])
            
            if total > 0:
                accuracy = (correct / total) * 100
                if accuracy < 90:
                    recommendations.append(f"Recognition model '{model}' accuracy is {accuracy:.1f}% (below 90%)")
        
        if not recommendations:
            recommendations.append("All tests passed successfully! API system is working optimally.")
        
        return recommendations
    
    def print_key_findings(self):
        """Print key findings from tests"""
        logger.info("\n🔍 KEY FINDINGS:")
        
        # Model performance summary
        detection_models = self.results["tests"].get("face_detection", {}).get("models", {})
        if detection_models:
            logger.info("   📊 Detection Model Performance:")
            for model in detection_models:
                logger.info(f"     • {model}: Available and tested")
        
        # Recognition accuracy
        recognition_data = self.results["tests"].get("face_recognition", {}).get("recognition", {})
        if recognition_data:
            logger.info("   🎯 Recognition Accuracy by Model:")
            for model, results in recognition_data.items():
                correct = sum(1 for r in results.values() if r.get("correct", False))
                total = len([r for r in results.values() if r.get("success", False)])
                if total > 0:
                    accuracy = (correct / total) * 100
                    logger.info(f"     • {model}: {accuracy:.1f}% accuracy ({correct}/{total})")
        
        # Print recommendations
        recommendations = self.results.get("recommendations", [])
        if recommendations:
            logger.info("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"   {i}. {rec}")

def main():
    """Main function"""
    print("🚀 Face Recognition System - Complete API Testing Suite")
    print("=" * 60)
    
    # Check requirements
    if not TEST_IMAGES_DIR.exists():
        print(f"❌ Test images directory not found: {TEST_IMAGES_DIR}")
        print("Please ensure test_images directory exists with test images.")
        return False
    
    # Check API server
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ API server not responding at {API_BASE_URL}")
            return False
    except:
        print(f"❌ Cannot connect to API server at {API_BASE_URL}")
        print("Please start the API server first.")
        return False
    
    print("✅ API server is accessible")
    print(f"📁 Using test images from: {TEST_IMAGES_DIR}")
    
    # Run tests
    tester = SimpleAPITester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        return True
    else:
        print("\n⚠️ Some tests failed. Check the logs for details.")
        return False

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
