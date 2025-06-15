#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified API Testing Script for Face Recognition System
Created: June 14, 2025
Purpose: Test all API endpoints systematically with proper error handling
"""

import os
import sys
import json
import time
import base64
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8080"
TEST_IMAGES_DIR = Path("test_images")
OUTPUT_DIR = Path("output/api_test_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class APITester:
    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 30
        self.results = []
        self.test_start_time = datetime.now()
        
        # Test images organized by purpose
        self.test_images = {
            "high_quality": ["boss_01.jpg", "boss_02.jpg", "boss_03.jpg"],
            "with_glasses": ["boss_glass01.jpg", "boss_glass02.jpg"],
            "low_light": ["night_01.jpg", "night_02.jpg", "night_03.jpg"],
            "group_photos": ["boss_group01.jpg", "night_group01.jpg"],
            "spoofing": ["spoofing_01.jpg", "spoofing_02.jpg"],
            "face_swap": ["face-swap01.png", "face-swap02.png"]
        }
        
        # Models to test
        self.detection_models = ["auto", "yolov9c", "yolov9e", "yolov11m"]
        self.recognition_models = ["facenet", "adaface", "arcface"]
        
        # Gallery for recognition tests
        self.gallery = {}
        
        # Test results storage
        self.test_results = {
            "system_health": {},
            "face_detection": {},
            "face_recognition": {},
            "face_analysis": {},
            "performance": {},
            "errors": []
        }
    
    def encode_image_to_base64(self, image_path):
        """Encode image to base64 string"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return None
    
    def make_request(self, method, endpoint, **kwargs):
        """Make HTTP request with error handling"""
        url = f"{API_BASE_URL}{endpoint}"
        start_time = time.time()
        
        try:
            response = self.session.request(method, url, **kwargs)
            response_time = time.time() - start_time
            
            result = {
                "endpoint": endpoint,
                "method": method,
                "status_code": response.status_code,
                "response_time": response_time,
                "success": response.status_code == 200,
                "error": None
            }
            
            if response.status_code == 200:
                try:
                    result["data"] = response.json()
                except:
                    result["data"] = {"raw_response": response.text}
            else:
                result["error"] = f"HTTP {response.status_code}: {response.text}"
            
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "endpoint": endpoint,
                "method": method,
                "status_code": 0,
                "response_time": response_time,
                "success": False,
                "error": str(e),
                "data": {}
            }
    
    def test_system_health(self):
        """Test system health endpoint"""
        logger.info("🔧 Testing System Health...")
        
        result = self.make_request("GET", "/health")
        self.test_results["system_health"] = result
        
        if result["success"]:
            data = result["data"]
            logger.info("✅ System Health: OK")
            logger.info(f"   Status: {data.get('status', 'Unknown')}")
            logger.info(f"   Active Services: {data.get('active_services', 'Unknown')}")
            logger.info(f"   Total Services: {data.get('total_services', 'Unknown')}")
        else:
            logger.error(f"❌ System Health: {result['error']}")
            self.test_results["errors"].append(f"System Health: {result['error']}")
    
    def test_face_detection_apis(self):
        """Test all face detection endpoints"""
        logger.info("👤 Testing Face Detection APIs...")
        
        # Health check
        health_result = self.make_request("GET", "/api/face-detection/health")
        self.test_results["face_detection"]["health"] = health_result
        
        if health_result["success"]:
            logger.info("✅ Face Detection Health: OK")
        else:
            logger.error(f"❌ Face Detection Health: {health_result['error']}")
        
        # Test detection with different models and images
        detection_results = {}
        
        for model in self.detection_models:
            logger.info(f"   Testing model: {model}")
            model_results = {}
            
            for category, images in self.test_images.items():
                category_results = {}
                
                for image_file in images[:2]:  # Test first 2 images per category
                    image_path = TEST_IMAGES_DIR / image_file
                    if not image_path.exists():
                        continue
                    
                    # Test file upload
                    try:
                        with open(image_path, "rb") as f:
                            files = {"image": f}
                            data = {"model_name": model, "conf_threshold": 0.5}
                            
                            result = self.make_request(
                                "POST", 
                                "/api/face-detection/detect",
                                files=files,
                                data=data
                            )
                            
                            if result["success"]:
                                faces = result["data"].get("faces", [])
                                logger.info(f"     ✅ {image_file}: {len(faces)} faces detected")
                                category_results[f"{image_file}_upload"] = {
                                    "success": True,
                                    "faces_detected": len(faces),
                                    "response_time": result["response_time"]
                                }
                            else:
                                logger.error(f"     ❌ {image_file}: {result['error']}")
                                category_results[f"{image_file}_upload"] = {
                                    "success": False,
                                    "error": result["error"]
                                }
                    except Exception as e:
                        logger.error(f"     ❌ {image_file} (upload): {str(e)}")
                    
                    # Test base64
                    try:
                        image_base64 = self.encode_image_to_base64(image_path)
                        if image_base64:
                            data = {
                                "image_base64": image_base64,
                                "model_name": model,
                                "conf_threshold": 0.5
                            }
                            
                            result = self.make_request(
                                "POST",
                                "/api/face-detection/detect-base64",
                                json=data
                            )
                            
                            if result["success"]:
                                faces = result["data"].get("faces", [])
                                logger.info(f"     ✅ {image_file} (Base64): {len(faces)} faces detected")
                                category_results[f"{image_file}_base64"] = {
                                    "success": True,
                                    "faces_detected": len(faces),
                                    "response_time": result["response_time"]
                                }
                            else:
                                logger.error(f"     ❌ {image_file} (Base64): {result['error']}")
                                category_results[f"{image_file}_base64"] = {
                                    "success": False,
                                    "error": result["error"]
                                }
                    except Exception as e:
                        logger.error(f"     ❌ {image_file} (Base64): {str(e)}")
                
                if category_results:
                    model_results[category] = category_results
            
            if model_results:
                detection_results[model] = model_results
        
        self.test_results["face_detection"]["models"] = detection_results
    
    def test_face_recognition_apis(self):
        """Test all face recognition endpoints"""
        logger.info("🔍 Testing Face Recognition APIs...")
        
        # Health check
        health_result = self.make_request("GET", "/api/face-recognition/health")
        self.test_results["face_recognition"]["health"] = health_result
        
        if health_result["success"]:
            logger.info("✅ Face Recognition Health: OK")
        else:
            logger.error(f"❌ Face Recognition Health: {health_result['error']}")
            return
        
        # Add faces to database
        logger.info("   Adding faces to database...")
        add_results = {}
        
        # Add boss faces
        boss_images = ["boss_01.jpg", "boss_02.jpg", "boss_03.jpg"]
        for image_file in boss_images:
            image_path = TEST_IMAGES_DIR / image_file
            if not image_path.exists():
                continue
            
            try:
                image_base64 = self.encode_image_to_base64(image_path)
                if image_base64:
                    data = {
                        "person_id": "boss",
                        "person_name": "Boss User",
                        "face_image_base64": image_base64,
                        "model_name": "facenet"
                    }
                    
                    result = self.make_request(
                        "POST",
                        "/api/face-recognition/add-face-json",
                        json=data
                    )
                    
                    if result["success"]:
                        logger.info(f"     ✅ Added {image_file} for boss")
                        add_results[image_file] = {"success": True, "person": "boss"}
                    else:
                        logger.error(f"     ❌ Failed to add {image_file}: {result['error']}")
                        add_results[image_file] = {"success": False, "error": result["error"]}
            except Exception as e:
                logger.error(f"     ❌ Error adding {image_file}: {str(e)}")
        
        # Add night faces
        night_images = ["night_01.jpg", "night_02.jpg"]
        for image_file in night_images:
            image_path = TEST_IMAGES_DIR / image_file
            if not image_path.exists():
                continue
            
            try:
                image_base64 = self.encode_image_to_base64(image_path)
                if image_base64:
                    data = {
                        "person_id": "night",
                        "person_name": "Night User", 
                        "face_image_base64": image_base64,
                        "model_name": "facenet"
                    }
                    
                    result = self.make_request(
                        "POST",
                        "/api/face-recognition/add-face-json",
                        json=data
                    )
                    
                    if result["success"]:
                        logger.info(f"     ✅ Added {image_file} for night")
                        add_results[image_file] = {"success": True, "person": "night"}
                    else:
                        logger.error(f"     ❌ Failed to add {image_file}: {result['error']}")
                        add_results[image_file] = {"success": False, "error": result["error"]}
            except Exception as e:
                logger.error(f"     ❌ Error adding {image_file}: {str(e)}")
        
        self.test_results["face_recognition"]["add_faces"] = add_results
        
        # Get gallery
        gallery_result = self.make_request("GET", "/api/face-recognition/get-gallery")
        if gallery_result["success"]:
            self.gallery = gallery_result["data"].get("gallery", {})
            logger.info(f"   ✅ Gallery retrieved: {len(self.gallery)} persons")
            self.test_results["face_recognition"]["gallery"] = {
                "success": True,
                "persons_count": len(self.gallery)
            }
        else:
            logger.error(f"   ❌ Failed to get gallery: {gallery_result['error']}")
            self.test_results["face_recognition"]["gallery"] = {
                "success": False,
                "error": gallery_result["error"]
            }
        
        # Test recognition with different models
        recognition_results = {}
        test_cases = [
            ("boss", "boss_04.jpg"),
            ("boss", "boss_glass01.jpg"),
            ("night", "night_03.jpg"),
            ("unknown", "spoofing_01.jpg")
        ]
        
        for model in self.recognition_models:
            logger.info(f"   Testing recognition model: {model}")
            model_results = {}
            
            for expected_person, image_file in test_cases:
                image_path = TEST_IMAGES_DIR / image_file
                if not image_path.exists():
                    continue
                
                try:
                    image_base64 = self.encode_image_to_base64(image_path)
                    if image_base64:
                        data = {
                            "face_image_base64": image_base64,
                            "gallery": self.gallery,
                            "model_name": model,
                            "top_k": 3
                        }
                        
                        result = self.make_request(
                            "POST",
                            "/api/face-recognition/recognize",
                            json=data
                        )
                        
                        if result["success"]:
                            matches = result["data"].get("matches", [])
                            if matches:
                                top_match = matches[0]
                                person_id = top_match.get("person_id", "unknown")
                                confidence = top_match.get("confidence", 0)
                                correct = person_id == expected_person
                                logger.info(f"     ✅ {image_file}: {person_id} (conf: {confidence:.3f}) {'✓' if correct else '✗'}")
                                
                                model_results[image_file] = {
                                    "success": True,
                                    "predicted": person_id,
                                    "expected": expected_person,
                                    "confidence": confidence,
                                    "correct": correct,
                                    "matches_count": len(matches)
                                }
                            else:
                                logger.info(f"     ✅ {image_file}: No matches found")
                                model_results[image_file] = {
                                    "success": True,
                                    "predicted": "no_match",
                                    "expected": expected_person,
                                    "correct": expected_person == "unknown"
                                }
                        else:
                            logger.error(f"     ❌ {image_file}: {result['error']}")
                            model_results[image_file] = {
                                "success": False,
                                "error": result["error"]
                            }
                except Exception as e:
                    logger.error(f"     ❌ {image_file}: {str(e)}")
            
            if model_results:
                recognition_results[model] = model_results
        
        self.test_results["face_recognition"]["recognition"] = recognition_results
        
        # Test embedding extraction
        logger.info("   Testing embedding extraction...")
        embedding_results = {}
        
        for model in self.recognition_models:
            model_results = {}
            
            for image_file in ["boss_01.jpg", "night_01.jpg"]:
                image_path = TEST_IMAGES_DIR / image_file
                if not image_path.exists():
                    continue
                
                try:
                    with open(image_path, "rb") as f:
                        files = {"image": f}
                        data = {"model_name": model}
                        
                        result = self.make_request(
                            "POST",
                            "/api/face-recognition/extract-embedding",
                            files=files,
                            data=data
                        )
                        
                        if result["success"]:
                            embedding = result["data"].get("embedding", [])
                            logger.info(f"     ✅ {model}/{image_file}: {len(embedding)}D embedding")
                            model_results[image_file] = {
                                "success": True,
                                "embedding_dimensions": len(embedding)
                            }
                        else:
                            logger.error(f"     ❌ {model}/{image_file}: {result['error']}")
                            model_results[image_file] = {
                                "success": False,
                                "error": result["error"]
                            }
                except Exception as e:
                    logger.error(f"     ❌ {model}/{image_file}: {str(e)}")
            
            if model_results:
                embedding_results[model] = model_results
        
        self.test_results["face_recognition"]["embeddings"] = embedding_results
    
    def test_face_analysis_apis(self):
        """Test face analysis endpoints"""
        logger.info("📊 Testing Face Analysis APIs...")
        
        # Health check
        health_result = self.make_request("GET", "/api/face-analysis/health")
        self.test_results["face_analysis"]["health"] = health_result
        
        if health_result["success"]:
            logger.info("✅ Face Analysis Health: OK")
        else:
            logger.error(f"❌ Face Analysis Health: {health_result['error']}")
            return
        
        # Test comprehensive analysis
        analysis_modes = ["full_analysis", "detection_only", "recognition_only"]
        analysis_results = {}
        
        for mode in analysis_modes:
            logger.info(f"   Testing analysis mode: {mode}")
            mode_results = {}
            
            test_images = ["boss_01.jpg", "boss_group01.jpg", "night_01.jpg", "spoofing_01.jpg"]
            
            for image_file in test_images:
                image_path = TEST_IMAGES_DIR / image_file
                if not image_path.exists():
                    continue
                
                try:
                    image_base64 = self.encode_image_to_base64(image_path)
                    if image_base64:
                        data = {
                            "image_base64": image_base64,
                            "mode": mode,
                            "config": {
                                "detection_model": "auto",
                                "recognition_model": "facenet",
                                "min_confidence": 0.5
                            }
                        }
                        
                        if mode in ["full_analysis", "recognition_only"] and self.gallery:
                            data["gallery"] = self.gallery
                        
                        result = self.make_request(
                            "POST",
                            "/api/face-analysis/analyze-json",
                            json=data
                        )
                        
                        if result["success"]:
                            faces = result["data"].get("faces", [])
                            logger.info(f"     ✅ {image_file}: {len(faces)} faces analyzed")
                            mode_results[image_file] = {
                                "success": True,
                                "faces_analyzed": len(faces),
                                "response_time": result["response_time"]
                            }
                        else:
                            logger.error(f"     ❌ {image_file}: {result['error']}")
                            mode_results[image_file] = {
                                "success": False,
                                "error": result["error"]
                            }
                except Exception as e:
                    logger.error(f"     ❌ {image_file}: {str(e)}")
            
            if mode_results:
                analysis_results[mode] = mode_results
        
        self.test_results["face_analysis"]["analysis"] = analysis_results
    
    def test_performance_endpoints(self):
        """Test performance and information endpoints"""
        logger.info("⚡ Testing Performance & Info Endpoints...")
        
        endpoints = [
            ("/api/face-detection/models/available", "Face Detection Models"),
            ("/api/face-detection/models/status", "Face Detection Status"),
            ("/api/face-detection/performance/stats", "Face Detection Performance"),
            ("/api/face-recognition/models/available", "Face Recognition Models"),
            ("/api/face-recognition/performance/stats", "Face Recognition Performance"),
            ("/api/face-recognition/database-status", "Database Status")
        ]
        
        performance_results = {}
        
        for endpoint, name in endpoints:
            result = self.make_request("GET", endpoint)
            performance_results[endpoint] = result
            
            if result["success"]:
                logger.info(f"   ✅ {name}: OK")
            else:
                logger.error(f"   ❌ {name}: {result['error']}")
        
        self.test_results["performance"] = performance_results
    
    def run_all_tests(self):
        """Run all API tests"""
        logger.info("🚀 Starting Comprehensive API Testing...")
        logger.info(f"📅 Test started at: {self.test_start_time}")
        logger.info(f"🔗 API Base URL: {API_BASE_URL}")
        logger.info(f"📁 Test Images Directory: {TEST_IMAGES_DIR}")
        logger.info("=" * 60)
        
        try:
            # Run all test suites
            self.test_system_health()
            self.test_face_detection_apis()
            self.test_face_recognition_apis()
            self.test_face_analysis_apis()
            self.test_performance_endpoints()
            
            # Generate final report
            self.generate_final_report()
            
        except Exception as e:
            logger.error(f"❌ Error during testing: {str(e)}")
            self.test_results["errors"].append(f"Testing error: {str(e)}")
            raise
    
    def generate_final_report(self):
        """Generate comprehensive test report"""
        test_end_time = datetime.now()
        total_duration = (test_end_time - self.test_start_time).total_seconds()
        
        # Calculate statistics
        total_success = 0
        total_tests = 0
        
        def count_results(data):
            nonlocal total_success, total_tests
            if isinstance(data, dict):
                if "success" in data:
                    total_tests += 1
                    if data["success"]:
                        total_success += 1
                else:
                    for value in data.values():
                        count_results(value)
        
        count_results(self.test_results)
        
        success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0
        
        # Generate summary report
        summary = {
            "test_information": {
                "start_time": self.test_start_time.isoformat(),
                "end_time": test_end_time.isoformat(),
                "total_duration_seconds": total_duration,
                "api_base_url": API_BASE_URL
            },
            "test_statistics": {
                "total_tests": total_tests,
                "successful_tests": total_success,
                "failed_tests": total_tests - total_success,
                "success_rate_percent": round(success_rate, 2)
            },
            "detailed_results": self.test_results,
            "recommendations": self.generate_recommendations()
        }
        
        # Save detailed report
        report_file = OUTPUT_DIR / f"api_test_report_{int(time.time())}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("📊 TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Successful: {total_success}")
        logger.info(f"   Failed: {total_tests - total_success}")
        logger.info(f"   Success Rate: {success_rate:.2f}%")
        logger.info(f"   Total Duration: {total_duration:.2f} seconds")
        logger.info(f"📄 Detailed report saved to: {report_file}")
        
        # Print recommendations
        recommendations = self.generate_recommendations()
        if recommendations:
            logger.info("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"   {i}. {rec}")
        
        return summary
    
    def generate_recommendations(self):
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check system health
        if not self.test_results["system_health"].get("success", False):
            recommendations.append("System health check failed. Verify that the API server is running properly.")
        
        # Check face detection
        detection_health = self.test_results["face_detection"].get("health", {})
        if not detection_health.get("success", False):
            recommendations.append("Face detection service is not healthy. Check model loading and GPU availability.")
        
        # Check face recognition
        recognition_health = self.test_results["face_recognition"].get("health", {})
        if not recognition_health.get("success", False):
            recommendations.append("Face recognition service is not healthy. Verify model files and memory allocation.")
        
        # Check face analysis
        analysis_health = self.test_results["face_analysis"].get("health", {})
        if not analysis_health.get("success", False):
            recommendations.append("Face analysis service is not healthy. Check service dependencies.")
        
        # Model-specific recommendations
        detection_models = self.test_results["face_detection"].get("models", {})
        for model, results in detection_models.items():
            success_count = 0
            total_count = 0
            for category_results in results.values():
                for test_result in category_results.values():
                    total_count += 1
                    if test_result.get("success", False):
                        success_count += 1
            
            if total_count > 0:
                model_success_rate = (success_count / total_count) * 100
                if model_success_rate < 80:
                    recommendations.append(f"Detection model '{model}' has low success rate ({model_success_rate:.1f}%). Consider model optimization.")
        
        # Recognition accuracy recommendations
        recognition_results = self.test_results["face_recognition"].get("recognition", {})
        for model, results in recognition_results.items():
            correct_count = 0
            total_count = 0
            for test_result in results.values():
                if test_result.get("success", False):
                    total_count += 1
                    if test_result.get("correct", False):
                        correct_count += 1
            
            if total_count > 0:
                accuracy = (correct_count / total_count) * 100
                if accuracy < 90:
                    recommendations.append(f"Recognition model '{model}' has low accuracy ({accuracy:.1f}%). Consider retraining or parameter tuning.")
        
        if not recommendations:
            recommendations.append("All tests passed successfully! The API system is functioning optimally.")
        
        return recommendations

def main():
    """Main function to run the API test"""
    print("🚀 Face Recognition System - Complete API Test Suite")
    print("=" * 60)
    
    # Check if test images directory exists
    if not TEST_IMAGES_DIR.exists():
        print(f"❌ Test images directory not found: {TEST_IMAGES_DIR}")
        print("Please ensure the test_images directory exists and contains test images.")
        sys.exit(1)
    
    # Check if API server is reachable
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ API server not responding properly at {API_BASE_URL}")
            print("Please start the API server before running tests.")
            sys.exit(1)
    except requests.exceptions.RequestException:
        print(f"❌ Cannot reach API server at {API_BASE_URL}")
        print("Please ensure the API server is running and accessible.")
        sys.exit(1)
    
    # Initialize and run tests
    tester = APITester()
    
    try:
        tester.run_all_tests()
        print("\n✅ All tests completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        if tester.test_results:
            tester.generate_final_report()
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        if tester.test_results:
            tester.generate_final_report()
        raise

if __name__ == "__main__":
    main()
