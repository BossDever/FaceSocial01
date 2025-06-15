#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive API Testing Script for Face Recognition System
Created: June 14, 2025
Purpose: Test all API endpoints with various test images and models
"""

import os
import sys
import json
import time
import base64
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_api_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8080"
TEST_IMAGES_DIR = Path("test_images")
OUTPUT_DIR = Path("output/comprehensive_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class TestResult:
    endpoint: str
    method: str
    status_code: int
    success: bool
    response_time: float
    response_data: Dict[str, Any]
    error_message: Optional[str] = None
    test_details: Optional[Dict[str, Any]] = None

class APITester:
    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 30
        self.results: List[TestResult] = []
        self.test_start_time = datetime.now()
        
        # Test images by category
        self.test_images = {
            "boss_single": [f"boss_{i:02d}.jpg" for i in range(1, 14)],
            "boss_glasses": [f"boss_glass{i:02d}.jpg" for i in range(1, 13)],
            "boss_group": ["boss_group01.jpg", "boss_group02.jpg", "boss_group03.jpg"],
            "night_single": [f"night_{i:02d}.jpg" for i in range(1, 11)],
            "night_group": ["night_group01.jpg", "night_group02.jpg"],
            "spoofing": [f"spoofing_{i:02d}.jpg" for i in range(1, 5)],
            "face_swap": [f"face-swap{i:02d}.png" for i in range(1, 4)]
        }
        
        # Models to test
        self.detection_models = ["auto", "yolov9c", "yolov9e", "yolov11m"]
        self.recognition_models = ["facenet", "adaface", "arcface"]
        
        # Gallery for recognition tests
        self.gallery = {}
        
    def encode_image_to_base64(self, image_path: Path) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def make_request(self, method: str, endpoint: str, **kwargs) -> TestResult:
        """Make HTTP request and return TestResult"""
        url = f"{API_BASE_URL}{endpoint}"
        start_time = time.time()
        
        try:
            response = self.session.request(method, url, **kwargs)
            response_time = time.time() - start_time
            
            # Try to parse JSON response
            try:
                response_data = response.json()
            except:
                response_data = {"raw_response": response.text}
            
            success = response.status_code == 200
            error_message = None if success else f"HTTP {response.status_code}: {response.text}"
            
            result = TestResult(
                endpoint=endpoint,
                method=method,
                status_code=response.status_code,
                success=success,
                response_time=response_time,
                response_data=response_data,
                error_message=error_message
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            result = TestResult(
                endpoint=endpoint,
                method=method,
                status_code=0,
                success=False,
                response_time=response_time,
                response_data={},
                error_message=str(e)
            )
        
        self.results.append(result)
        return result
    
    def test_system_health(self):
        """Test system health endpoint"""
        logger.info("🔍 Testing System Health...")
        result = self.make_request("GET", "/health")
        
        if result.success:
            logger.info("✅ System Health: OK")
            logger.info(f"   Services: {result.response_data.get('active_services', 'Unknown')}")
        else:
            logger.error(f"❌ System Health: Failed - {result.error_message}")
    
    def test_face_detection_health(self):
        """Test face detection health endpoint"""
        logger.info("🔍 Testing Face Detection Health...")
        result = self.make_request("GET", "/api/face-detection/health")
        
        if result.success:
            logger.info("✅ Face Detection Health: OK")
            service_info = result.response_data.get('service_info', {})
            logger.info(f"   Models: {service_info.get('available_models', 'Unknown')}")
        else:
            logger.error(f"❌ Face Detection Health: Failed - {result.error_message}")
    
    def test_face_detection_models(self):
        """Test face detection with different models"""
        logger.info("🔍 Testing Face Detection Models...")
        
        # Test each model with different image categories
        test_cases = [
            ("boss_single", "boss_01.jpg"),
            ("boss_glasses", "boss_glass01.jpg"),
            ("boss_group", "boss_group01.jpg"),
            ("night_single", "night_01.jpg"),
            ("spoofing", "spoofing_01.jpg")
        ]
        
        for model in self.detection_models:
            logger.info(f"   Testing model: {model}")
            
            for category, image_file in test_cases:
                image_path = TEST_IMAGES_DIR / image_file
                if not image_path.exists():
                    continue
                
                # Test with file upload
                with open(image_path, "rb") as f:
                    files = {"image": f}
                    data = {
                        "model_name": model,
                        "conf_threshold": 0.5,
                        "iou_threshold": 0.4
                    }
                    
                    result = self.make_request(
                        "POST", 
                        "/api/face-detection/detect",
                        files=files,
                        data=data
                    )
                    
                    result.test_details = {
                        "model": model,
                        "image_category": category,
                        "image_file": image_file
                    }
                    
                    if result.success:
                        faces_detected = len(result.response_data.get('faces', []))
                        logger.info(f"     ✅ {category}/{image_file}: {faces_detected} faces detected")
                    else:
                        logger.error(f"     ❌ {category}/{image_file}: {result.error_message}")
                
                # Test with base64
                try:
                    image_base64 = self.encode_image_to_base64(image_path)
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
                    
                    result.test_details = {
                        "model": model,
                        "image_category": category,
                        "image_file": image_file,
                        "method": "base64"
                    }
                    
                    if result.success:
                        faces_detected = len(result.response_data.get('faces', []))
                        logger.info(f"     ✅ {category}/{image_file} (Base64): {faces_detected} faces detected")
                    else:
                        logger.error(f"     ❌ {category}/{image_file} (Base64): {result.error_message}")
                        
                except Exception as e:
                    logger.error(f"     ❌ {category}/{image_file} (Base64): {str(e)}")
    
    def test_face_recognition_health(self):
        """Test face recognition health endpoint"""
        logger.info("🔍 Testing Face Recognition Health...")
        result = self.make_request("GET", "/api/face-recognition/health")
        
        if result.success:
            logger.info("✅ Face Recognition Health: OK")
            service_info = result.response_data.get('service_info', {})
            logger.info(f"   Models: {service_info.get('available_models', 'Unknown')}")
        else:
            logger.error(f"❌ Face Recognition Health: Failed - {result.error_message}")
    
    def test_face_recognition_add_faces(self):
        """Test adding faces to gallery"""
        logger.info("🔍 Testing Face Recognition - Adding Faces to Gallery...")
        
        # Add boss faces
        boss_images = ["boss_01.jpg", "boss_02.jpg", "boss_03.jpg"]
        for i, image_file in enumerate(boss_images):
            image_path = TEST_IMAGES_DIR / image_file
            if not image_path.exists():
                continue
                
            try:
                image_base64 = self.encode_image_to_base64(image_path)
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
                
                result.test_details = {
                    "person_id": "boss",
                    "image_file": image_file,
                    "model": "facenet"
                }
                
                if result.success:
                    logger.info(f"     ✅ Added {image_file} for boss")
                else:
                    logger.error(f"     ❌ Failed to add {image_file}: {result.error_message}")
                    
            except Exception as e:
                logger.error(f"     ❌ Error adding {image_file}: {str(e)}")
        
        # Add night faces
        night_images = ["night_01.jpg", "night_02.jpg", "night_03.jpg"]
        for i, image_file in enumerate(night_images):
            image_path = TEST_IMAGES_DIR / image_file
            if not image_path.exists():
                continue
                
            try:
                image_base64 = self.encode_image_to_base64(image_path)
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
                
                result.test_details = {
                    "person_id": "night",
                    "image_file": image_file,
                    "model": "facenet"
                }
                
                if result.success:
                    logger.info(f"     ✅ Added {image_file} for night")
                else:
                    logger.error(f"     ❌ Failed to add {image_file}: {result.error_message}")
                    
            except Exception as e:
                logger.error(f"     ❌ Error adding {image_file}: {str(e)}")
    
    def test_face_recognition_gallery_management(self):
        """Test gallery management endpoints"""
        logger.info("🔍 Testing Face Recognition - Gallery Management...")
        
        # Get gallery
        result = self.make_request("GET", "/api/face-recognition/get-gallery")
        if result.success:
            gallery_data = result.response_data.get('gallery', {})
            logger.info(f"   ✅ Gallery retrieved: {len(gallery_data)} persons")
            self.gallery = gallery_data
        else:
            logger.error(f"   ❌ Failed to get gallery: {result.error_message}")
        
        # Get database status
        result = self.make_request("GET", "/api/face-recognition/database-status")
        if result.success:
            status = result.response_data
            logger.info(f"   ✅ Database status: {status.get('total_persons', 0)} persons, {status.get('total_faces', 0)} faces")
        else:
            logger.error(f"   ❌ Failed to get database status: {result.error_message}")
    
    def test_face_recognition_models(self):
        """Test face recognition with different models"""
        logger.info("🔍 Testing Face Recognition Models...")
        
        test_cases = [
            ("boss", "boss_04.jpg"),  # Different boss image
            ("boss", "boss_glass01.jpg"),  # Boss with glasses
            ("night", "night_04.jpg"),  # Different night image
            ("unknown", "spoofing_01.jpg")  # Unknown person
        ]
        
        for model in self.recognition_models:
            logger.info(f"   Testing model: {model}")
            
            for expected_person, image_file in test_cases:
                image_path = TEST_IMAGES_DIR / image_file
                if not image_path.exists():
                    continue
                
                try:
                    image_base64 = self.encode_image_to_base64(image_path)
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
                    
                    result.test_details = {
                        "model": model,
                        "expected_person": expected_person,
                        "image_file": image_file
                    }
                    
                    if result.success:
                        matches = result.response_data.get('matches', [])
                        if matches:
                            top_match = matches[0]
                            confidence = top_match.get('confidence', 0)
                            person_id = top_match.get('person_id', 'unknown')
                            logger.info(f"     ✅ {image_file}: {person_id} (confidence: {confidence:.3f})")
                        else:
                            logger.info(f"     ✅ {image_file}: No matches found")
                    else:
                        logger.error(f"     ❌ {image_file}: {result.error_message}")
                        
                except Exception as e:
                    logger.error(f"     ❌ {image_file}: {str(e)}")
    
    def test_face_recognition_extract_embedding(self):
        """Test face embedding extraction"""
        logger.info("🔍 Testing Face Recognition - Extract Embedding...")
        
        test_images = ["boss_01.jpg", "night_01.jpg", "boss_glass01.jpg"]
        
        for model in self.recognition_models:
            logger.info(f"   Testing model: {model}")
            
            for image_file in test_images:
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
                        
                        result.test_details = {
                            "model": model,
                            "image_file": image_file
                        }
                        
                        if result.success:
                            embedding = result.response_data.get('embedding', [])
                            logger.info(f"     ✅ {image_file}: Embedding extracted ({len(embedding)} dimensions)")
                        else:
                            logger.error(f"     ❌ {image_file}: {result.error_message}")
                            
                except Exception as e:
                    logger.error(f"     ❌ {image_file}: {str(e)}")
    
    def test_face_analysis_health(self):
        """Test face analysis health endpoint"""
        logger.info("🔍 Testing Face Analysis Health...")
        result = self.make_request("GET", "/api/face-analysis/health")
        
        if result.success:
            logger.info("✅ Face Analysis Health: OK")
            service_info = result.response_data.get('service_info', {})
            logger.info(f"   Capabilities: {service_info.get('capabilities', 'Unknown')}")
        else:
            logger.error(f"❌ Face Analysis Health: Failed - {result.error_message}")
    
    def test_face_analysis_comprehensive(self):
        """Test comprehensive face analysis"""
        logger.info("🔍 Testing Face Analysis - Comprehensive Analysis...")
        
        analysis_modes = ["full_analysis", "detection_only", "recognition_only"]
        test_cases = [
            ("boss_single", "boss_01.jpg"),
            ("boss_glasses", "boss_glass01.jpg"),
            ("boss_group", "boss_group01.jpg"),
            ("night_single", "night_01.jpg"),
            ("spoofing", "spoofing_01.jpg"),
            ("face_swap", "face-swap01.png")
        ]
        
        for mode in analysis_modes:
            logger.info(f"   Testing mode: {mode}")
            
            for category, image_file in test_cases:
                image_path = TEST_IMAGES_DIR / image_file
                if not image_path.exists():
                    continue
                
                try:
                    image_base64 = self.encode_image_to_base64(image_path)
                    data = {
                        "image_base64": image_base64,
                        "mode": mode,
                        "config": {
                            "detection_model": "auto",
                            "recognition_model": "facenet",
                            "min_confidence": 0.5
                        },
                        "gallery": self.gallery if mode in ["full_analysis", "recognition_only"] else None
                    }
                    
                    result = self.make_request(
                        "POST",
                        "/api/face-analysis/analyze-json",
                        json=data
                    )
                    
                    result.test_details = {
                        "mode": mode,
                        "image_category": category,
                        "image_file": image_file
                    }
                    
                    if result.success:
                        faces = result.response_data.get('faces', [])
                        analysis_results = result.response_data.get('analysis_results', {})
                        logger.info(f"     ✅ {category}/{image_file}: {len(faces)} faces analyzed")
                        
                        if analysis_results:
                            for key, value in analysis_results.items():
                                if isinstance(value, (int, float)):
                                    logger.info(f"        {key}: {value}")
                    else:
                        logger.error(f"     ❌ {category}/{image_file}: {result.error_message}")
                        
                except Exception as e:
                    logger.error(f"     ❌ {category}/{image_file}: {str(e)}")
    
    def test_performance_stats(self):
        """Test performance statistics endpoints"""
        logger.info("🔍 Testing Performance Statistics...")
        
        endpoints = [
            "/api/face-detection/performance/stats",
            "/api/face-recognition/performance/stats"
        ]
        
        for endpoint in endpoints:
            result = self.make_request("GET", endpoint)
            if result.success:
                stats = result.response_data.get('stats', {})
                logger.info(f"   ✅ {endpoint}: {len(stats)} metrics available")
            else:
                logger.error(f"   ❌ {endpoint}: {result.error_message}")
    
    def test_model_information(self):
        """Test model information endpoints"""
        logger.info("🔍 Testing Model Information...")
        
        endpoints = [
            "/api/face-detection/models/available",
            "/api/face-detection/models/status",
            "/api/face-recognition/models/available"
        ]
        
        for endpoint in endpoints:
            result = self.make_request("GET", endpoint)
            if result.success:
                models = result.response_data.get('models', result.response_data.get('available_models', []))
                logger.info(f"   ✅ {endpoint}: {len(models)} models available")
            else:
                logger.error(f"   ❌ {endpoint}: {result.error_message}")
    
    def run_all_tests(self):
        """Run all API tests"""
        logger.info("🚀 Starting Comprehensive API Testing...")
        logger.info(f"📅 Test started at: {self.test_start_time}")
        logger.info(f"🔗 API Base URL: {API_BASE_URL}")
        logger.info(f"📁 Test Images Directory: {TEST_IMAGES_DIR}")
        
        # System health tests
        self.test_system_health()
        
        # Face detection tests
        self.test_face_detection_health()
        self.test_face_detection_models()
        
        # Face recognition tests
        self.test_face_recognition_health()
        self.test_face_recognition_add_faces()
        self.test_face_recognition_gallery_management()
        self.test_face_recognition_models()
        self.test_face_recognition_extract_embedding()
        
        # Face analysis tests
        self.test_face_analysis_health()
        self.test_face_analysis_comprehensive()
        
        # Performance and model info tests
        self.test_performance_stats()
        self.test_model_information()
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report"""
        test_end_time = datetime.now()
        total_duration = (test_end_time - self.test_start_time).total_seconds()
        
        # Calculate statistics
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Group results by endpoint
        endpoint_stats = {}
        for result in self.results:
            endpoint = result.endpoint
            if endpoint not in endpoint_stats:
                endpoint_stats[endpoint] = {"total": 0, "success": 0, "failed": 0, "avg_time": 0}
            
            endpoint_stats[endpoint]["total"] += 1
            if result.success:
                endpoint_stats[endpoint]["success"] += 1
            else:
                endpoint_stats[endpoint]["failed"] += 1
            endpoint_stats[endpoint]["avg_time"] += result.response_time
        
        # Calculate average times
        for endpoint in endpoint_stats:
            if endpoint_stats[endpoint]["total"] > 0:
                endpoint_stats[endpoint]["avg_time"] /= endpoint_stats[endpoint]["total"]
        
        # Generate report
        report = {
            "test_summary": {
                "start_time": self.test_start_time.isoformat(),
                "end_time": test_end_time.isoformat(),
                "total_duration_seconds": total_duration,
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate_percent": round(success_rate, 2)
            },
            "endpoint_statistics": endpoint_stats,
            "detailed_results": [
                {
                    "endpoint": r.endpoint,
                    "method": r.method,
                    "status_code": r.status_code,
                    "success": r.success,
                    "response_time": r.response_time,
                    "error_message": r.error_message,
                    "test_details": r.test_details
                }
                for r in self.results
            ],
            "model_test_results": self._analyze_model_performance(),
            "image_category_results": self._analyze_image_category_performance(),
            "recommendations": self._generate_recommendations()
        }
        
        # Save report
        report_file = OUTPUT_DIR / f"comprehensive_test_report_{int(time.time())}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        logger.info("📊 Test Results Summary:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Successful: {successful_tests}")
        logger.info(f"   Failed: {failed_tests}")
        logger.info(f"   Success Rate: {success_rate:.2f}%")
        logger.info(f"   Total Duration: {total_duration:.2f} seconds")
        logger.info(f"📄 Detailed report saved to: {report_file}")
        
        return report
    
    def _analyze_model_performance(self) -> Dict[str, Any]:
        """Analyze performance by model"""
        model_performance = {}
        
        for result in self.results:
            if result.test_details and 'model' in result.test_details:
                model = result.test_details['model']
                if model not in model_performance:
                    model_performance[model] = {"total": 0, "success": 0, "avg_time": 0}
                
                model_performance[model]["total"] += 1
                if result.success:
                    model_performance[model]["success"] += 1
                model_performance[model]["avg_time"] += result.response_time
        
        # Calculate averages and success rates
        for model in model_performance:
            stats = model_performance[model]
            if stats["total"] > 0:
                stats["avg_time"] /= stats["total"]
                stats["success_rate"] = (stats["success"] / stats["total"]) * 100
        
        return model_performance
    
    def _analyze_image_category_performance(self) -> Dict[str, Any]:
        """Analyze performance by image category"""
        category_performance = {}
        
        for result in self.results:
            if result.test_details and 'image_category' in result.test_details:
                category = result.test_details['image_category']
                if category not in category_performance:
                    category_performance[category] = {"total": 0, "success": 0, "avg_time": 0}
                
                category_performance[category]["total"] += 1
                if result.success:
                    category_performance[category]["success"] += 1
                category_performance[category]["avg_time"] += result.response_time
        
        # Calculate averages and success rates
        for category in category_performance:
            stats = category_performance[category]
            if stats["total"] > 0:
                stats["avg_time"] /= stats["total"]
                stats["success_rate"] = (stats["success"] / stats["total"]) * 100
        
        return category_performance
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check overall success rate
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        if success_rate < 90:
            recommendations.append("Overall success rate is below 90%. Investigate failed endpoints.")
        
        # Check response times
        avg_response_time = sum(r.response_time for r in self.results) / total_tests if total_tests > 0 else 0
        if avg_response_time > 5:
            recommendations.append("Average response time is high (>5s). Consider optimizing performance.")
        
        # Check for consistent failures
        failed_endpoints = {}
        for result in self.results:
            if not result.success:
                endpoint = result.endpoint
                failed_endpoints[endpoint] = failed_endpoints.get(endpoint, 0) + 1
        
        for endpoint, count in failed_endpoints.items():
            if count > 3:
                recommendations.append(f"Endpoint {endpoint} has {count} failures. Needs investigation.")
        
        if not recommendations:
            recommendations.append("All tests performed well. System is functioning optimally.")
        
        return recommendations

def main():
    """Main function to run the comprehensive API test"""
    print("🚀 Face Recognition System - Comprehensive API Test")
    print("=" * 60)
    
    # Check if test images directory exists
    if not TEST_IMAGES_DIR.exists():
        print(f"❌ Test images directory not found: {TEST_IMAGES_DIR}")
        sys.exit(1)
    
    # Initialize tester
    tester = APITester()
    
    try:
        # Run all tests
        tester.run_all_tests()
        print("\n✅ All tests completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        if tester.results:
            tester.generate_report()
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        if tester.results:
            tester.generate_report()
        raise

if __name__ == "__main__":
    main()
