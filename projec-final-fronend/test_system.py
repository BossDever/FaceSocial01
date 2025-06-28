#!/usr/bin/env python3
"""
Test script สำหรับทดสอบระบบ Face Recognition Registration และ Login
รองรับการทดสอบทั้ง API และ UI Components
"""

import requests
import json
import time
import base64
import os
from typing import Dict, Any, Optional

class FaceRecognitionTester:
    def __init__(self):
        self.frontend_url = "http://localhost:3000"
        self.backend_url = "http://localhost:8080"
        self.test_results = []
        
    def log_test(self, test_name: str, success: bool, message: str, data: Dict = None):
        """บันทึกผลการทดสอบ"""
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data": data
        }
        self.test_results.append(result)
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name} - {message}")
        
    def test_health_checks(self):
        """ทดสอบ Health Checks ของทุกระบบ"""
        print("\n=== Testing Health Checks ===")
        
        # Test Frontend Health
        try:
            response = requests.get(f"{self.frontend_url}/api/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.log_test("Frontend Health", True, f"Status: {data.get('status')}", data)
            else:
                self.log_test("Frontend Health", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Frontend Health", False, f"Connection error: {str(e)}")
            
        # Test Backend Health
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.log_test("Backend Health", True, f"Status: {data.get('status')}", data)
            else:
                self.log_test("Backend Health", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Backend Health", False, f"Connection error: {str(e)}")
    
    def test_user_availability(self):
        """ทดสอบการตรวจสอบ username/email ว่าสามารถใช้ได้หรือไม่"""
        print("\n=== Testing User Availability ===")
        
        # Test existing email
        try:
            response = requests.get(f"{self.frontend_url}/api/auth/register?email=test@example.com")
            if response.status_code == 200:
                data = response.json()
                available = data.get('available', True)
                if not available:  # Should not be available (already exists)
                    self.log_test("Email Availability Check", True, "Existing email correctly detected", data)
                else:
                    self.log_test("Email Availability Check", False, "Existing email not detected", data)
            else:
                self.log_test("Email Availability Check", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Email Availability Check", False, f"Error: {str(e)}")
            
        # Test new email
        try:
            test_email = f"newuser{int(time.time())}@test.com"
            response = requests.get(f"{self.frontend_url}/api/auth/register?email={test_email}")
            if response.status_code == 200:
                data = response.json()
                available = data.get('available', False)
                if available:  # Should be available (new email)
                    self.log_test("New Email Availability", True, "New email correctly available", data)
                else:
                    self.log_test("New Email Availability", False, "New email incorrectly unavailable", data)
            else:
                self.log_test("New Email Availability", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.log_test("New Email Availability", False, f"Error: {str(e)}")    def test_face_detection_api(self):
        """ทดสอบ Face Detection API"""
        print("\n=== Testing Face Detection API ===")
        
        # Check if test images exist
        test_image_path = "../test_images/boss_01.jpg"
        if not os.path.exists(test_image_path):
            self.log_test("Face Detection API", False, f"Test image not found: {test_image_path}")
            return
        
        try:
            with open(test_image_path, 'rb') as f:
                test_image_data = f.read()
            
            files = {'file': ('test.jpg', test_image_data, 'image/jpeg')}
            data = {
                'model_name': 'auto',
                'conf_threshold': '0.5',
                'max_faces': '1'
            }
            
            response = requests.post(f"{self.backend_url}/api/face-detection/detect", 
                                   files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success') and result.get('faces'):
                    self.log_test("Face Detection API", True, f"Detected {len(result['faces'])} face(s)", result)
                else:
                    self.log_test("Face Detection API", False, "No faces detected", result)
            else:
                self.log_test("Face Detection API", False, f"HTTP {response.status_code}: {response.text}")
        except Exception as e:
            self.log_test("Face Detection API", False, f"Error: {str(e)}")
    
    def test_database_connection(self):
        """ทดสอบการเชื่อมต่อกับฐานข้อมูล"""
        print("\n=== Testing Database Connection ===")
        
        # Test through health endpoint
        try:
            response = requests.get(f"{self.frontend_url}/api/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                db_status = data.get('services', {}).get('database')
                if db_status == 'connected':
                    self.log_test("Database Connection", True, "Database connected successfully")
                else:
                    self.log_test("Database Connection", False, f"Database status: {db_status}")
            else:
                self.log_test("Database Connection", False, f"Health check failed: HTTP {response.status_code}")
        except Exception as e:
            self.log_test("Database Connection", False, f"Error: {str(e)}")
    
    def test_login_with_existing_user(self):
        """ทดสอบการเข้าสู่ระบบด้วยรหัสผ่าน"""
        print("\n=== Testing Password Login ===")
        
        login_data = {
            "email": "test@example.com",
            "password": "password123",
            "method": "password"
        }
        
        try:
            response = requests.post(f"{self.frontend_url}/api/auth/login",
                                   json=login_data, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    self.log_test("Password Login", True, "Login successful", data)
                else:
                    self.log_test("Password Login", False, f"Login failed: {data.get('message')}", data)
            else:
                try:
                    error_data = response.json()
                    self.log_test("Password Login", False, f"HTTP {response.status_code}: {error_data.get('message', 'Unknown error')}")
                except:
                    self.log_test("Password Login", False, f"HTTP {response.status_code}: {response.text}")
        except Exception as e:
            self.log_test("Password Login", False, f"Error: {str(e)}")
    
    def print_summary(self):
        """แสดงสรุปผลการทดสอบ"""
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✓ Passed: {passed_tests}")
        print(f"✗ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "No tests run")
        
        if failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.test_results:
                if not result['success']:
                    print(f"  • {result['test']}: {result['message']}")
        
        print("\n✅ PASSED TESTS:")
        for result in self.test_results:
            if result['success']:
                print(f"  • {result['test']}: {result['message']}")
                
        # Save detailed results to file
        with open('test_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        print(f"\n📁 Detailed results saved to: test_results.json")
    
    def run_all_tests(self):
        """รันการทดสอบทั้งหมด"""
        print("🚀 Starting Face Recognition System Tests...")
        print(f"Frontend URL: {self.frontend_url}")
        print(f"Backend URL: {self.backend_url}")
        
        # รันการทดสอบตามลำดับ
        self.test_health_checks()
        self.test_database_connection()
        self.test_user_availability()
        self.test_face_detection_api()
        self.test_login_with_existing_user()
        
        # แสดงสรุป
        self.print_summary()

if __name__ == "__main__":
    tester = FaceRecognitionTester()
    tester.run_all_tests()
