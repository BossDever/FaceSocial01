#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Endpoint Checker
ตรวจสอบ API endpoints ที่มีอยู่ทั้งหมด
"""

import requests
import json
from typing import Dict, Any

API_BASE_URL = "http://localhost:8080"

def check_endpoint(endpoint: str, method: str = "GET") -> Dict[str, Any]:
    """ตรวจสอบ endpoint"""
    try:
        if method.upper() == "GET":
            response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=10)
        else:
            response = requests.post(f"{API_BASE_URL}{endpoint}", timeout=10)
        
        return {
            "endpoint": endpoint,
            "method": method,
            "status": response.status_code,
            "available": response.status_code in [200, 422],  # 422 = validation error (endpoint exists)
            "response_size": len(response.text)
        }
    except Exception as e:
        return {
            "endpoint": endpoint,
            "method": method,
            "status": "ERROR",
            "available": False,
            "error": str(e)
        }

def get_openapi_spec():
    """ดึง OpenAPI specification"""
    try:
        response = requests.get(f"{API_BASE_URL}/openapi.json", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"❌ Cannot get OpenAPI spec: {e}")
    return None

def main():
    print("🔍 API Endpoint Checker")
    print("=" * 50)
    
    # ตรวจสอบการเชื่อมต่อ
    print("📡 Checking server connection...")
    health_check = check_endpoint("/health")
    if not health_check["available"]:
        print(f"❌ Server not available: {health_check}")
        return
    
    print("✅ Server is running!")
    print()
    
    # ดึง OpenAPI spec
    print("📋 Getting API specification...")
    openapi_spec = get_openapi_spec()
    
    if openapi_spec:
        print("✅ OpenAPI spec available")
        print(f"📄 API Title: {openapi_spec.get('info', {}).get('title', 'Unknown')}")
        print(f"📄 API Version: {openapi_spec.get('info', {}).get('version', 'Unknown')}")
        print()
        
        # แสดง endpoints จาก OpenAPI spec
        print("🔗 Available Endpoints from OpenAPI:")
        print("-" * 50)
        
        paths = openapi_spec.get("paths", {})
        endpoint_count = 0
        
        for path, methods in paths.items():
            for method, details in methods.items():
                if method.upper() in ["GET", "POST", "PUT", "DELETE"]:
                    endpoint_count += 1
                    summary = details.get("summary", "No description")
                    print(f"{method.upper():6} {path:50} | {summary}")
        
        print(f"\n📊 Total endpoints: {endpoint_count}")
        
    else:
        print("⚠️ OpenAPI spec not available, checking common endpoints...")
        
        # รายการ endpoints ทั่วไป
        common_endpoints = [
            ("/health", "GET"),
            ("/docs", "GET"),
            ("/redoc", "GET"),
            ("/openapi.json", "GET"),
            
            # Face Detection
            ("/api/face-detection/health", "GET"),
            ("/api/face-detection/models/available", "GET"),
            ("/api/face-detection/detect", "POST"),
            ("/api/face-detection/detect-base64", "POST"),
            
            # Face Recognition
            ("/api/face-recognition/health", "GET"),
            ("/api/face-recognition/models/available", "GET"),
            ("/api/face-recognition/recognize", "POST"),
            ("/api/face-recognition/add-face", "POST"),
            ("/api/face-recognition/add-face-json", "POST"),
            ("/api/face-recognition/get-gallery", "GET"),
            ("/api/face-recognition/clear-gallery", "POST"),
            ("/api/face-recognition/gallery-stats", "GET"),
            ("/api/face-recognition/database-status", "GET"),
            
            # Face Analysis
            ("/api/face-analysis/health", "GET"),
            ("/api/face-analysis/analyze", "POST"),
            ("/api/face-analysis/analyze-json", "POST"),
        ]
        
        print("🔗 Checking Common Endpoints:")
        print("-" * 70)
        
        available_count = 0
        for endpoint, method in common_endpoints:
            result = check_endpoint(endpoint, method)
            status_icon = "✅" if result["available"] else "❌"
            status_text = result["status"]
            
            print(f"{status_icon} {method:6} {endpoint:45} | Status: {status_text}")
            
            if result["available"]:
                available_count += 1
        
        print(f"\n📊 Available endpoints: {available_count}/{len(common_endpoints)}")
    
    # ตรวจสอบ service status
    print("\n🏥 Service Health Status:")
    print("-" * 30)
    
    services = [
        ("Face Detection", "/api/face-detection/health"),
        ("Face Recognition", "/api/face-recognition/health"),
        ("Face Analysis", "/api/face-analysis/health")
    ]
    
    for service_name, health_endpoint in services:
        result = check_endpoint(health_endpoint)
        status_icon = "✅" if result["available"] else "❌"
        print(f"{status_icon} {service_name}")
    
    print("\n🌐 Web Interfaces:")
    print("-" * 20)
    print(f"📖 Swagger UI: {API_BASE_URL}/docs")
    print(f"📖 Redoc: {API_BASE_URL}/redoc")
    print(f"🔧 Health Check: {API_BASE_URL}/health")
    
    print("\n✅ API Check Complete!")

if __name__ == "__main__":
    main()
