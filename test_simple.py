#!/usr/bin/env python3
"""
Simple test script to verify imports and basic functionality
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all major imports"""
    print("🧪 Testing imports...")
    
    try:
        # Test core imports
        from src.core.log_config import get_logger
        print("✅ Core log_config import successful")
        
        # Test face recognition service
        from src.ai_services.face_recognition.face_recognition_service import FaceRecognitionService
        print("✅ FaceRecognitionService import successful")
        
        # Test detection utils (this was causing issues)
        from src.ai_services.face_detection import get_detection_utils
        utils = get_detection_utils()
        print("✅ Face detection utils import successful")
        print(f"   Available utils: {list(utils.keys())}")
        
        # Test multi-framework imports
        frameworks_available = []
        
        try:
            import deepface
            frameworks_available.append("DeepFace")
        except ImportError:
            print("⚠️  DeepFace not available")
            
        try:
            import dlib
            frameworks_available.append("Dlib")
        except ImportError:
            print("⚠️  Dlib not available")
            
        try:
            import insightface
            frameworks_available.append("InsightFace")
        except ImportError:
            print("⚠️  InsightFace not available")
            
        try:
            import facenet_pytorch
            frameworks_available.append("FaceNet-PyTorch")
        except ImportError:
            print("⚠️  FaceNet-PyTorch not available")
            
        print(f"✅ Available frameworks: {frameworks_available}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_service_initialization():
    """Test service initialization"""
    print("\n🔧 Testing service initialization...")
    
    try:
        from src.ai_services.face_recognition.face_recognition_service import FaceRecognitionService
        
        # Test basic initialization
        service = FaceRecognitionService()
        print("✅ Basic service initialization successful")
        
        # Test multi-framework initialization
        service_multi = FaceRecognitionService(
            enable_multi_framework=True,
            config={"preferred_model": "facenet"}
        )
        print("✅ Multi-framework service initialization successful")
        
        # Test framework detection
        available = service_multi.get_available_frameworks()
        print(f"✅ Available frameworks: {available}")
        
        return True
        
    except Exception as e:
        print(f"❌ Service initialization test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting simple face recognition tests...\n")
    
    # Test imports
    imports_ok = test_imports()
    
    # Test service initialization
    service_ok = test_service_initialization()
    
    # Summary
    print(f"\n📊 Test Results:")
    print(f"   Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"   Service: {'✅ PASS' if service_ok else '❌ FAIL'}")
    
    if imports_ok and service_ok:
        print("\n🎉 All basic tests passed! The system is ready for testing.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
