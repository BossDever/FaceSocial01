#!/usr/bin/env python3
"""ทดสอบเปรียบเทียบ Recognition vs Registration"""

import requests

def test_recognition_vs_registration():
    print('🧪 Testing REGISTRATION vs RECOGNITION for framework models:')
    
    test_models = ['deepface', 'facenet_pytorch', 'dlib', 'insightface', 'edgeface']
    
    for model in test_models:
        print(f'\n📋 Testing {model.upper()}:')
        
        # Test Recognition (ที่เราทราบว่าทำงานได้)
        print('  1. Recognition endpoint:')
        recognition_data = {
            'face_image_base64': 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//2Q=',
            'gallery': {'test': [{'embedding': [0.1] * 512, 'person_name': 'Test'}]},
            'model_name': model,
            'top_k': 1,
            'similarity_threshold': 0.3
        }
        
        try:
            response = requests.post('http://localhost:8080/api/face-recognition/recognize',
                                   json=recognition_data, timeout=30)
            if response.status_code == 200:
                print(f'    ✅ Recognition: SUCCESS')
            else:
                print(f'    ❌ Recognition: FAILED ({response.status_code})')
                print(f'       Error: {response.text[:100]}')
        except Exception as e:
            print(f'    ❌ Recognition: Exception - {str(e)[:100]}')
        
        # Test Registration (ที่ล้มเหลว)  
        print('  2. Registration endpoint:')
        registration_data = {
            'person_id': 'test_person',
            'person_name': 'Test Person',
            'face_image_base64': 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//2Q=',
            'model_name': model
        }
        
        try:
            response = requests.post('http://localhost:8080/api/face-recognition/add-face-json',
                                   json=registration_data, timeout=30)
            if response.status_code == 200:
                print(f'    ✅ Registration: SUCCESS')
            else:
                print(f'    ❌ Registration: FAILED ({response.status_code})')
                print(f'       Error: {response.text[:100]}')
        except Exception as e:
            print(f'    ❌ Registration: Exception - {str(e)[:100]}')
    
    print('\n📊 CONCLUSION:')
    print('- If recognition works but registration fails:')
    print('  → Framework models may not support registration mode')
    print('  → Use only ONNX models for registration')
    print('  → Use all models for recognition testing')

if __name__ == "__main__":
    test_recognition_vs_registration()
