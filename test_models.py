#!/usr/bin/env python3
"""ทดสอบโมเดลที่มีอยู่ในระบบ"""

import requests
import json

def test_models():
    test_models = ['facenet', 'adaface', 'arcface', 'deepface', 'facenet_pytorch', 'dlib', 'insightface', 'edgeface']
    
    print('🔧 Testing Face Recognition models:')
    working_models = []
    
    for model in test_models:
        try:
            test_data = {
                'face_image_base64': 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//2Q=',
                'gallery': {'test': [{'embedding': [0.1] * 512, 'person_name': 'Test'}]},
                'model_name': model,
                'top_k': 1,
                'similarity_threshold': 0.5
            }
            
            response = requests.post('http://localhost:8080/api/face-recognition/recognize', 
                                   json=test_data, timeout=15)
            
            if response.status_code == 200:
                print(f'  ✅ {model}: Working')
                working_models.append(model)
            else:
                error_msg = response.text[:150] if response.text else 'No error message'
                print(f'  ❌ {model}: HTTP {response.status_code} - {error_msg}')
                
        except Exception as e:
            print(f'  ❌ {model}: Error - {str(e)[:100]}')
    
    print(f'\n📊 Summary: {len(working_models)}/{len(test_models)} models working')
    print(f'✅ Working models: {", ".join(working_models)}')
    return working_models

if __name__ == "__main__":
    test_models()
