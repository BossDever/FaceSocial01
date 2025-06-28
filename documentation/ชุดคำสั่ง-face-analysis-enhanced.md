# ชุดคำสั่ง: Face Analysis & Enhanced Registration
## ระบบวิเคราะห์ใบหน้าแบบครบวงจรและการลงทะเบียนขั้นสูง

### 📋 สารบัญ
7.1 [ภาพรวม Enhanced Face Analysis](#71-ภาพรวม-enhanced-face-analysis)
7.2 [Face Analysis Service](#72-face-analysis-service)
7.3 [Enhanced Registration Service](#73-enhanced-registration-service)
7.4 [Quality Assessment System](#74-quality-assessment-system)
7.5 [Landmark Detection](#75-landmark-detection)
7.6 [API Endpoints](#76-api-endpoints)
7.7 [Frontend Integration](#77-frontend-integration)
7.8 [Performance Management](#78-performance-management)

---

## 7.1 ภาพรวม Enhanced Face Analysis

ระบบ Face Analysis แบบครบวงจรที่รวม Face Detection, Face Recognition, Quality Assessment และ Enhanced Registration ไว้ในระบบเดียว โดยมีการจัดการ VRAM และ Performance Optimization

### 🏗️ สถาปัตยกรรม
```
Face Analysis Service
├── Face Detection (YOLO Models)
├── Face Recognition (AdaFace/ArcFace/FaceNet)
├── Quality Assessment (Enhanced)
├── Landmark Detection (68 points)
└── Enhanced Registration (Validation)
```

### 🎯 ฟีเจอร์หลัก
- **Comprehensive Analysis**: วิเคราะห์ใบหน้าแบบครบวงจร
- **Quality Control**: ประเมินคุณภาพรูปภาพก่อนการลงทะเบียน
- **Performance Optimization**: จัดการ Memory และ GPU อย่างมีประสิทธิภาพ
- **Enhanced Validation**: ตรวจสอบความเหมาะสมของรูปภาพสำหรับการลงทะเบียน

---

## 7.2 Face Analysis Service

### 7.2.1 Service Class
```python
from src.ai_services.face_analysis.face_analysis_service import FaceAnalysisService
from src.ai_services.face_analysis.models import AnalysisConfig, AnalysisMode

class FaceAnalysisService:
    """
    Enhanced Face Analysis Service
    รวม Face Detection + Face Recognition ในระบบเดียว
    """
    
    def __init__(
        self,
        vram_manager=None,
        config=None,
        face_detection_service=None,
        face_recognition_service=None,
    ):
        self.vram_manager = vram_manager
        self.config = config or {}
        self.face_detection_service = face_detection_service
        self.face_recognition_service = face_recognition_service
        
        # Performance tracking
        self.stats = {
            "total_analyses": 0,
            "total_faces_detected": 0,
            "total_faces_recognized": 0,
            "processing_times": [],
            "success_rates": [],
        }
```

### 7.2.2 Analysis Configuration
```python
from src.ai_services.face_analysis.models import AnalysisConfig, AnalysisMode, QualityLevel

# Configuration options
config = AnalysisConfig(
    mode=AnalysisMode.FULL_ANALYSIS,  # DETECTION_ONLY, RECOGNITION_ONLY, FULL_ANALYSIS
    detection_model="yolo",
    recognition_model="adaface",
    confidence_threshold=0.7,
    min_face_size=30,
    max_faces=10,
    use_quality_based_selection=True,
    quality_level=QualityLevel.HIGH,
    include_landmarks=True,
    include_pose_analysis=True,
    include_age_gender=False,
    include_emotions=False,
)
```

### 7.2.3 การใช้งาน Face Analysis
```python
async def analyze_faces_comprehensive(image_path: str):
    """ตัวอย่างการวิเคราะห์ใบหน้าแบบครบวงจร"""
    
    # Load image
    import cv2
    image = cv2.imread(image_path)
    
    # Create analysis config
    config = AnalysisConfig(
        mode=AnalysisMode.FULL_ANALYSIS,
        detection_model="yolov11m-face",
        recognition_model="adaface",
        confidence_threshold=0.8,
        min_face_size=50,
        use_quality_based_selection=True,
        include_landmarks=True,
        include_pose_analysis=True
    )
    
    # Perform analysis
    result = await face_analysis_service.analyze_faces(
        image=image,
        config=config,
        gallery=None  # หรือส่ง gallery สำหรับการจดจำ
    )
    
    # Process results
    if result.success:
        print(f"พบใบหน้า: {len(result.faces)} หน้า")
        print(f"เวลาในการประมวลผล: {result.processing_time:.3f}s")
        
        for i, face in enumerate(result.faces):
            print(f"ใบหน้า {i+1}:")
            print(f"  - Confidence: {face.detection_confidence:.3f}")
            print(f"  - Quality Score: {face.quality_score:.3f}")
            print(f"  - Bounding Box: {face.bbox}")
            
            if face.recognition_result:
                print(f"  - จดจำได้: {face.recognition_result.person_name}")
                print(f"  - Similarity: {face.recognition_result.similarity:.3f}")
    else:
        print(f"การวิเคราะห์ล้มเหลว: {result.error}")
```

---

## 7.3 Enhanced Registration Service

### 7.3.1 Service Class
```python
from src.ai_services.face_analysis.enhanced_registration_service import EnhancedFaceRegistrationService

class EnhancedFaceRegistrationService:
    """Enhanced face registration with comprehensive quality checks"""
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # Initialize quality assessment
        self.quality_assessor = EnhancedFaceQualityAssessment(
            self.config.get("quality", {})
        )
        
        # Initialize landmark detector
        self.landmark_detector = FaceLandmarkDetector()
        
        # Enhanced thresholds
        self.enhanced_thresholds = {
            "minimum_quality_score": 0.75,
            "strict_frontal_check": True,
            "require_landmarks": True,
            "max_pose_angle": 15.0,  # degrees
        }
```

### 7.3.2 การตรวจสอบความเหมาะสมของรูปภาพ
```python
async def validate_registration_image_example():
    """ตัวอย่างการตรวจสอบรูปภาพสำหรับการลงทะเบียน"""
    
    # Read image file
    with open("user_photo.jpg", "rb") as f:
        image_bytes = f.read()
    
    # Validate image
    validation_result = await enhanced_registration_service.validate_registration_image(
        image_bytes=image_bytes,
        person_id="user123",
        strict_mode=True
    )
    
    if validation_result["success"]:
        print("✅ รูปภาพเหมาะสมสำหรับการลงทะเบียน")
        
        # Quality metrics
        quality = validation_result["quality_assessment"]
        print(f"คะแนนคุณภาพ: {quality['overall_score']:.3f}")
        print(f"ความชัด: {quality['sharpness_score']:.3f}")
        print(f"แสงสว่าง: {quality['brightness_score']:.3f}")
        
        # Landmark analysis
        landmarks = validation_result["landmark_analysis"]
        if landmarks["landmarks_found"]:
            print(f"จำนวน Landmarks: {landmarks['landmark_count']}")
            print(f"ใบหน้าหันตรง: {landmarks['is_frontal']}")
        
        # Pose analysis
        pose = validation_result["pose_analysis"]
        print(f"มุมการหัน - Yaw: {pose['yaw_angle']:.1f}°, Pitch: {pose['pitch_angle']:.1f}°")
    
    else:
        print("❌ รูปภาพไม่เหมาะสมสำหรับการลงทะเบียน")
        print(f"เหตุผล: {validation_result['error']}")
        
        # Recommendations
        for recommendation in validation_result["recommendations"]:
            print(f"💡 {recommendation}")
```

### 7.3.3 Enhanced Registration Workflow
```python
async def enhanced_registration_workflow(person_id: str, person_name: str, image_bytes: bytes):
    """Workflow สำหรับการลงทะเบียนใบหน้าขั้นสูง"""
    
    # Step 1: Validate image quality
    validation_result = await enhanced_registration_service.validate_registration_image(
        image_bytes=image_bytes,
        person_id=person_id,
        strict_mode=True
    )
    
    if not validation_result["success"]:
        return {
            "success": False,
            "stage": "validation",
            "error": validation_result["error"],
            "recommendations": validation_result["recommendations"]
        }
    
    # Step 2: Extract face embedding
    img_buffer = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
    
    # Detect and extract best face
    detection_result = await face_detection_service.detect_faces(
        image=image,
        model_name="yolov11m-face",
        conf_threshold=0.8,
        min_quality_threshold=70.0,
        return_landmarks=True
    )
    
    if not detection_result.faces:
        return {
            "success": False,
            "stage": "detection",
            "error": "ไม่พบใบหน้าในรูปภาพ"
        }
    
    # Get best quality face
    best_face = max(detection_result.faces, key=lambda f: f.quality_score)
    
    # Step 3: Generate embedding
    embedding_result = await face_recognition_service.extract_embedding(
        face_image=best_face.cropped_face,
        model_name="adaface"
    )
    
    if not embedding_result.success:
        return {
            "success": False,
            "stage": "embedding",
            "error": "ไม่สามารถสร้าง embedding ได้"
        }
    
    # Step 4: Save to database (pseudo-code)
    registration_data = {
        "person_id": person_id,
        "person_name": person_name,
        "embedding": embedding_result.embedding.tolist(),
        "model_used": embedding_result.model_used,
        "quality_score": best_face.quality_score,
        "validation_metrics": validation_result,
        "bbox": [best_face.x1, best_face.y1, best_face.x2, best_face.y2],
        "landmarks": best_face.landmarks if hasattr(best_face, 'landmarks') else None
    }
    
    # await database.save_face_registration(registration_data)
    
    return {
        "success": True,
        "person_id": person_id,
        "quality_score": best_face.quality_score,
        "model_used": embedding_result.model_used,
        "validation_summary": validation_result["quality_assessment"]
    }
```

---

## 7.4 Quality Assessment

### 7.4.1 Enhanced Quality Assessment
```python
from src.ai_services.face_detection.enhanced_quality_assessment import EnhancedFaceQualityAssessment

class EnhancedFaceQualityAssessment:
    """Comprehensive face quality assessment"""
    
    def assess_face_quality(self, image, use_landmarks=True):
        """ประเมินคุณภาพใบหน้าแบบครบวงจร"""
        
        quality_metrics = {
            "sharpness_score": self._calculate_sharpness(image),
            "brightness_score": self._assess_brightness(image),
            "contrast_score": self._assess_contrast(image),
            "blur_score": self._detect_motion_blur(image),
            "noise_score": self._assess_noise_level(image),
            "exposure_score": self._assess_exposure(image),
        }
        
        if use_landmarks:
            landmark_metrics = self._assess_landmark_quality(image)
            quality_metrics.update(landmark_metrics)
        
        return quality_metrics
    
    def get_quality_summary(self, quality_metrics):
        """สรุปคะแนนคุณภาพ"""
        
        # Calculate overall score
        weights = {
            "sharpness_score": 0.25,
            "brightness_score": 0.15,
            "contrast_score": 0.15,
            "blur_score": 0.20,
            "noise_score": 0.10,
            "exposure_score": 0.15
        }
        
        overall_score = sum(
            quality_metrics.get(metric, 0.0) * weight
            for metric, weight in weights.items()
        )
        
        return {
            "overall_score": overall_score,
            "quality_level": self._get_quality_level(overall_score),
            "individual_scores": quality_metrics,
            "recommendations": self._generate_recommendations(quality_metrics)
        }
```

### 7.4.2 Quality Metrics Examples
```python
async def assess_image_quality_example(image_path: str):
    """ตัวอย่างการประเมินคุณภาพรูปภาพ"""
    
    image = cv2.imread(image_path)
    
    # Assess quality
    quality_metrics = quality_assessor.assess_face_quality(
        image=image,
        use_landmarks=True
    )
    
    summary = quality_assessor.get_quality_summary(quality_metrics)
    
    print(f"คะแนนรวม: {summary['overall_score']:.3f}")
    print(f"ระดับคุณภาพ: {summary['quality_level']}")
    
    print("\nคะแนนแต่ละด้าน:")
    for metric, score in summary['individual_scores'].items():
        print(f"  {metric}: {score:.3f}")
    
    print("\nข้อเสนะแนะ:")
    for recommendation in summary['recommendations']:
        print(f"  💡 {recommendation}")
```

---

## 7.5 Landmark Detection

### 7.5.1 Landmark Detection Service
```python
from src.ai_services.face_detection.landmark_detector import FaceLandmarkDetector

class FaceLandmarkDetector:
    """Face landmark detection and analysis"""
    
    def detect_landmarks(self, image, detailed=True):
        """ตรวจจับ landmarks ของใบหน้า (68 จุด)"""
        
        results = []
        faces = self.face_cascade.detectMultiScale(image)
        
        for (x, y, w, h) in faces:
            face_roi = image[y:y+h, x:x+w]
            
            # Detect 68 landmarks
            landmarks = self._detect_68_landmarks(face_roi)
            
            if landmarks is not None:
                # Adjust coordinates to full image
                adjusted_landmarks = self._adjust_landmarks_to_full_image(
                    landmarks, x, y
                )
                
                face_data = {
                    "bbox": [x, y, x+w, y+h],
                    "landmarks": adjusted_landmarks,
                    "landmark_count": len(adjusted_landmarks)
                }
                
                if detailed:
                    face_data.update(self._analyze_face_geometry(adjusted_landmarks))
                
                results.append(face_data)
        
        return results
    
    def is_frontal_face(self, face_data, max_angle=15.0):
        """ตรวจสอบว่าเป็นใบหน้าหันตรงหรือไม่"""
        
        landmarks = face_data.get("landmarks", [])
        if len(landmarks) < 68:
            return False
        
        # Calculate pose angles
        pose_angles = self._calculate_pose_angles(landmarks)
        
        return (
            abs(pose_angles["yaw"]) <= max_angle and
            abs(pose_angles["pitch"]) <= max_angle and
            abs(pose_angles["roll"]) <= max_angle
        )
```

### 7.5.2 Pose Analysis
```python
async def analyze_face_pose_example(image_path: str):
    """ตัวอย่างการวิเคราะห์ท่าทางใบหน้า"""
    
    image = cv2.imread(image_path)
    
    # Detect landmarks
    landmark_results = landmark_detector.detect_landmarks(
        image=image,
        detailed=True
    )
    
    if landmark_results:
        for i, face in enumerate(landmark_results):
            print(f"ใบหน้า {i+1}:")
            print(f"  จำนวน Landmarks: {face['landmark_count']}")
            
            # Pose analysis
            if "pose_analysis" in face:
                pose = face["pose_analysis"]
                print(f"  มุมการหัน:")
                print(f"    Yaw (ซ้าย-ขวา): {pose['yaw_angle']:.1f}°")
                print(f"    Pitch (ขึ้น-ลง): {pose['pitch_angle']:.1f}°")
                print(f"    Roll (เอียง): {pose['roll_angle']:.1f}°")
            
            # Check if frontal
            is_frontal = landmark_detector.is_frontal_face(face)
            print(f"  ใบหน้าหันตรง: {'✅' if is_frontal else '❌'}")
            
            # Face geometry
            if "face_geometry" in face:
                geometry = face["face_geometry"]
                print(f"  ความกว้างใบหน้า: {geometry['face_width']:.1f}px")
                print(f"  ความสูงใบหน้า: {geometry['face_height']:.1f}px")
                print(f"  อัตราส่วน: {geometry['face_ratio']:.2f}")
```

---

## 7.6 API Endpoints

### 7.6.1 Complete Analysis Endpoint
```python
from src.api.complete_endpoints import face_analysis_router

@face_analysis_router.post("/analyze/comprehensive")
async def comprehensive_face_analysis(
    request: FaceAnalysisJSONRequest,
    face_analysis_service = Depends(get_face_analysis_service)
):
    """API สำหรับการวิเคราะห์ใบหน้าแบบครบวงจร"""
    
    try:
        # Decode image
        image = decode_base64_image(request.image_base64)
        
        # Create analysis config
        config = AnalysisConfig(
            mode=AnalysisMode.FULL_ANALYSIS,
            **request.config or {}
        )
        
        # Perform analysis
        result = await face_analysis_service.analyze_faces(
            image=image,
            config=config,
            gallery=request.gallery
        )
        
        return {
            "success": result.success,
            "faces": [face.to_dict() for face in result.faces],
            "processing_time": result.processing_time,
            "model_info": result.model_info,
            "stats": result.stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@face_analysis_router.post("/register/enhanced")
async def enhanced_face_registration(
    person_id: str = Form(...),
    person_name: str = Form(...),
    image: UploadFile = File(...),
    strict_mode: bool = Form(True),
    enhanced_registration_service = Depends(get_enhanced_registration_service)
):
    """API สำหรับการลงทะเบียนใบหน้าขั้นสูง"""
    
    try:
        # Read image
        image_bytes = await image.read()
        
        # Validate and register
        result = await enhanced_registration_service.validate_registration_image(
            image_bytes=image_bytes,
            person_id=person_id,
            strict_mode=strict_mode
        )
        
        if result["success"]:
            # Proceed with registration
            registration_result = await enhanced_registration_workflow(
                person_id=person_id,
                person_name=person_name,
                image_bytes=image_bytes
            )
            return registration_result
        else:
            return result
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 7.7 การใช้งานใน Frontend

### ⚛️ React Hook สำหรับ Enhanced Analysis
```typescript
// hooks/useEnhancedFaceAnalysis.ts
import { useState, useCallback } from 'react';

interface AnalysisConfig {
  mode: 'detection_only' | 'recognition_only' | 'full_analysis';
  detection_model?: string;
  recognition_model?: string;
  confidence_threshold?: number;
  include_landmarks?: boolean;
  include_pose_analysis?: boolean;
}

interface AnalysisResult {
  success: boolean;
  faces: FaceResult[];
  processing_time: number;
  model_info: any;
  stats: any;
}

export const useEnhancedFaceAnalysis = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const analyzeImage = useCallback(async (
    imageFile: File,
    config: AnalysisConfig = { mode: 'full_analysis' }
  ) => {
    setIsAnalyzing(true);
    setError(null);

    try {
      // Convert image to base64
      const base64 = await new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onload = () => {
          const result = reader.result as string;
          resolve(result.split(',')[1]); // Remove data:image/jpeg;base64, prefix
        };
        reader.readAsDataURL(imageFile);
      });

      // Call API
      const response = await fetch('/api/face-analysis/analyze/comprehensive', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_base64: base64,
          mode: config.mode,
          config: config,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const analysisResult = await response.json();
      setResult(analysisResult);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'การวิเคราะห์ล้มเหลว');
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  return {
    analyzeImage,
    isAnalyzing,
    result,
    error,
    clearResult: () => setResult(null),
    clearError: () => setError(null),
  };
};
```

### 🎯 Enhanced Registration Component
```typescript
// components/EnhancedFaceRegistration.tsx
import React, { useState } from 'react';
import { useEnhancedFaceAnalysis } from '../hooks/useEnhancedFaceAnalysis';

interface RegistrationProps {
  onSuccess: (result: any) => void;
  onError: (error: string) => void;
}

export const EnhancedFaceRegistration: React.FC<RegistrationProps> = ({
  onSuccess,
  onError,
}) => {
  const [personId, setPersonId] = useState('');
  const [personName, setPersonName] = useState('');
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isRegistering, setIsRegistering] = useState(false);
  const [validationResult, setValidationResult] = useState<any>(null);

  const { analyzeImage, isAnalyzing } = useEnhancedFaceAnalysis();

  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      
      // Auto-validate image quality
      analyzeImage(file, {
        mode: 'detection_only',
        include_landmarks: true,
        include_pose_analysis: true,
      }).then((result) => {
        if (result) {
          setValidationResult(result);
        }
      });
    }
  };

  const handleRegistration = async () => {
    if (!selectedImage || !personId || !personName) {
      onError('กรุณากรอกข้อมูลให้ครบถ้วน');
      return;
    }

    setIsRegistering(true);

    try {
      const formData = new FormData();
      formData.append('person_id', personId);
      formData.append('person_name', personName);
      formData.append('image', selectedImage);
      formData.append('strict_mode', 'true');

      const response = await fetch('/api/face-analysis/register/enhanced', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      
      if (result.success) {
        onSuccess(result);
      } else {
        onError(result.error || 'การลงทะเบียนล้มเหลว');
      }

    } catch (err) {
      onError(err instanceof Error ? err.message : 'เกิดข้อผิดพลาด');
    } finally {
      setIsRegistering(false);
    }
  };

  return (
    <div className="enhanced-registration">
      <h3>ลงทะเบียนใบหน้าขั้นสูง</h3>
      
      {/* Form fields */}
      <div className="form-group">
        <label>รหัสบุคคล:</label>
        <input
          type="text"
          value={personId}
          onChange={(e) => setPersonId(e.target.value)}
          placeholder="เช่น USER001"
        />
      </div>

      <div className="form-group">
        <label>ชื่อ-นามสกุล:</label>
        <input
          type="text"
          value={personName}
          onChange={(e) => setPersonName(e.target.value)}
          placeholder="เช่น นาย สมชาย ใจดี"
        />
      </div>

      {/* Image upload */}
      <div className="form-group">
        <label>รูปภาพใบหน้า:</label>
        <input
          type="file"
          accept="image/*"
          onChange={handleImageSelect}
        />
      </div>

      {/* Image preview */}
      {previewUrl && (
        <div className="image-preview">
          <img src={previewUrl} alt="Preview" style={{ maxWidth: '300px' }} />
        </div>
      )}

      {/* Validation results */}
      {validationResult && (
        <div className="validation-results">
          <h4>ผลการตรวจสอบคุณภาพ:</h4>
          {validationResult.faces?.length > 0 ? (
            <div className="quality-info">
              <p>✅ พบใบหน้า: {validationResult.faces.length} หน้า</p>
              {validationResult.faces.map((face: any, index: number) => (
                <div key={index} className="face-info">
                  <p>ความมั่นใจ: {(face.detection_confidence * 100).toFixed(1)}%</p>
                  <p>คะแนนคุณภาพ: {(face.quality_score * 100).toFixed(1)}%</p>
                  {face.pose_analysis && (
                    <p>มุมการหัน: {Math.abs(face.pose_analysis.yaw_angle).toFixed(1)}°</p>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <p className="error">❌ ไม่พบใบหน้าในรูปภาพ</p>
          )}
        </div>
      )}

      {/* Action buttons */}
      <div className="actions">
        <button
          onClick={handleRegistration}
          disabled={isRegistering || isAnalyzing || !selectedImage}
          className="register-btn"
        >
          {isRegistering ? 'กำลังลงทะเบียน...' : 'ลงทะเบียน'}
        </button>
      </div>

      {/* Loading indicator */}
      {(isAnalyzing || isRegistering) && (
        <div className="loading">
          <p>กำลังประมวลผล...</p>
        </div>
      )}
    </div>
  );
};
```

---

## 7.8 การจัดการ Performance

### ⚡ Performance Monitoring
```python
async def monitor_analysis_performance():
    """ตัวอย่างการติดตาม Performance"""
    
    # Get service stats
    stats = face_analysis_service.stats
    
    print("📊 สถิติการใช้งาน:")
    print(f"  จำนวนการวิเคราะห์ทั้งหมด: {stats['total_analyses']}")
    print(f"  ใบหน้าที่ตรวจพบ: {stats['total_faces_detected']}")
    print(f"  ใบหน้าที่จดจำได้: {stats['total_faces_recognized']}")
    
    if stats['processing_times']:
        avg_time = sum(stats['processing_times']) / len(stats['processing_times'])
        print(f"  เวลาประมวลผลเฉลี่ย: {avg_time:.3f}s")
    
    if stats['success_rates']:
        avg_success = sum(stats['success_rates']) / len(stats['success_rates'])
        print(f"  อัตราความสำเร็จ: {avg_success:.1%}")

# การติดตาม VRAM
async def monitor_vram_usage():
    """ติดตาม VRAM usage"""
    
    if vram_manager:
        vram_info = await vram_manager.get_memory_info()
        print("🔧 สถานะ VRAM:")
        print(f"  VRAM ทั้งหมด: {vram_info['total_vram_gb']:.1f} GB")
        print(f"  VRAM ที่ใช้: {vram_info['allocated_vram_gb']:.1f} GB")
        print(f"  VRAM ว่าง: {vram_info['free_vram_gb']:.1f} GB")
        print(f"  การใช้งาน: {vram_info['usage_percentage']:.1f}%")
```

### 🎯 Performance Optimization Tips

1. **Batch Processing**: ประมวลผลรูปภาพหลายรูปพร้อมกัน
2. **Model Caching**: Cache โมเดลใน VRAM เพื่อลดเวลา loading
3. **Quality Pre-filtering**: กรองรูปภาพคุณภาพต่ำก่อนส่งไป recognition
4. **Async Processing**: ใช้ asynchronous processing สำหรับ batch operations
5. **Memory Management**: จัดการ VRAM อย่างมีประสิทธิภาพ

---

## สรุป

ระบบ Face Analysis & Enhanced Registration ให้ความสามารถในการวิเคราะห์ใบหน้าแบบครบวงจร พร้อมการตรวจสอบคุณภาพและการลงทะเบียนขั้นสูง:

### ✅ ฟีเจอร์หลัก
- **Comprehensive Analysis**: รวม Detection + Recognition + Quality Assessment
- **Enhanced Registration**: ตรวจสอบความเหมาะสมก่อนลงทะเบียน  
- **Quality Control**: ประเมินคุณภาพรูปภาพแบบละเอียด
- **Performance Monitoring**: ติดตามสถิติและ VRAM usage
- **Flexible Configuration**: ปรับแต่งการทำงานตามความต้องการ

### 🎯 การประยุกต์ใช้
- ระบบลงทะเบียนใบหน้าสำหรับ Access Control
- แพลตฟอร์มสื่อสังคมกับ Face Tagging
- ระบบตรวจสอบตัวตนด้วยใบหน้า
- การวิเคราะห์ใบหน้าในรูปภาพกลุ่ม

ระบบนี้ช่วยให้การทำงานกับใบหน้ามีประสิทธิภาพและแม่นยำมากขึ้น พร้อมรองรับการใช้งานในระดับ Production
