// API Types for Face Recognition System

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  width: number;
  height: number;
  center_x: number;
  center_y: number;
  confidence: number;
  area?: number;
}

export interface FaceDetectionResult {
  bbox: BoundingBox;
  quality_score: number;
  model_used: string;
  processing_time: number;
}

export interface AntiSpoofingResult {
  is_real: boolean;
  confidence: number;
  model_used: string;
  region: {
    x: number;
    y: number;
    w: number;
    h: number;
  };
  processing_time: number;
}

export interface AgeGenderResult {
  age: number;
  gender: string;
  gender_confidence: number;
  face_region: {
    x: number;
    y: number;
    w: number;
    h: number;
  };
}

export interface FaceAnalysisResult {
  emotion: string;
  emotion_confidence: number;
  age: number;
  gender: string;
  gender_confidence: number;
  face_region: {
    x: number;
    y: number;
    w: number;
    h: number;
  };
}

export interface BatchProcessingResult {
  total_files: number;
  processed_files: number;
  failed_files: number;
  results: Array<{
    filename: string;
    detection: FaceDetectionResult[];
    anti_spoofing: AntiSpoofingResult[];
    analysis: FaceAnalysisResult[];
  }>;
}

// Full API Response Types
export interface AntiSpoofingApiResponse {
  success: boolean;
  faces_detected: number;
  faces_analysis: AntiSpoofingResult[];
  overall_result: {
    is_real: boolean;
    confidence: number;
    spoofing_detected: boolean;
    real_faces: number;
    fake_faces: number;
  };
  processing_time: number;
  model: string;
  message: string;
  error: string | null;
}

export interface FaceDetectionApiResponse {
  faces: FaceDetectionResult[];
  face_count: number;
  total_processing_time: number;
  model: string;
  message: string;
  error: string | null;
}
