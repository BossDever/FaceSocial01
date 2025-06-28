import { useState, useCallback } from 'react';
import { FaceRecognitionService } from '@/lib/services/faceRecognitionService';
import type { 
  FaceDetectionResult, 
  AntiSpoofingResult, 
  FaceAnalysisResult,
  AgeGenderResult,
  BatchProcessingResult 
} from '@/types/api';

// Interfaces สำหรับ API responses
interface FaceDetectionApiResponse {
  faces?: FaceDetectionResult[];
  face_count?: number;
  total_processing_time?: number;
}

interface ProcessingState {
  loading: boolean;
  results: {
    detection: FaceDetectionResult[] | null;
    antiSpoofing: AntiSpoofingResult[] | null;
    analysis: FaceAnalysisResult[] | null;
    ageGender: AgeGenderResult[] | null;
    batch: BatchProcessingResult | null;
  };
  error: string | null;
  processingTime: number;
}

// Face Analysis API response interface
interface AgeGenderApiResponse {
  analyses?: AgeGenderResult[];
  total_faces?: number;
  processing_time?: number;
}

export function useFaceRecognition() {
  const [state, setState] = useState<ProcessingState>({
    loading: false,
    results: {
      detection: null,
      antiSpoofing: null,
      analysis: null,
      ageGender: null,
      batch: null
    },
    error: null,
    processingTime: 0
  });

  const resetState = useCallback(() => {
    setState({
      loading: false,
      results: {
        detection: null,
        antiSpoofing: null,
        analysis: null,
        ageGender: null,
        batch: null
      },
      error: null,
      processingTime: 0
    });
  }, []);

  const detectFaces = useCallback(async (file: File) => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    const startTime = Date.now();

    try {
      const response = await FaceRecognitionService.detectFaces(file);
      const processingTime = Date.now() - startTime;
      
      // API ส่งกลับมาเป็น { faces: [...], face_count: n, ... }
      const apiResult = response.data || response;
      const faceResponse = apiResult as FaceDetectionApiResponse;
      const faces = faceResponse?.faces || [];
      
      setState(prev => ({
        ...prev,
        loading: false,
        results: { ...prev.results, detection: faces },
        processingTime: faceResponse?.total_processing_time || processingTime
      }));

      return faces;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Face detection failed';
      setState(prev => ({ ...prev, loading: false, error: errorMessage }));
      throw error;
    }
  }, []);  const detectSpoofing = useCallback(async (file: File, confidenceThreshold: number = 0.5) => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    const startTime = Date.now();

    try {
      const response = await FaceRecognitionService.detectSpoofing(file, confidenceThreshold);
      const processingTime = Date.now() - startTime;

      // API ตอบกลับเป็น { success: true, faces_analysis: [...], overall_result: {...}, ... }
      const apiResult = response.data || response;
      
      setState(prev => ({
        ...prev,
        loading: false,
        results: { ...prev.results, antiSpoofing: apiResult },
        processingTime: apiResult?.processing_time || processingTime
      }));

      return apiResult; // Return full API result
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Anti-spoofing detection failed';
      setState(prev => ({ ...prev, loading: false, error: errorMessage }));
      throw error;
    }
  }, []);

  const analyzeFaces = useCallback(async (file: File) => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    const startTime = Date.now();

    try {
      const response = await FaceRecognitionService.analyzeFaces(file);
      const processingTime = Date.now() - startTime;

      setState(prev => ({
        ...prev,
        loading: false,
        results: { ...prev.results, analysis: response.data },
        processingTime
      }));

      return response.data;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Face analysis failed';
      setState(prev => ({ ...prev, loading: false, error: errorMessage }));
      throw error;
    }
  }, []);

  const processComplete = useCallback(async (file: File) => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    const startTime = Date.now();

    try {
      const response = await FaceRecognitionService.processComplete(file);
      const processingTime = Date.now() - startTime;

      setState(prev => ({
        ...prev,
        loading: false,
        results: {
          detection: response.data.detection,
          antiSpoofing: response.data.anti_spoofing,
          analysis: response.data.analysis,
          ageGender: null,
          batch: null
        },
        processingTime
      }));

      return response.data;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Complete processing failed';
      setState(prev => ({ ...prev, loading: false, error: errorMessage }));
      throw error;
    }
  }, []);

  const processBatch = useCallback(async (files: File[]) => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    const startTime = Date.now();

    try {
      const response = await FaceRecognitionService.processBatch(files);
      const processingTime = Date.now() - startTime;

      setState(prev => ({
        ...prev,
        loading: false,
        results: { ...prev.results, batch: response.data },
        processingTime
      }));

      return response.data;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Batch processing failed';
      setState(prev => ({ ...prev, loading: false, error: errorMessage }));
      throw error;
    }
  }, []);

  const compareFaces = useCallback(async (file1: File, file2: File) => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    const startTime = Date.now();

    try {
      const response = await FaceRecognitionService.compareFaces(file1, file2);
      const processingTime = Date.now() - startTime;

      setState(prev => ({
        ...prev,
        loading: false,
        processingTime
      }));

      return response.data;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Face comparison failed';
      setState(prev => ({ ...prev, loading: false, error: errorMessage }));
      throw error;
    }
  }, []);

  // Face Analysis Detection
  const analyzeAgeGender = useCallback(async (file: File) => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    const startTime = Date.now();

    try {
      const response = await FaceRecognitionService.analyzeFaces(file);
      const processingTime = Date.now() - startTime;

      // API ตอบกลับเป็น { analyses: [...], total_faces: n, ... }
      const apiResult = response.data || response;
      const ageGenderResponse = apiResult as AgeGenderApiResponse;
      const analyses = ageGenderResponse?.analyses || [];

      setState(prev => ({
        ...prev,
        loading: false,
        results: { ...prev.results, ageGender: analyses },
        processingTime: ageGenderResponse?.processing_time || processingTime
      }));

      return apiResult; // ส่งกลับ API result เต็ม
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Age & Gender analysis failed';
      setState(prev => ({ ...prev, loading: false, error: errorMessage }));
      throw error;
    }
  }, []);

  return {
    ...state,
    detectFaces,
    detectSpoofing,
    analyzeFaces,
    processComplete,
    processBatch,
    compareFaces,
    resetState,
    analyzeAgeGender
  };
}
