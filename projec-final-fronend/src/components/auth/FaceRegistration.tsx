'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { 
  Button, 
  Card, 
  Alert, 
  Progress, 
  Typography, 
  Space,
  Spin,
  Modal,
  Row,
  Col,
  Statistic,
  Badge
} from 'antd';
import { 
  CameraOutlined, 
  CheckCircleOutlined, 
  ExclamationCircleOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;

interface FaceRegistrationProps {
  onComplete: (faceData: {
    embedding: number[];
    imageUrl: string;
    qualityScore: number;
    confidence: number;
    totalImages?: number;
    userInfo?: {
      firstName: string;
      lastName: string;
      userId: string;
    };
  }) => void;
  onError: (error: string) => void;
  onBack?: () => void; // เพิ่ม callback สำหรับกลับไปแก้ไขข้อมูล
  userInfo?: {
    firstName: string;
    lastName: string;
    userId?: string;
  };
  loading?: boolean; // รับ loading state จาก parent
  onLoadingChange?: (loading: boolean) => void; // callback สำหรับเปลี่ยน loading state
}

const FaceRegistration: React.FC<FaceRegistrationProps> = ({
  onComplete,
  onError,
  onBack,
  userInfo,
  loading: externalLoading = false,
  onLoadingChange
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  
  // 🔧 เพิ่ม refs เพื่อป้องกันการเรียกซ้ำ
  const isProcessingRef = useRef(false);
  const initializingRef = useRef(false);
  const detectionIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const completionTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const registrationTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const hasCompletedRef = useRef(false);

  const [isStreaming, setIsStreaming] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [capturedImages, setCapturedImages] = useState<string[]>([]);
  const [faceDetected, setFaceDetected] = useState(false);
  const [faceQuality, setFaceQuality] = useState(0);
  const [isRealFace, setIsRealFace] = useState(false);
  const [step, setStep] = useState<'setup' | 'detecting' | 'captured' | 'processing' | 'registering' | 'completed'>('setup');
  const [detectionProgress, setDetectionProgress] = useState(0);
  const [registrationProgress, setRegistrationProgress] = useState(0);
  const [imageCount, setImageCount] = useState(0);
  const [showGuide, setShowGuide] = useState(false);
  const [showCameraModal, setShowCameraModal] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const TARGET_IMAGES = 15;

  console.log('FaceRegistration render - step:', step, 'isStreaming:', isStreaming, 'processing:', isProcessingRef.current);

  // Update external loading state
  useEffect(() => {
    if (onLoadingChange) {
      onLoadingChange(isProcessing || externalLoading);
    }
  }, [isProcessing, externalLoading, onLoadingChange]);

  // 🔧 ปรับปรุง cleanup function
  const cleanupTimeouts = useCallback(() => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    if (completionTimeoutRef.current) {
      clearTimeout(completionTimeoutRef.current);
      completionTimeoutRef.current = null;
    }
    if (registrationTimeoutRef.current) {
      clearTimeout(registrationTimeoutRef.current);
      registrationTimeoutRef.current = null;
    }
  }, []);

  // 🔧 ปรับปรุง stop camera function
  const stopCamera = useCallback(() => {
    console.log('🛑 Stopping camera...');
    cleanupTimeouts();
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsStreaming(false);
    setFaceDetected(false);
    setFaceQuality(0);
    setDetectionProgress(0);
  }, [cleanupTimeouts]);

  // Initialize camera
  const initCamera = useCallback(async () => {
    if (initializingRef.current) {
      console.log('🚫 Camera initialization already in progress');
      return;
    }
    initializingRef.current = true;

    try {
      console.log('📷 Starting camera initialization...');
      setError(null);
      
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        }
      });
      
      console.log('✅ Camera stream obtained successfully');
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsStreaming(true);
        console.log('✅ Video stream assigned and ready');
      }
    } catch (err) {
      console.error('❌ Error accessing camera:', err);
      setError('ไม่สามารถเข้าถึงกล้องได้ กรุณาอนุญาตการใช้งานกล้อง');
      onError('ไม่สามารถเข้าถึงกล้องได้');
    } finally {
      initializingRef.current = false;
    }
  }, [onError]);

  // Start camera modal
  const startCameraModal = useCallback(() => {
    console.log('🎬 Starting camera modal');
    setShowCameraModal(true);
    setStep('detecting');
  }, []);

  // Close camera modal
  const closeCameraModal = useCallback(() => {
    console.log('🚪 Closing camera modal');
    stopCamera();
    setShowCameraModal(false);
    setStep('setup');
    // Reset states
    setCapturedImages([]);
    setImageCount(0);
    hasCompletedRef.current = false;
    isProcessingRef.current = false;
  }, [stopCamera]);

  // เพิ่ม function สำหรับ test API connectivity
  const testAPIConnectivity = useCallback(async () => {
    try {
      console.log('🔍 Testing API connectivity...');
      
      // Test 1: Health check
      const healthResponse = await fetch('http://localhost:8080/api/face-recognition/health');
      console.log('🔍 Health check:', healthResponse.ok ? '✅' : '❌');
      
      return healthResponse.ok;
    } catch (error) {
      console.error('❌ API connectivity test failed:', error);
      return false;
    }
  }, []);

  // 🔧 ปรับปรุง processRegistration ด้วยการป้องกันการเรียกซ้ำ
  const processRegistration = useCallback(async () => {
    const callId = `process_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    console.log(`🔄 [${callId}] processRegistration called`);
    
    // Test API connectivity ก่อน
    const apiConnected = await testAPIConnectivity();
    if (!apiConnected) {
      setError('ไม่สามารถเชื่อมต่อกับ Face Recognition API ได้');
      onError('ไม่สามารถเชื่อมต่อกับ Face Recognition API ได้');
      return;
    }
    
    if (isProcessingRef.current) {
      console.warn(`🚫 [${callId}] Registration already in progress, ignoring duplicate call`);
      return;
    }

    if (hasCompletedRef.current) {
      console.warn(`🚫 [${callId}] Registration already completed, ignoring duplicate call`);
      return;
    }

    // 🔧 ตั้ง flags พร้อมกัน
    isProcessingRef.current = true;
    hasCompletedRef.current = true;
    console.log(`🔒 [${callId}] Marked as processing and completed`);

    try {
      setIsProcessing(true);
      setRegistrationProgress(0);
      
      console.log(`🔄 [${callId}] Starting registration processing...`);
      
      const embeddings: number[][] = [];
      const qualityScores: number[] = [];
      const imagesToProcess = Math.min(capturedImages.length, TARGET_IMAGES);
      
      console.log(`🖼️ [${callId}] Processing ${imagesToProcess} images for embedding extraction`);
      
      // Process each image sequentially with detailed logging
      for (let i = 0; i < imagesToProcess; i++) {
        if (!isProcessingRef.current) {
          console.log(`🛑 [${callId}] Processing interrupted at image ${i}`);
          return;
        }

        setRegistrationProgress(Math.round((i / imagesToProcess) * 80));
        
        try {
          const response = await fetch(capturedImages[i]);
          const blob = await response.blob();
          
          // 🔧 ใช้ FormData เหมือนเดิม (ไม่ใช่ JSON)
          const formData = new FormData();
          formData.append('file', blob, `face_${i + 1}.jpg`);
          formData.append('model_name', 'facenet');

          const embeddingResponse = await fetch('http://localhost:8080/api/face-recognition/extract-embedding', {
            method: 'POST',
            body: formData
          });
          
          if (!embeddingResponse.ok) {
            console.error(`❌ [${callId}] Failed to extract embedding for image ${i + 1}:`, embeddingResponse.status);
            continue;
          }

          const embeddingResult = await embeddingResponse.json();
          
          if (embeddingResult.success && embeddingResult.embedding) {
            embeddings.push(embeddingResult.embedding);
            qualityScores.push(embeddingResult.quality_score || 0.8);
            console.log(`✅ [${callId}] Successfully extracted embedding ${i + 1}/${imagesToProcess}`);
          } else {
            console.error(`❌ [${callId}] Failed to extract embedding for image ${i + 1}:`, embeddingResult.message);
          }
        } catch (err) {
          console.error(`❌ [${callId}] Error processing image ${i + 1}:`, err);
        }
        
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
      if (embeddings.length === 0) {
        throw new Error('ไม่สามารถสกัด embedding จากภาพใบหน้าได้ กรุณาลองใหม่');
      }
      
      setRegistrationProgress(90);
      
      const bestEmbeddingIndex = qualityScores.indexOf(Math.max(...qualityScores));
      const finalEmbedding = embeddings[bestEmbeddingIndex] || embeddings[0];
      const averageQuality = qualityScores.reduce((a, b) => a + b, 0) / qualityScores.length;
      
      setRegistrationProgress(100);
      
      console.log(`🎯 [${callId}] Face registration processed successfully:`, {
        totalImagesCollected: capturedImages.length,
        imagesProcessed: imagesToProcess,
        successfulEmbeddings: embeddings.length,
        averageQuality: averageQuality,
        bestEmbeddingIndex
      });
      
      // 🔧 เรียก onComplete โดยตรง
      console.log(`✅ [${callId}] Calling onComplete...`);
      
      await onComplete({
        embedding: finalEmbedding,
        imageUrl: capturedImages[bestEmbeddingIndex],
        qualityScore: averageQuality * 100,
        confidence: averageQuality,
        totalImages: capturedImages.length,
        userInfo: userInfo ? {
          firstName: userInfo.firstName,
          lastName: userInfo.lastName,
          userId: userInfo.userId || ''
        } : undefined
      });
      
      console.log(`✅ [${callId}] onComplete call finished successfully`);
      
    } catch (err) {
      console.error(`❌ [${callId}] Registration processing error:`, err);
      const errorMessage = err instanceof Error ? err.message : 'เกิดข้อผิดพลาดในการลงทะเบียนใบหน้า';
      setError(errorMessage);
      onError(errorMessage);
      
      // Reset flags on error
      hasCompletedRef.current = false;
    } finally {
      console.log(`🔓 [${callId}] Finally block - resetting processing flag`);
      isProcessingRef.current = false;
      setIsProcessing(false);
    }
  }, [capturedImages, onComplete, onError, userInfo, testAPIConnectivity]);

  // 🔧 ปรับปรุง capture function ด้วยการป้องกันการเรียกซ้ำ
  const captureAndSaveCroppedImage = useCallback(async (croppedCanvas: HTMLCanvasElement) => {
    if (imageCount >= TARGET_IMAGES || isProcessingRef.current || hasCompletedRef.current) {
      console.log('🚫 Capture blocked:', { imageCount, TARGET_IMAGES, processing: isProcessingRef.current, completed: hasCompletedRef.current });
      return;
    }

    const imageDataUrl = croppedCanvas.toDataURL('image/jpeg', 0.9);
    const newImageCount = imageCount + 1;
    
    setCapturedImages(prev => [...prev, imageDataUrl]);
    setImageCount(newImageCount);
    
    console.log(`📸 Captured cropped face image ${newImageCount}/${TARGET_IMAGES}`);
    
    // Save image to server (optional)
    try {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const fileName = `face-crop-${newImageCount.toString().padStart(2, '0')}-${timestamp}.jpg`;
      
      await fetch('/api/save-face-image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          imageData: imageDataUrl,
          fileName: fileName,
          userId: 'current-registration'
        })
      });
      
      console.log(`✅ Saved cropped face image: ${fileName}`);
    } catch (error) {
      console.warn('⚠️ Failed to save cropped face image:', error);
    }
    
    // 🔧 ปรับปรุงการตรวจสอบการครบภาพ
    if (newImageCount >= TARGET_IMAGES && !hasCompletedRef.current) {
      console.log('🎯 All images collected, initiating completion sequence...');
      
      cleanupTimeouts();
      setStep('completed');
      
      completionTimeoutRef.current = setTimeout(() => {
        stopCamera();
        setShowCameraModal(false);
        setStep('registering');
        
        registrationTimeoutRef.current = setTimeout(() => {
          if (!isProcessingRef.current) {
            processRegistration();
          }
        }, 200);
      }, 1500);
    }
  }, [imageCount, TARGET_IMAGES, stopCamera, cleanupTimeouts, processRegistration]);

  // 🔧 ปรับปรุง face detection ด้วยการป้องกัน race condition
  const detectFace = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !isStreaming || 
        imageCount >= TARGET_IMAGES || isProcessingRef.current || hasCompletedRef.current) {
      return;
    }

    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');
    
    if (!ctx || video.readyState < 2) {
      return;
    }

    try {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);

      canvas.toBlob(async (blob) => {
        if (!blob || imageCount >= TARGET_IMAGES || isProcessingRef.current || hasCompletedRef.current) return;

        const formData = new FormData();
        formData.append('image', blob, 'face.jpg');

        try {
          const faceDetectionData = new FormData();
          faceDetectionData.append('file', blob, 'face.jpg');
          faceDetectionData.append('model_name', 'yolov11m');
          faceDetectionData.append('conf_threshold', '0.5');
          faceDetectionData.append('max_faces', '5');

          const faceResponse = await fetch('http://localhost:8080/api/face-detection/detect', {
            method: 'POST',
            body: faceDetectionData
          });

          if (!faceResponse.ok) {
            console.error('❌ Face detection API error:', faceResponse.status);
            return;
          }

          const faceResult = await faceResponse.json();
          
          if (faceResult.success && faceResult.faces && faceResult.faces.length > 0) {
            const face = faceResult.faces[0];
            setFaceDetected(true);

            const bbox = face.bbox;
            const croppedCanvas = document.createElement('canvas');
            const croppedCtx = croppedCanvas.getContext('2d');
            
            if (croppedCtx && bbox) {
              let face_bbox;
              if (bbox.x1 !== undefined && bbox.y1 !== undefined && bbox.x2 !== undefined && bbox.y2 !== undefined) {
                face_bbox = {
                  x: Math.round(bbox.x1),
                  y: Math.round(bbox.y1),
                  width: Math.round(bbox.x2 - bbox.x1),
                  height: Math.round(bbox.y2 - bbox.y1)
                };
              } else if (bbox.x !== undefined && bbox.y !== undefined && bbox.width !== undefined && bbox.height !== undefined) {
                face_bbox = {
                  x: Math.round(bbox.x),
                  y: Math.round(bbox.y),
                  width: Math.round(bbox.width),
                  height: Math.round(bbox.height)
                };
              } else {
                console.error(`❌ Unknown bbox format: ${JSON.stringify(bbox)}`);
                return;
              }

              const padding = 0.2;
              const faceWidth = face_bbox.width;
              const faceHeight = face_bbox.height;
              const paddingX = faceWidth * padding;
              const paddingY = faceHeight * padding;
              
              const cropX = Math.max(0, Math.round(face_bbox.x - paddingX));
              const cropY = Math.max(0, Math.round(face_bbox.y - paddingY));
              const cropWidth = Math.min(canvas.width - cropX, Math.round(faceWidth + (paddingX * 2)));
              const cropHeight = Math.min(canvas.height - cropY, Math.round(faceHeight + (paddingY * 2)));
              
              const minSize = 224;
              const maxSize = 640;
              const targetSize = Math.max(minSize, Math.min(maxSize, Math.max(cropWidth, cropHeight)));
              
              let finalWidth, finalHeight;
              if (cropWidth > cropHeight) {
                finalWidth = targetSize;
                finalHeight = Math.floor((cropHeight / cropWidth) * targetSize);
              } else {
                finalHeight = targetSize;
                finalWidth = Math.floor((cropWidth / cropHeight) * targetSize);
              }
              
              finalWidth = Math.max(finalWidth, minSize);
              finalHeight = Math.max(finalHeight, minSize);
              
              croppedCanvas.width = finalWidth;
              croppedCanvas.height = finalHeight;
              
              croppedCtx.drawImage(
                canvas, 
                cropX, cropY, cropWidth, cropHeight,
                0, 0, finalWidth, finalHeight
              );
              
              croppedCanvas.toBlob(async (croppedBlob) => {
                if (!croppedBlob || imageCount >= TARGET_IMAGES || isProcessingRef.current || hasCompletedRef.current) return;

                const antispoofingData = new FormData();
                antispoofingData.append('image', croppedBlob, 'cropped_face.jpg');
                antispoofingData.append('confidence_threshold', '0.5');

                let isReal = true;
                try {
                  const antispoofingResponse = await fetch('http://localhost:8080/api/anti-spoofing/detect-upload', {
                    method: 'POST',
                    body: antispoofingData
                  });

                  if (!antispoofingResponse.ok) {
                    console.error('❌ Anti-spoofing API error:', antispoofingResponse.status);
                    isReal = true;
                  } else {
                    const antispoofingResult = await antispoofingResponse.json();
                    isReal = antispoofingResult.success && !antispoofingResult.overall_result?.spoofing_detected;
                  }
                } catch (err) {
                  console.error('❌ Anti-spoofing error:', err);
                  isReal = true;
                }
                
                setIsRealFace(isReal);
                
                // 🔧 NEW: ใช้ Enhanced Quality Check
                const enhancedQuality = await checkFaceQualityEnhanced(canvas);
                
                // 🔧 Fallback: ถ้า Enhanced Quality API ล้มเหลว ให้ใช้วิธีเดิม
                if (!enhancedQuality.success) {
                  console.warn('⚠️ Enhanced quality check failed, using fallback method');
                  
                  const fallbackQuality = Math.min(face.bbox.confidence * 100, face.quality_score || 85);
                  setFaceQuality(fallbackQuality);
                  setDetectionProgress(Math.min(fallbackQuality, 100));
                  
                  // ใช้เงื่อนไขสำรองที่เข้มงวดขึ้น
                  const passesFallbackChecks = (
                    face.bbox.confidence >= 0.8 && // เพิ่มจาก 0.7 เป็น 0.8
                    Math.min(face_bbox.width, face_bbox.height) >= 140 && // เพิ่มจาก 120 เป็น 140
                    fallbackQuality >= 75 && // เพิ่มจาก 65 เป็น 75
                    isReal
                  );
                  
                  if (passesFallbackChecks && imageCount < TARGET_IMAGES && !isProcessingRef.current && !hasCompletedRef.current) {
                    console.log(`🎯 Auto-capturing fallback quality face image ${imageCount + 1}/${TARGET_IMAGES}`, {
                      confidence: face.bbox.confidence.toFixed(3),
                      faceSize: `${face_bbox.width}x${face_bbox.height}`,
                      qualityScore: fallbackQuality.toFixed(1),
                      isReal
                    });
                    
                    await captureAndSaveCroppedImage(croppedCanvas);
                    return; // หยุดการประมวลผลต่อ
                  }
                }
                
                console.log(`🚀 Enhanced Quality Assessment:`, {
                  success: enhancedQuality.success,
                  qualityScore: enhancedQuality.overallQuality.toFixed(1),
                  grade: enhancedQuality.qualityGrade,
                  confidence: face.bbox.confidence.toFixed(3),
                  faceSize: `${face_bbox.width}x${face_bbox.height}`,
                  isSharp: enhancedQuality.isSharp,
                  isWellLit: enhancedQuality.isWellLit,
                  isFrontal: enhancedQuality.isFrontal,
                  hasLandmarks: enhancedQuality.hasLandmarks,
                  poseAngles: enhancedQuality.poseAngles,
                  isReal
                });

                // 🔧 เงื่อนไขการบันทึกภาพที่เข้มงวดมาก - ปรับให้เข้มงวดกว่าเดิม
                const minQualityThreshold = 80; // เพิ่มจาก 71% เป็น 80% quality score
                const maxPoseAngle = 10; // ลดจาก 15 เป็น 10 degrees - เข้มงวดกว่า
                const minFaceConfidence = 0.85; // เพิ่มจาก 0.8 เป็น 0.85
                const minFaceSize = 150; // เพิ่มจาก 120 เป็น 150 pixels
                
                const passesEnhancedChecks = (
                  enhancedQuality.success &&
                  enhancedQuality.overallQuality >= minQualityThreshold &&
                  enhancedQuality.isSharp &&
                  enhancedQuality.isWellLit &&
                  enhancedQuality.isFrontal &&
                  enhancedQuality.hasLandmarks &&
                  Math.abs(enhancedQuality.poseAngles.yaw) <= maxPoseAngle &&
                  Math.abs(enhancedQuality.poseAngles.pitch) <= maxPoseAngle &&
                  Math.abs(enhancedQuality.poseAngles.roll) <= (maxPoseAngle * 1.5) && // ลดจาก *2 เป็น *1.5
                  face.bbox.confidence >= minFaceConfidence &&
                  Math.min(face_bbox.width, face_bbox.height) >= minFaceSize &&
                  isReal
                );
                
                setFaceQuality(enhancedQuality.overallQuality);
                setDetectionProgress(Math.min(enhancedQuality.overallQuality, 100));

                if (passesEnhancedChecks && imageCount < TARGET_IMAGES && !isProcessingRef.current && !hasCompletedRef.current) {
                  console.log(`🎯 Auto-capturing PREMIUM quality face image ${imageCount + 1}/${TARGET_IMAGES}`, {
                    qualityScore: enhancedQuality.overallQuality.toFixed(1),
                    grade: enhancedQuality.qualityGrade,
                    confidence: face.bbox.confidence.toFixed(3),
                    faceSize: `${face_bbox.width}x${face_bbox.height}`,
                    poseAngles: enhancedQuality.poseAngles,
                    isReal
                  });
                  
                  await captureAndSaveCroppedImage(croppedCanvas);
                } else {
                  // แสดงเหตุผลที่ไม่ผ่าน Enhanced Quality Check
                  const reasons = [];
                  if (!enhancedQuality.success) reasons.push('Enhanced check failed');
                  if (enhancedQuality.overallQuality < minQualityThreshold) reasons.push(`Quality too low (${enhancedQuality.overallQuality.toFixed(1)}%)`);
                  if (!enhancedQuality.isSharp) reasons.push('Image blurry');
                  if (!enhancedQuality.isWellLit) reasons.push('Poor lighting');
                  if (!enhancedQuality.isFrontal) reasons.push('Not frontal');
                  if (!enhancedQuality.hasLandmarks) reasons.push('No landmarks detected');
                  if (Math.abs(enhancedQuality.poseAngles.yaw) > maxPoseAngle) reasons.push(`Yaw angle too large (${enhancedQuality.poseAngles.yaw.toFixed(1)}°)`);
                  if (Math.abs(enhancedQuality.poseAngles.pitch) > maxPoseAngle) reasons.push(`Pitch angle too large (${enhancedQuality.poseAngles.pitch.toFixed(1)}°)`);
                  if (face.bbox.confidence < 0.8) reasons.push('Low detection confidence');
                  if (!isReal) reasons.push('Not real face');
                  
                  console.log(`⚠️ Enhanced quality check failed:`, {
                    reasons: reasons.join(', '),
                    recommendations: enhancedQuality.recommendations,
                    currentValues: {
                      quality: enhancedQuality.overallQuality.toFixed(1),
                      yaw: enhancedQuality.poseAngles.yaw.toFixed(1),
                      pitch: enhancedQuality.poseAngles.pitch.toFixed(1),
                      confidence: face.bbox.confidence.toFixed(3),
                      faceSize: Math.min(face_bbox.width, face_bbox.height)
                    }
                  });
                }
              }, 'image/jpeg', 0.8);
            }
          } else {
            setFaceDetected(false);
            setIsRealFace(false);
            setFaceQuality(0);
            setDetectionProgress(0);
          }
        } catch (error) {
          console.error('❌ Face detection API error:', error);
        }
      });
          
    } catch (error) {
      console.error('❌ Canvas processing error:', error);
    }
  }, [isStreaming, imageCount, TARGET_IMAGES, captureAndSaveCroppedImage]);

  // Manual capture function
  const handleCapture = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || imageCount >= TARGET_IMAGES || isProcessingRef.current || hasCompletedRef.current) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    const imageDataUrl = canvas.toDataURL('image/jpeg', 0.9);
    const newImageCount = imageCount + 1;
    
    setCapturedImages(prev => [...prev, imageDataUrl]);
    setImageCount(newImageCount);
    
    console.log(`📸 Manual capture: image ${newImageCount}/${TARGET_IMAGES}`);
  }, [imageCount, TARGET_IMAGES]);

  // 🔧 ปรับปรุง useEffect สำหรับการเริ่ม camera
  useEffect(() => {
    if (showCameraModal && !initializingRef.current && !streamRef.current) {
      initCamera();
    } else if (!showCameraModal && streamRef.current) {
      stopCamera();
    }
  }, [showCameraModal, initCamera, stopCamera]);

  // 🔧 ปรับปรุง useEffect สำหรับ face detection
  useEffect(() => {
    if (showCameraModal && isStreaming && step === 'detecting' && 
        imageCount < TARGET_IMAGES && !isProcessingRef.current && !hasCompletedRef.current) {
      
      console.log('🔍 Starting face detection interval');
      
      const startTimeout = setTimeout(() => {
        if (!isProcessingRef.current && !hasCompletedRef.current) {
          detectionIntervalRef.current = setInterval(() => {
            if (!isProcessingRef.current && !hasCompletedRef.current && imageCount < TARGET_IMAGES) {
              detectFace();
            }
          }, 500); // คืนกลับเป็น 500ms เหมือนเดิม
        }
      }, 500);

      return () => {
        clearTimeout(startTimeout);
        if (detectionIntervalRef.current) {
          clearInterval(detectionIntervalRef.current);
          detectionIntervalRef.current = null;
        }
      };
    }

    return () => {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
        detectionIntervalRef.current = null;
      }
    };
  }, [showCameraModal, isStreaming, step, imageCount, TARGET_IMAGES, detectFace]);

  // 🔧 Cleanup on unmount
  useEffect(() => {
    return () => {
      console.log('🧹 Component unmounting, cleaning up...');
      cleanupTimeouts();
      stopCamera();
      isProcessingRef.current = false;
      hasCompletedRef.current = false;
    };
  }, [cleanupTimeouts, stopCamera]);

  // 🔧 ฟังก์ชันตรวจสอบความคมชัดของภาพ
  const calculateImageSharpness = useCallback((canvas: HTMLCanvasElement, faceBox: any): number => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return 0;

    // Extract face region
    const faceRegion = ctx.getImageData(
      faceBox.x, 
      faceBox.y, 
      faceBox.width, 
      faceBox.height
    );
    
    const data = faceRegion.data;
    const width = faceRegion.width;
    const height = faceRegion.height;
    
    // Convert to grayscale and calculate Laplacian variance (sharpness metric)
    let variance = 0;
    let mean = 0;
    
    // Calculate mean
    for (let i = 0; i < data.length; i += 4) {
      const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      mean += gray;
    }
    mean /= (data.length / 4);
    
    // Calculate variance (simplified sharpness detection)
    for (let i = 0; i < data.length; i += 4) {
      const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      variance += Math.pow(gray - mean, 2);
    }
    variance /= (data.length / 4);
    
    // Return normalized sharpness score (0-100)
    return Math.min(100, Math.max(0, variance / 100));
  }, []);

  // 🔧 ฟังก์ชันตรวจสอบการโฟกัสของใบหน้า
  const checkFaceQuality = useCallback((canvas: HTMLCanvasElement, face: any, faceBox: any): {
    isSharp: boolean;
    isWellLit: boolean;
    isFrontal: boolean;
    overallQuality: number;
  } => {
    // 1. ตรวจสอบความคมชัด
    const sharpness = calculateImageSharpness(canvas, faceBox);
    const isSharp = sharpness > 30; // threshold สำหรับความคมชัด
    
    // 2. ตรวจสอบแสง - ใช้ confidence เป็นตัวชี้วัด
    const brightness = face.bbox.confidence;
    const isWellLit = brightness > 0.8; // แสงเพียงพอ
    
    // 3. ตรวจสอบว่าใบหน้าหันหน้าตรง (ใช้ landmarks หากมี)
    const landmarks = face.landmarks;
    let isFrontal = true;
    
    if (landmarks && landmarks.length >= 5) {
      // Simple frontal face check using eye positions
      const leftEye = landmarks[0];
      const rightEye = landmarks[1];
      
      if (leftEye && rightEye) {
        const eyeDistance = Math.abs(leftEye.x - rightEye.x);
        const faceWidth = faceBox.width;
        const eyeRatio = eyeDistance / faceWidth;
        
        // ตรวจสอบว่าใบหน้าหันตรงหรือไม่
        isFrontal = eyeRatio > 0.2 && eyeRatio < 0.5;
      }
    }
    
    // 4. คำนวณคุณภาพรวม
    const qualityFactors = [
      isSharp ? 30 : 0,
      isWellLit ? 30 : 0,
      isFrontal ? 25 : 0,
      Math.min(15, sharpness / 2)
    ];
    
    const overallQuality = qualityFactors.reduce((sum, factor) => sum + factor, 0);
    
    return {
      isSharp,
      isWellLit,
      isFrontal,
      overallQuality: Math.min(100, overallQuality)
    };
  }, [calculateImageSharpness]);

  // 🚀 NEW: Enhanced Face Quality Check using MediaPipe APIs
  const checkFaceQualityEnhanced = useCallback(async (canvas: HTMLCanvasElement): Promise<{
    success: boolean;
    isSharp: boolean;
    isWellLit: boolean;
    isFrontal: boolean;
    hasLandmarks: boolean;
    overallQuality: number;
    qualityGrade: string;
    poseAngles: { yaw: number; pitch: number; roll: number };
    recommendations: string[];
    error?: string;
  }> => {
    try {
      // Convert canvas to blob
      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob((blob) => {
          if (blob) resolve(blob);
        }, 'image/jpeg', 0.8);
      });

      // Prepare form data
      const formData = new FormData();
      formData.append('file', blob, 'face_check.jpg');
      formData.append('person_id', userInfo?.userId || 'temp_user');
      formData.append('strict_mode', 'true');
      formData.append('use_landmarks', 'true');

      // Call enhanced validation API
      const response = await fetch('http://localhost:8080/api/face-enhanced/validate-quality', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }

      const result = await response.json();
      
      // Parse quality assessment
      const qualityAssessment = result.quality_assessment || {};
      const landmarkAnalysis = result.landmark_analysis || {};
      const poseEstimation = landmarkAnalysis.pose_estimation || {};

      // Extract quality metrics
      const detailedScores = qualityAssessment.detailed_scores || {};
      const sharpnessScore = parseFloat(detailedScores.sharpness || '0');
      const brightnessScore = parseFloat(detailedScores.brightness || '0');
      const frontalScore = parseFloat(detailedScores.frontal || '0');
      const overallScore = parseFloat(qualityAssessment.overall_score || '0');

      return {
        success: result.success || false,
        isSharp: sharpnessScore > 0.6,
        isWellLit: brightnessScore > 0.6,
        isFrontal: frontalScore > 0.7,
        hasLandmarks: landmarkAnalysis.landmarks_found || false,
        overallQuality: overallScore * 100, // Convert to percentage
        qualityGrade: qualityAssessment.quality_grade || 'Poor',
        poseAngles: {
          yaw: parseFloat(poseEstimation.yaw || '0'),
          pitch: parseFloat(poseEstimation.pitch || '0'),
          roll: parseFloat(poseEstimation.roll || '0')
        },
        recommendations: qualityAssessment.recommendations || []
      };

    } catch (error) {
      console.error('Enhanced quality check failed:', error);
      return {
        success: false,
        isSharp: false,
        isWellLit: false,
        isFrontal: false,
        hasLandmarks: false,
        overallQuality: 0,
        qualityGrade: 'Error',
        poseAngles: { yaw: 0, pitch: 0, roll: 0 },
        recommendations: ['ไม่สามารถตรวจสอบคุณภาพได้'],
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }, [userInfo?.userId]);

  const renderSetupStep = () => (
    <div className="text-center">
      <div className="mb-6">
        <CameraOutlined className="text-6xl text-blue-500 mb-4" />
        <Title level={4}>สแกนใบหน้าเพื่อความปลอดภัย</Title>
        <Paragraph type="secondary">
          ระบบจะใช้ AI เพื่อตรวจจับใบหน้าจริง ตรวจสอบคุณภาพ และสร้าง Face Embedding เพื่อความปลอดภัย
        </Paragraph>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
        <Row gutter={16}>
          <Col span={12}>
            <div className="text-center">
              <InfoCircleOutlined className="text-2xl text-blue-500 mb-2" />
              <Text strong>ขั้นตอนการสแกน</Text>
              <ul className="text-left text-sm mt-2">
                <li>• <strong>Face Detection:</strong> ตรวจจับใบหน้าและตัดภาพ</li>
                <li>• <strong>Anti-Spoofing:</strong> ตรวจสอบใบหน้าจริง</li>
                <li>• <strong>Quality Check:</strong> ประเมินคุณภาพภาพ</li>
                <li>• <strong>Embedding:</strong> สร้าง Face Embedding</li>
              </ul>
            </div>
          </Col>
          <Col span={12}>
            <div className="text-center">
              <CheckCircleOutlined className="text-2xl text-green-500 mb-2" />
              <Text strong>เคล็ดลับ</Text>
              <ul className="text-left text-sm mt-2">
                <li>• ใบหน้าควรอยู่ในแสงสว่างเพียงพอ</li>
                <li>• มองตรงเข้ากล้อง หลีกเลี่ยงการเอียง &gt;10°</li>
                <li>• ไม่สวมแว่นกันแดดหรือหน้ากาก</li>
                <li>• ระบบตรวจสอบคุณภาพ ≥80% เท่านั้น</li>
                <li>• เก็บภาพอัตโนมัติทุก 0.5 วินาที</li>
                <li>• ต้องการ {TARGET_IMAGES} ภาพคุณภาพสูง</li>
              </ul>
            </div>
          </Col>
        </Row>
      </div>

      <Space direction="vertical" size="middle" style={{ width: '100%' }}>
        <Button
          type="primary"
          size="large"
          icon={<CameraOutlined />}
          onClick={startCameraModal}
          loading={isProcessing || externalLoading}
          disabled={isProcessingRef.current || hasCompletedRef.current}
        >
          เริ่มสแกนใบหน้า
        </Button>
        
        <Button
          type="link"
          onClick={() => setShowGuide(true)}
          icon={<InfoCircleOutlined />}
        >
          วิธีการสแกนใบหน้าที่ถูกต้อง
        </Button>

        {/* เพิ่มปุ่มกลับไปแก้ไขข้อมูล */}
        {onBack && (
          <Button
            type="default"
            onClick={onBack}
            disabled={isProcessing || externalLoading}
          >
            กลับไปแก้ไขข้อมูล
          </Button>
        )}
      </Space>
    </div>
  );

  const renderRegisteringStep = () => (
    <div className="text-center">
      <div className="mb-6">
        <Spin size="large" />
        <Title level={4} className="mt-4">กำลังลงทะเบียนใบหน้า...</Title>
        <Paragraph type="secondary">
          กรุณารอสักครู่ ระบบกำลังประมวลผลภาพ {TARGET_IMAGES} ภาพเพื่อสร้างโปรไฟล์ใบหน้า
        </Paragraph>
      </div>
      
      <Progress 
        percent={registrationProgress} 
        status="active" 
        strokeColor={{
          '0%': '#108ee9',
          '100%': '#87d068',
        }}
      />
      
      <div className="mt-4 text-sm text-gray-500">
        <p>• กำลังสกัดจุดเด่นของใบหน้าจากภาพ {Math.min(capturedImages.length, TARGET_IMAGES)} ภาพ...</p>
        <p>• กำลังสร้างลายเซ็นดิจิทัลของใบหน้า...</p>
        <p>• กำลังเข้ารหัสข้อมูลเพื่อความปลอดภัย...</p>
      </div>
    </div>
  );

  const renderCompletedStep = () => (
    <div className="text-center">
      <div className="mb-6">
        <CheckCircleOutlined className="text-6xl text-green-500 mb-4" />
        <Title level={4}>เก็บภาพครบแล้ว!</Title>
        <Paragraph type="secondary">
          ระบบได้เก็บภาพใบหน้าของคุณครบ {TARGET_IMAGES} ภาพแล้ว
        </Paragraph>
        <Paragraph type="secondary">
          กำลังปิดกล้องและเริ่มประมวลผลการลงทะเบียน...
        </Paragraph>
      </div>
      
      <Progress percent={100} status="success" strokeColor="#52c41a" />
      
      <div className="mt-4">
        <Alert
          message="เสร็จสิ้นการสแกน"
          description="ระบบจะเริ่มประมวลผลข้อมูลใบหน้าของคุณในอีกสักครู่"
          type="success"
          showIcon
        />
      </div>
    </div>
  );

  return (
    <Card className="w-full max-w-2xl mx-auto">
      {error && (
        <Alert
          message="เกิดข้อผิดพลาด"
          description={error}
          type="error"
          showIcon
          closable
          onClose={() => setError(null)}
          className="mb-4"
        />
      )}

      {step === 'setup' && renderSetupStep()}
      {step === 'registering' && renderRegisteringStep()}
      {step === 'completed' && renderCompletedStep()}

      {/* Camera Modal */}
      <Modal
        title={
          <Space>
            <CameraOutlined />
            {imageCount >= TARGET_IMAGES ? 'เก็บภาพครบแล้ว!' : 'สแกนใบหน้าเพื่อความปลอดภัย'}
            {isStreaming && faceDetected && imageCount < TARGET_IMAGES && (
              <Badge status="success" text="ตรวจพบใบหน้า" />
            )}
            {isStreaming && !faceDetected && imageCount < TARGET_IMAGES && (
              <Badge status="processing" text="กำลังตรวจจับ..." />
            )}
            {imageCount >= TARGET_IMAGES && (
              <Badge status="success" text="เสร็จสิ้น" />
            )}
          </Space>
        }
        open={showCameraModal}
        onCancel={closeCameraModal}
        width={800}
        footer={
          imageCount >= TARGET_IMAGES ? [
            <Button key="close" onClick={closeCameraModal} type="primary">
              ปิด
            </Button>
          ] : [
            <Button key="close" onClick={closeCameraModal}>
              ปิด
            </Button>,
            <Button 
              key="manual-capture" 
              type="primary" 
              onClick={handleCapture}
              disabled={!faceDetected || !isRealFace || faceQuality < 70 || imageCount >= TARGET_IMAGES || isProcessingRef.current || hasCompletedRef.current}
            >
              📸 จับภาพด้วยตนเอง ({imageCount}/{TARGET_IMAGES})
            </Button>
          ]
        }
        destroyOnClose
      >
        <div className="space-y-4">
          {error && (
            <Alert
              message="ข้อผิดพลาด"
              description={error}
              type="error"
              showIcon
            />
          )}

          {/* Statistics */}
          {isStreaming && (
            <Card size="small">
              <Row gutter={16}>
                <Col span={6}>
                  <Statistic
                    title="ภาพที่เก็บแล้ว"
                    value={imageCount}
                    suffix={`/${TARGET_IMAGES}`}
                    valueStyle={{ 
                      color: imageCount >= TARGET_IMAGES ? '#3f8600' : '#1890ff' 
                    }}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="คุณภาพใบหน้า"
                    value={faceQuality}
                    precision={1}
                    suffix="%"
                    valueStyle={{ 
                      color: faceQuality >= 80 ? '#3f8600' : faceQuality >= 70 ? '#faad14' : '#cf1322' 
                    }}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="Anti-Spoofing"
                    value={isRealFace ? "ผ่าน" : "ตรวจสอบ"}
                    valueStyle={{ 
                      color: isRealFace ? '#3f8600' : '#faad14' 
                    }}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="สถานะ"
                    value={faceDetected ? "ตรวจพบ" : "กำลังค้นหา"}
                    valueStyle={{ 
                      color: faceDetected ? '#3f8600' : '#1890ff' 
                    }}
                  />
                </Col>
              </Row>
            </Card>
          )}

          {/* Video */}
          <div className="relative bg-black rounded-lg overflow-hidden">
            <video
              ref={videoRef}  
              autoPlay
              playsInline
              muted
              className="w-full h-auto"
              style={{ 
                maxHeight: '400px',
                transform: 'scaleX(-1)'
              }}
            />
            
            <canvas
              ref={canvasRef}
              className="hidden"
            />

            {/* Overlays */}
            {faceDetected && imageCount < TARGET_IMAGES && (
              <div className="absolute top-4 left-4 bg-green-500 text-white px-3 py-1 rounded-full">
                <CheckCircleOutlined className="mr-1" />
                ตรวจพบใบหน้า
              </div>
            )}

            {faceDetected && imageCount < TARGET_IMAGES && (
              <div className={`absolute top-4 right-4 px-3 py-1 rounded-full text-white ${
                isRealFace ? 'bg-green-500' : 'bg-orange-500'
              }`}>
                {isRealFace ? '✓ คนจริง' : '⚠ ตรวจสอบ'}
              </div>
            )}

            {imageCount >= TARGET_IMAGES && (
              <div className="absolute inset-0 bg-green-500 bg-opacity-80 flex items-center justify-center rounded-lg">
                <div className="text-center text-white">
                  <CheckCircleOutlined className="text-6xl mb-4" />
                  <h3 className="text-xl font-bold">เก็บภาพครบแล้ว!</h3>
                  <p>กำลังปิดกล้องและเริ่มประมวลผล...</p>
                </div>
              </div>
            )}

            <div className="absolute bottom-4 left-4 bg-blue-500 text-white px-3 py-1 rounded-full">
              {imageCount}/{TARGET_IMAGES} ภาพ
            </div>
          </div>

          {/* Progress bars */}
          <Row gutter={16}>
            <Col span={12}>
              <div className="text-center">
                <Text>ความคืบหน้าคุณภาพ</Text>
                <Progress
                  percent={detectionProgress}
                  status={detectionProgress >= 80 ? 'success' : 'active'}
                  strokeColor={{
                    '0%': '#108ee9',
                    '100%': '#87d068',
                  }}
                />
              </div>
            </Col>
            <Col span={12}>
              <div className="text-center">
                <Text>ความคืบหน้าการเก็บภาพ</Text>
                <Progress
                  percent={Math.round((imageCount / TARGET_IMAGES) * 100)}
                  status={imageCount >= TARGET_IMAGES ? 'success' : 'active'}
                  strokeColor={{
                    '0%': '#fa541c',
                    '100%': '#52c41a',
                  }}
                />
              </div>
            </Col>
          </Row>

          {/* Alerts */}
          <Alert
            message="เงื่อนไขการเก็บภาพใบหน้า (เข้มงวด)"
            description="• คุณภาพภาพ ≥80% • ใบหน้าตรงเข้ากล้อง (เอียงไม่เกิน 10°) • ขนาดใบหน้า ≥150 pixels • แสงสว่างเพียงพอ • ผ่านการตรวจสอบใบหน้าจริง • ระบบเก็บอัตโนมัติทุก 0.5 วินาที • เก็บเฉพาะภาพที่ผ่านเกณฑ์เท่านั้น"
            type="info"
            showIcon
          />

          {faceDetected && isRealFace && faceQuality >= 80 && imageCount < TARGET_IMAGES && (
            <Alert
              message="✅ กำลังเก็บภาพใบหน้าคุณภาพสูง!"
              description={`ระบบกำลังเก็บภาพอัตโนมัติทุก 0.5 วินาที (${imageCount}/${TARGET_IMAGES}) - คุณภาพ: ${faceQuality.toFixed(1)}% (ใช้งานได้)`}
              type="success"
              showIcon
            />
          )}

          {faceDetected && isRealFace && faceQuality >= 70 && faceQuality < 80 && imageCount < TARGET_IMAGES && (
            <Alert
              message="⚠️ คุณภาพใบหน้าต้องปรับปรุง"
              description={`คุณภาพปัจจุบัน: ${faceQuality.toFixed(1)}% (ต้องการ ≥80%) - กรุณาปรับแสงและมองตรงเข้ากล้อง`}
              type="warning"
              showIcon
            />
          )}

          {faceDetected && (!isRealFace || faceQuality < 70) && imageCount < TARGET_IMAGES && (
            <Alert
              message="❌ ไม่สามารถเก็บภาพได้"
              description={`${!isRealFace ? 'ไม่ผ่านการตรวจสอบใบหน้าจริง' : `คุณภาพต่ำ: ${faceQuality.toFixed(1)}%`} - กรุณาทำตามคำแนะนำด้านบน`}
              type="error"
              showIcon
            />
          )}

          {imageCount >= TARGET_IMAGES && (
            <Alert
              message="เก็บภาพครบแล้ว!"
              description="กำลังปิดกล้องและเริ่มประมวลผลการลงทะเบียน..."
              type="success"
              showIcon
            />
          )}
        </div>
      </Modal>

      {/* Guide Modal */}
      <Modal
        title="วิธีการสแกนใบหน้าที่ถูกต้อง"
        open={showGuide}
        onCancel={() => setShowGuide(false)}
        footer={[
          <Button key="close" type="primary" onClick={() => setShowGuide(false)}>
            เข้าใจแล้ว
          </Button>
        ]}
      >
        <div className="space-y-4">
          <div>
            <Title level={5}>✅ ควรทำ:</Title>
            <ul>
              <li>• ใช้แสงธรรมชาติหรือแสงสว่างเพียงพอ</li>
              <li>• ใบหน้าอยู่ตรงกลางกรอบ</li>
              <li>• มองตรงเข้ากล้อง</li>
              <li>• ใบหน้าชัดเจน ไม่เบลอ</li>
              <li>• ไม่มีผมปิดบังใบหน้า</li>
            </ul>
          </div>
          
          <div>
            <Title level={5}>❌ ไม่ควรทำ:</Title>
            <ul>
              <li>• สวมแว่นกันแดดหรือแว่นสีเข้ม</li>
              <li>• ใช้หน้ากากหรือผ้าปิดใบหน้า</li>
              <li>• ถ่ายในที่มืดหรือแสงน้อย</li>
              <li>• หันหน้าไปด้านข้าง</li>
              <li>• ไกลกล้องหรือใกล้กล้องเกินไป</li>
            </ul>
          </div>
        </div>
      </Modal>
    </Card>
  );
};

export default FaceRegistration;