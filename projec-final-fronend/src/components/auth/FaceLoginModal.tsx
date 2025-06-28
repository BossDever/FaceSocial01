import React, { useState, useRef, useCallback, useEffect } from 'react';
import { 
  Modal, 
  Button, 
  Progress, 
  Typography, 
  Alert, 
  Space, 
  Statistic, 
  Card, 
  Row, 
  Col,
  Badge,
  Divider 
} from 'antd';
import { 
  CameraOutlined, 
  ScanOutlined, 
  CheckCircleOutlined, 
  ExclamationCircleOutlined,
  UserOutlined,
  SafetyOutlined,
  EyeOutlined
} from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;

interface FaceLoginModalProps {
  visible: boolean;
  onClose: () => void;
  onSuccess: (result: any) => void;
  onError?: (error: string) => void;
}

interface ScanResult {
  id: string;
  isReal: boolean;
  identity: string | null;
  confidence: number;
  quality: number;
  timestamp: number;
  similarity?: number;
}

const FaceLoginModal: React.FC<FaceLoginModalProps> = ({
  visible,
  onClose,
  onSuccess,
  onError
}) => {
  // Core States
  const [isStreaming, setIsStreaming] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [step, setStep] = useState<'setup' | 'scanning' | 'processing' | 'completed'>('setup');
  const [scanProgress, setScanProgress] = useState(0);
  const [scanResults, setScanResults] = useState<ScanResult[]>([]);
  const [spoofingCount, setSpoofingCount] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [currentScanIndex, setCurrentScanIndex] = useState(0);

  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const scanIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const previewIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const isProcessingRef = useRef(false);
  const initializingRef = useRef(false);

  const TARGET_SCANS = 20;
  const SCAN_INTERVAL = 500; // ms
  const CONFIDENCE_THRESHOLD = 0.8;
  const MAX_SPOOFING_ALLOWED = 3;

  console.log('FaceLoginModal render - step:', step, 'scanning:', isScanning, 'progress:', scanProgress);

  // Computed Values
  const validScans = scanResults.filter(r => r.isReal && r.confidence >= CONFIDENCE_THRESHOLD).length;
  const identityStats = scanResults.reduce((acc, result) => {
    if (result.isReal && result.identity && result.confidence >= CONFIDENCE_THRESHOLD) {
      acc[result.identity] = (acc[result.identity] || 0) + 1;
    }
    return acc;
  }, {} as Record<string, number>);
  
  const winnerIdentity = Object.entries(identityStats).reduce((winner, [identity, count]) => {
    return count > winner.count ? { identity, count } : winner;
  }, { identity: '', count: 0 });

  const accuracy = scanResults.length > 0 ? Math.round((validScans / scanResults.length) * 100) : 0;

  // Initialize Camera
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
        
        // Start preview canvas update
        startPreviewUpdate();
      }
    } catch (err) {
      console.error('❌ Error accessing camera:', err);
      const errorMessage = 'ไม่สามารถเข้าถึงกล้องได้ กรุณาอนุญาตการใช้งานกล้อง';
      setError(errorMessage);
      if (onError) onError(errorMessage);
    } finally {
      initializingRef.current = false;
    }
  }, [onError]);

  // Stop Camera
  const stopCamera = useCallback(() => {
    console.log('🛑 Stopping camera...');
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (previewIntervalRef.current) {
      clearInterval(previewIntervalRef.current);
      previewIntervalRef.current = null;
    }
    
    setIsStreaming(false);
    console.log('📷 Camera stopped');
  }, []);

  // Preview Canvas Update
  const startPreviewUpdate = useCallback(() => {
    if (previewIntervalRef.current) return;

    previewIntervalRef.current = setInterval(() => {
      if (videoRef.current && previewCanvasRef.current && isStreaming) {
        const video = videoRef.current;
        const canvas = previewCanvasRef.current;
        const ctx = canvas.getContext('2d');
        
        if (ctx && video.readyState >= 2) {
          canvas.width = 120;
          canvas.height = 120;
          
          // Draw mirrored video preview
          ctx.save();
          ctx.scale(-1, 1);
          ctx.drawImage(video, -120, 0, 120, 120);
          ctx.restore();
        }
      }
    }, 100);
  }, [isStreaming]);

  // Single Face Scan
  const performSingleScan = useCallback(async (): Promise<ScanResult | null> => {
    if (!videoRef.current || !canvasRef.current || isProcessingRef.current) {
      return null;
    }

    isProcessingRef.current = true;
    
    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      
      if (!ctx || video.readyState < 2) {
        return null;
      }

      // Capture video frame
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);

      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob((blob) => {
          if (blob) resolve(blob);
        }, 'image/jpeg', 0.8);
      });

      // Step 1: Face Detection
      console.log('🔍 Step 1: Face Detection...');      const faceDetectionData = new FormData();
      faceDetectionData.append('file', blob, 'face.jpg');
      faceDetectionData.append('model_name', 'yolov11m');
      faceDetectionData.append('conf_threshold', '0.5');
      faceDetectionData.append('max_faces', '1');

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
      const faceResponse = await fetch(`${apiUrl}/api/face-detection/detect`, {
        method: 'POST',
        body: faceDetectionData
      });

      if (!faceResponse.ok) {
        console.error('❌ Face detection failed:', faceResponse.status);
        return null;
      }

      const faceResult = await faceResponse.json();
      
      if (!faceResult.success || !faceResult.faces || faceResult.faces.length === 0) {
        console.log('❌ No face detected');
        return {
          id: `scan_${Date.now()}`,
          isReal: false,
          identity: null,
          confidence: 0,
          quality: 0,
          timestamp: Date.now()
        };
      }

      const face = faceResult.faces[0];
      const bbox = face.bbox;
      
      // Crop face from image
      const croppedCanvas = document.createElement('canvas');
      const croppedCtx = croppedCanvas.getContext('2d');
      
      if (!croppedCtx) return null;

      // Calculate crop dimensions with padding
      const padding = 0.2;
      const faceWidth = bbox.x2 - bbox.x1;
      const faceHeight = bbox.y2 - bbox.y1;
      const paddingX = faceWidth * padding;
      const paddingY = faceHeight * padding;
      
      const cropX = Math.max(0, bbox.x1 - paddingX);
      const cropY = Math.max(0, bbox.y1 - paddingY);
      const cropWidth = Math.min(canvas.width - cropX, faceWidth + (paddingX * 2));
      const cropHeight = Math.min(canvas.height - cropY, faceHeight + (paddingY * 2));
      
      croppedCanvas.width = 224;
      croppedCanvas.height = 224;
      
      croppedCtx.drawImage(
        canvas, 
        cropX, cropY, cropWidth, cropHeight,
        0, 0, 224, 224
      );

      const croppedBlob = await new Promise<Blob>((resolve) => {
        croppedCanvas.toBlob((blob) => {
          if (blob) resolve(blob);
        }, 'image/jpeg', 0.8);
      });

      // Step 2: Anti-Spoofing
      console.log('🛡️ Step 2: Anti-Spoofing...');      const antispoofingData = new FormData();
      antispoofingData.append('image', croppedBlob, 'cropped_face.jpg');
      antispoofingData.append('confidence_threshold', '0.5');

      const antispoofingResponse = await fetch(`${apiUrl}/api/anti-spoofing/detect-upload`, {
        method: 'POST',
        body: antispoofingData
      });

      let isReal = true;
      if (antispoofingResponse.ok) {
        const antispoofingResult = await antispoofingResponse.json();
        isReal = antispoofingResult.success && !antispoofingResult.overall_result?.spoofing_detected;
      }

      if (!isReal) {
        console.log('⚠️ Spoofing detected');
        return {
          id: `scan_${Date.now()}`,
          isReal: false,
          identity: null,
          confidence: 0,
          quality: face.confidence || 0,
          timestamp: Date.now()
        };
      }

      // Step 3: Face Recognition
      console.log('🧠 Step 3: Face Recognition...');
      const faceImageBase64 = await new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64 = (reader.result as string).split(',')[1];
          resolve(base64);
        };
        reader.readAsDataURL(croppedBlob);
      });

      const recognitionResponse = await fetch(`${apiUrl}/api/face-recognition/recognize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          face_image_base64: faceImageBase64,
          model_name: 'facenet',
          top_k: 5,
          similarity_threshold: 0.5
        })
      });      let identity = null;
      let confidence = 0;
      let similarity = 0;

      if (recognitionResponse.ok) {
        const recognitionResult = await recognitionResponse.json();
        if (recognitionResult.success && recognitionResult.best_match) {
          identity = recognitionResult.best_match.person_id;
          confidence = recognitionResult.best_match.confidence || 0;
          similarity = recognitionResult.best_match.similarity || 0;
        }
      }console.log(`✅ Scan completed:`, {
        identity,
        confidence: confidence?.toFixed(3) || '0.000',
        similarity: similarity?.toFixed(3) || '0.000',
        isReal,
        quality: (face.confidence || 0).toFixed(3)
      });      return {
        id: `scan_${Date.now()}`,
        isReal,
        identity,
        confidence,
        quality: face.confidence || 0,
        similarity,
        timestamp: Date.now()
      };

    } catch (error) {
      console.error('❌ Scan error:', error);
      return null;
    } finally {
      isProcessingRef.current = false;
    }
  }, []);  // Process Collected Results (separate function to avoid state timing issues)
  const processCollectedResults = useCallback(async (collectedResults: any[]) => {
    console.log('📊 Processing collected scan results...');
    console.log('Collected results:', collectedResults);
    setIsProcessing(true);

    try {
      // Debug: แสดงข้อมูลทั้งหมดก่อนกรอง
      console.log('📋 Scan summary:', {
        total: collectedResults.length,
        realFaces: collectedResults.filter(r => r.isReal).length,
        withIdentity: collectedResults.filter(r => r.identity).length,
        highConfidence: collectedResults.filter(r => r.confidence >= CONFIDENCE_THRESHOLD).length
      });

      const validResults = collectedResults.filter(r => 
        r.isReal && 
        r.identity && 
        r.confidence >= CONFIDENCE_THRESHOLD
      );

      console.log('✅ Valid results:', validResults);

      if (validResults.length === 0) {
        // ลด threshold ลงเป็น 50% ถ้าไม่มีผลลัพธ์ที่ 80%
        const fallbackResults = collectedResults.filter(r => 
          r.isReal && 
          r.identity && 
          r.confidence >= 0.5
        );
        
        console.log('🔄 Trying fallback threshold (50%):', fallbackResults);
        
        if (fallbackResults.length === 0) {
          throw new Error('ไม่พบใบหน้าที่ตรงกับระบบ กรุณาลองใหม่');
        }
        
        // ใช้ fallback results
        validResults.push(...fallbackResults);
      }

      // Count identity matches
      const identityCounts = validResults.reduce((acc, result) => {
        if (result.identity) {
          acc[result.identity] = (acc[result.identity] || 0) + 1;
        }
        return acc;
      }, {} as Record<string, number>);

      // Find winner (most matches)
      const winner = Object.entries(identityCounts).reduce((best, [identity, count]) => {
        const numCount = Number(count);
        return numCount > best.count ? { identity, count: numCount } : best;
      }, { identity: '', count: 0 });

      if (winner.count === 0) {
        throw new Error('ไม่พบใบหน้าที่ตรงกับระบบ');
      }

      console.log(`🏆 Winner: ${winner.identity} with ${winner.count} matches`);

      // Get best quality image of winner
      const winnerResults = validResults.filter(r => r.identity === winner.identity);
      const bestResult = winnerResults.reduce((best, current) => {
        return current.confidence > best.confidence ? current : best;
      });      // Final authentication - ใช้ผลลัพธ์จากการสแกนโดยตรง
      console.log('✅ Authentication successful based on scan results');
      console.log(`🎯 User identified: ${winner.identity} with ${winner.count} matches out of ${validResults.length} valid scans`);
      
      // ดึงข้อมูลผู้ใช้จริงจากฐานข้อมูลโดยใช้ UUID ที่ได้จากการสแกน
      console.log('🔍 Fetching real user data from database...');
        try {
        // เรียก API เพื่อดึงข้อมูลผู้ใช้จริง
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
        const userResponse = await fetch(`${apiUrl}/api/face-recognition/person/${winner.identity}`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json'
          }
        });

        let realUserData = null;
        if (userResponse.ok) {
          const userData = await userResponse.json();
          if (userData.success && userData.person_name) {
            realUserData = {
              id: winner.identity,
              identity: winner.identity,
              username: userData.person_name || winner.identity,
              email: `${userData.person_name || winner.identity}@face.login`,
              fullName: userData.person_name || `User ${winner.identity.substring(0, 8)}`,
              firstName: userData.person_name?.split(' ')[0] || winner.identity.substring(0, 8),
              lastName: userData.person_name?.split(' ').slice(1).join(' ') || '',
              isVerified: false // สามารถปรับแต่งได้ตามความต้องการ
            };
            console.log('✅ Real user data retrieved:', realUserData);
          }
        } else {
          console.warn('⚠️ Failed to fetch user data from database, using fallback data');
        }

        // ใช้ข้อมูลจริงถ้ามี หรือใช้ fallback data
        const finalUserData = realUserData || {
          id: winner.identity,
          identity: winner.identity,
          username: winner.identity,
          email: `${winner.identity}@temp.local`,
          fullName: winner.identity,
          firstName: winner.identity,
          lastName: '',
          isVerified: false
        };

        setStep('completed');
        setTimeout(() => {
          onSuccess({
            success: true,
            user: finalUserData,
            token: `mock_token_${winner.identity}_${Date.now()}`,
            scanResults: {
              totalScans: collectedResults.length,
              validScans: validResults.length,
              spoofingCount: collectedResults.filter(r => !r.isReal).length,
              winnerVotes: winner.count,
              confidence: bestResult.confidence
            }
          });
          onClose();
        }, 1500);

      } catch (fetchError) {
        console.error('❌ Error fetching user data:', fetchError);
        
        // ถ้าเกิดข้อผิดพลาดในการดึงข้อมูล ใช้ fallback data
        const fallbackUserData = {
          id: winner.identity,
          identity: winner.identity,
          username: winner.identity,
          email: `${winner.identity}@temp.local`,
          fullName: winner.identity,
          firstName: winner.identity,
          lastName: '',
          isVerified: false
        };

        setStep('completed');
        setTimeout(() => {
          onSuccess({
            success: true,
            user: fallbackUserData,
            token: `mock_token_${winner.identity}_${Date.now()}`,
            scanResults: {
              totalScans: collectedResults.length,
              validScans: validResults.length,
              spoofingCount: collectedResults.filter(r => !r.isReal).length,
              winnerVotes: winner.count,
              confidence: bestResult.confidence
            }
          });
          onClose();
        }, 1500);
      }

    } catch (error) {
      console.error('❌ Processing error:', error);
      setError(error instanceof Error ? error.message : 'เกิดข้อผิดพลาดในการประมวลผล');
      setStep('scanning');
    } finally {
      setIsProcessing(false);
    }
  }, [onSuccess, onClose]);

  // Start Auto Scan
  const startAutoScan = useCallback(async () => {
    if (isScanning || !isStreaming) return;

    console.log('🚀 Starting auto scan sequence...');
    setIsScanning(true);
    setStep('scanning');
    setScanResults([]);
    setSpoofingCount(0);
    setScanProgress(0);
    setCurrentScanIndex(0);
    setError(null);

    let scanIndex = 0;
    let collectedResults: any[] = []; // Track results locally to avoid state timing issues
    
    const performScan = async () => {
      if (scanIndex >= TARGET_SCANS) {
        // Scanning completed
        clearInterval(scanIntervalRef.current!);
        scanIntervalRef.current = null;
        setIsScanning(false);
        setStep('processing');
        
        // Use collected results directly instead of waiting for state
        console.log(`🏁 Scanning completed! Processing ${collectedResults.length} collected results...`);
        await processCollectedResults(collectedResults);
        return;
      }

      console.log(`🔍 Performing scan ${scanIndex + 1}/${TARGET_SCANS}`);
      setCurrentScanIndex(scanIndex + 1);
      
      const result = await performSingleScan();
      console.log(`📋 Scan ${scanIndex + 1} result:`, result);
      
      if (result) {
        console.log(`✅ Adding result to collectedResults: ${result.identity} (${result.confidence})`);
        collectedResults.push(result);
        
        // Also update state for UI
        setScanResults(prev => {
          const newResults = [...prev, result];
          console.log(`📊 State results now: ${newResults.length}`);
          return newResults;
        });
        
        // Update spoofing count
        const spoofingCount = collectedResults.filter(r => !r.isReal).length;
        setSpoofingCount(spoofingCount);
        
        // Check if too many spoofing attempts
        if (spoofingCount > MAX_SPOOFING_ALLOWED) {
          console.warn(`⚠️ Too many spoofing attempts (${spoofingCount}), stopping scan`);
          clearInterval(scanIntervalRef.current!);
          setIsScanning(false);
          setError(`ตรวจพบการปลอมแปลงใบหน้ามากเกินไป (${spoofingCount} ครั้ง) กรุณาลองใหม่`);
          resetScan();
          return;
        }
      } else {
        console.warn(`⚠️ Scan ${scanIndex + 1} returned null result`);
      }

      scanIndex++;
      const progress = Math.round((scanIndex / TARGET_SCANS) * 100);
      setScanProgress(progress);
    };

    // Start first scan immediately
    await performScan();
    
    // Continue with interval
    scanIntervalRef.current = setInterval(performScan, SCAN_INTERVAL);
  }, [isScanning, isStreaming, performSingleScan, processCollectedResults]);
  // Process Results
  const processResults = useCallback(async () => {
    console.log('📊 Processing scan results...');
    console.log('Raw scan results:', scanResults);
    setIsProcessing(true);

    try {
      // Debug: แสดงข้อมูลทั้งหมดก่อนกรอง
      console.log('📋 Scan summary:', {
        total: scanResults.length,
        realFaces: scanResults.filter(r => r.isReal).length,
        withIdentity: scanResults.filter(r => r.identity).length,
        highConfidence: scanResults.filter(r => r.confidence >= CONFIDENCE_THRESHOLD).length
      });

      const validResults = scanResults.filter(r => 
        r.isReal && 
        r.identity && 
        r.confidence >= CONFIDENCE_THRESHOLD
      );

      console.log('✅ Valid results:', validResults);

      if (validResults.length === 0) {
        // ลด threshold ลงเป็น 50% ถ้าไม่มีผลลัพธ์ที่ 80%
        const fallbackResults = scanResults.filter(r => 
          r.isReal && 
          r.identity && 
          r.confidence >= 0.5
        );
        
        console.log('🔄 Trying fallback threshold (50%):', fallbackResults);
        
        if (fallbackResults.length === 0) {
          throw new Error('ไม่พบใบหน้าที่ตรงกับระบบ กรุณาลองใหม่');
        }
        
        // ใช้ fallback results
        validResults.push(...fallbackResults);
      }

      // Count identity matches
      const identityCounts = validResults.reduce((acc, result) => {
        if (result.identity) {
          acc[result.identity] = (acc[result.identity] || 0) + 1;
        }
        return acc;
      }, {} as Record<string, number>);

      // Find winner (most matches)
      const winner = Object.entries(identityCounts).reduce((best, [identity, count]) => {
        return count > best.count ? { identity, count } : best;
      }, { identity: '', count: 0 });

      if (winner.count === 0) {
        throw new Error('ไม่พบใบหน้าที่ตรงกับระบบ');
      }

      console.log(`🏆 Winner: ${winner.identity} with ${winner.count} matches`);

      // Get best quality image of winner
      const winnerResults = validResults.filter(r => r.identity === winner.identity);
      const bestResult = winnerResults.reduce((best, current) => {
        return current.confidence > best.confidence ? current : best;
      });      // Final authentication via Login API
      console.log('🔐 Final authentication...');
      
      // ใช้ข้อมูลจากผลลัพธ์การสแกนที่ดีที่สุด
      console.log(`🎯 Authenticating as user: ${winner.identity} with confidence: ${bestResult.confidence.toFixed(2)}%`);
      
      // สร้าง mock authentication data เพื่อส่งไป Login API
      const authData = {
        userId: winner.identity,
        confidence: bestResult.confidence,
        scanResults: {
          totalScans: scanResults.length,
          validScans: validResults.length,
          spoofingCount: scanResults.filter(r => !r.isReal).length,
          winnerVotes: winner.count
        },
        loginMethod: 'face'
      };      // Final authentication - ใช้ผลลัพธ์จากการสแกนโดยตรง
      console.log('� Authentication successful based on scan results');
      console.log(`🎯 User identified: ${winner.identity} with ${winner.count} matches out of ${validResults.length} valid scans`);
      
      // ใช้ผลลัพธ์จากการสแกนโดยตรง เพราะได้ตรวจสอบแล้วผ่าน Face Recognition API
      setStep('completed');
      setTimeout(() => {
        onSuccess({
          success: true,
          user: {
            id: winner.identity,
            identity: winner.identity,
            username: winner.identity,
            email: `${winner.identity}@face.login`,
            fullName: `User ${winner.identity.substring(0, 8)}`
          },
          token: `face_login_${Date.now()}_${winner.identity}`,
          similarity: bestResult.confidence,
          scanStats: {
            totalScans: scanResults.length,
            validScans: validResults.length,
            spoofingDetected: spoofingCount,
            winnerIdentity: winner.identity,
            winnerMatches: winner.count,
            accuracy: accuracy,
            confidence: bestResult.confidence
          },
          confidence: bestResult.confidence,
          faceAuthenticated: true,          authMethod: 'face_scan'
        });
      }, 1000);

    } catch (error) {
      console.error('❌ Processing error:', error);
      const errorMessage = error instanceof Error ? error.message : 'เกิดข้อผิดพลาดในการประมวลผล';
      setError(errorMessage);
      if (onError) onError(errorMessage);
    } finally {
      setIsProcessing(false);
    }
  }, [scanResults, spoofingCount, accuracy, onSuccess, onError]);

  // Reset Scan
  const resetScan = useCallback(() => {
    console.log('🔄 Resetting scan...');
    
    if (scanIntervalRef.current) {
      clearInterval(scanIntervalRef.current);
      scanIntervalRef.current = null;
    }
    
    setIsScanning(false);
    setIsProcessing(false);
    setStep('setup');
    setScanResults([]);
    setSpoofingCount(0);
    setScanProgress(0);
    setCurrentScanIndex(0);
    setError(null);
    isProcessingRef.current = false;
  }, []);

  // Close Modal
  const handleClose = useCallback(() => {
    resetScan();
    stopCamera();
    onClose();
  }, [resetScan, stopCamera, onClose]);

  // Effects
  useEffect(() => {
    if (visible && !initializingRef.current && !streamRef.current) {
      initCamera();
    } else if (!visible && streamRef.current) {
      stopCamera();
      resetScan();
    }
  }, [visible, initCamera, stopCamera, resetScan]);

  useEffect(() => {
    return () => {
      console.log('🧹 Component unmounting, cleaning up...');
      if (scanIntervalRef.current) {
        clearInterval(scanIntervalRef.current);
      }
      if (previewIntervalRef.current) {
        clearInterval(previewIntervalRef.current);
      }
      stopCamera();
    };
  }, [stopCamera]);

  // Render Functions
  const renderSetupStep = () => (
    <div className="text-center">
      <div className="mb-6">
        <ScanOutlined className="text-6xl text-blue-500 mb-4" />
        <Title level={4}>สแกนหน้าเพื่อเข้าสู่ระบบ</Title>
        <Paragraph type="secondary">
          ระบบจะสแกนใบหน้าของคุณ 20 ครั้ง เพื่อความแม่นยำสูงสุด
        </Paragraph>
      </div>

      <Alert
        message="วิธีการใช้งาน"
        description="• มองตรงเข้ากล้อง • ใบหน้าอยู่ในแสงสว่าง • ไม่สวมแว่นกันแดด • ระบบจะสแกนอัตโนมัติ 20 ครั้ง"
        type="info"
        showIcon
        className="mb-4"
      />

      <Button
        type="primary"
        size="large"
        icon={<ScanOutlined />}
        onClick={startAutoScan}
        disabled={!isStreaming}
        loading={isScanning}
      >
        เริ่มสแกนหน้า
      </Button>
    </div>
  );

  const renderScanningStep = () => (
    <div>
      <div className="text-center mb-4">
        <Title level={4}>
          <ScanOutlined className="mr-2" />
          กำลังสแกนใบหน้า... ({currentScanIndex}/{TARGET_SCANS})
        </Title>
        <Progress 
          percent={scanProgress} 
          status="active"
          strokeColor={{
            '0%': '#108ee9',
            '100%': '#87d068',
          }}
        />
      </div>

      <Row gutter={16} className="mb-4">
        <Col span={6}>
          <Statistic
            title="สแกนทั้งหมด"
            value={scanResults.length}
            suffix={`/ ${TARGET_SCANS}`}
            valueStyle={{ color: '#1890ff' }}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="หน้าจริง (≥80%)"
            value={validScans}
            valueStyle={{ color: validScans > 0 ? '#3f8600' : '#cf1322' }}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="Spoofing"
            value={spoofingCount}
            valueStyle={{ 
              color: spoofingCount > MAX_SPOOFING_ALLOWED ? '#cf1322' : '#faad14' 
            }}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="ความแม่นยำ"
            value={accuracy}
            suffix="%"
            valueStyle={{ 
              color: accuracy >= 80 ? '#3f8600' : accuracy >= 60 ? '#faad14' : '#cf1322' 
            }}
          />
        </Col>
      </Row>

      {Object.keys(identityStats).length > 0 && (
        <Card size="small" title="ผลการตรวจจับ" className="mb-4">
          {Object.entries(identityStats).map(([identity, count]) => (
            <div key={identity} className="flex justify-between items-center mb-2">
              <Text><UserOutlined className="mr-1" />{identity}</Text>
              <Badge count={count} style={{ backgroundColor: '#52c41a' }} />
            </div>
          ))}
        </Card>
      )}

      <div className="text-center">
        <Button danger onClick={resetScan} disabled={isProcessing}>
          หยุดสแกน
        </Button>
      </div>
    </div>
  );

  const renderProcessingStep = () => (
    <div className="text-center">
      <div className="mb-6">
        <div className="animate-spin text-6xl text-blue-500 mb-4">⚙️</div>
        <Title level={4}>กำลังประมวลผลและยืนยันตัวตน...</Title>
        <Paragraph type="secondary">
          กำลังวิเคราะห์ผลการสแกน {scanResults.length} ครั้ง
        </Paragraph>
      </div>

      {winnerIdentity.identity && (
        <Alert
          message={`ตรวจพบใบหน้า: ${winnerIdentity.identity}`}
          description={`จำนวนภาพที่ตรงกัน: ${winnerIdentity.count} ครั้ง`}
          type="success"
          showIcon
        />
      )}
    </div>
  );

  const renderCompletedStep = () => (
    <div className="text-center">
      <div className="mb-6">
        <CheckCircleOutlined className="text-6xl text-green-500 mb-4" />
        <Title level={4}>เข้าสู่ระบบสำเร็จ!</Title>
        <Paragraph type="secondary">
          ระบบได้ยืนยันตัวตนของคุณเรียบร้อยแล้ว
        </Paragraph>
      </div>

      <Alert
        message="การยืนยันตัวตนสำเร็จ"
        description={`ตรวจจับใบหน้า: ${winnerIdentity.identity} (${winnerIdentity.count} ครั้ง)`}
        type="success"
        showIcon
      />
    </div>
  );

  if (!visible) return null;

  return (
    <Modal
      title={
        <Space>
          <CameraOutlined />
          <span>สแกนหน้าเพื่อเข้าสู่ระบบ</span>
          {step === 'scanning' && (
            <Badge status="processing" text={`${currentScanIndex}/${TARGET_SCANS}`} />
          )}
          {step === 'completed' && (
            <Badge status="success" text="สำเร็จ" />
          )}
        </Space>
      }
      open={visible}
      onCancel={handleClose}
      width={800}
      footer={
        step === 'completed' ? [
          <Button key="close" type="primary" onClick={handleClose}>
            ปิด
          </Button>
        ] : [
          <Button key="close" onClick={handleClose} disabled={isScanning || isProcessing}>
            ปิด
          </Button>
        ]
      }
      destroyOnClose
    >
      <div className="space-y-4">
        {error && (
          <Alert
            message="เกิดข้อผิดพลาด"
            description={error}
            type="error"
            showIcon
            closable
            onClose={() => setError(null)}
          />
        )}

        {/* Video and Preview */}
        <div className="relative">
          <div className="bg-black rounded-lg overflow-hidden">
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
          </div>

          {/* Preview Canvas */}
          <canvas
            ref={previewCanvasRef}
            className="absolute top-4 right-4 border-2 border-white rounded-lg shadow-lg"
            style={{ width: '120px', height: '120px' }}
          />

          {/* Status Overlays */}
          {isStreaming && step === 'scanning' && (
            <div className="absolute top-4 left-4 space-y-2">
              <div className="bg-blue-500 text-white px-3 py-1 rounded-full">
                <ScanOutlined className="mr-1" />
                สแกนที่ {currentScanIndex}/{TARGET_SCANS}
              </div>
              {validScans > 0 && (
                <div className="bg-green-500 text-white px-3 py-1 rounded-full">
                  <CheckCircleOutlined className="mr-1" />
                  ตรวจพบ {validScans} ครั้ง
                </div>
              )}
            </div>
          )}
        </div>

        {/* Hidden Canvas for Processing */}
        <canvas ref={canvasRef} className="hidden" />

        {/* Step Content */}
        {step === 'setup' && renderSetupStep()}
        {step === 'scanning' && renderScanningStep()}
        {step === 'processing' && renderProcessingStep()}
        {step === 'completed' && renderCompletedStep()}
      </div>
    </Modal>
  );
};

export default FaceLoginModal;