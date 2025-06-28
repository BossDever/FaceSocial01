import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Modal, Button, Space, Alert, Card, Typography, Badge } from 'antd';
import { 
  CameraOutlined, 
  StopOutlined, 
  SafetyOutlined,
  LoadingOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined
} from '@ant-design/icons';

const { Text } = Typography;

interface RealTimeAntiSpoofingProps {
  visible: boolean;
  onClose: () => void;
}

interface SpoofingResult {
  face_id: number;
  is_real: boolean;
  confidence: number;
  spoofing_detected: boolean;
  region: {
    x: number;
    y: number;
    w: number;
    h: number;
  };
}

const RealTimeAntiSpoofing: React.FC<RealTimeAntiSpoofingProps> = ({ 
  visible, 
  onClose 
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [results, setResults] = useState<SpoofingResult[]>([]);
  const [overallResult, setOverallResult] = useState<{
    is_real: boolean;
    confidence: number;
    spoofing_detected: boolean;
    real_faces: number;
    fake_faces: number;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const [detectionCount, setDetectionCount] = useState(0);

  // เริ่มกล้อง
  const startCamera = useCallback(async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        }
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsStreaming(true);
      }
    } catch (err) {
      setError('ไม่สามารถเข้าถึงกล้องได้ กรุณาตรวจสอบการอนุญาต');
      console.error('Camera access error:', err);
    }
  }, []);

  // หยุดกล้อง
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsStreaming(false);
    setIsDetecting(false);
    setResults([]);
    setOverallResult(null);
  }, []);
  // ฟังก์ชันตัดใบหน้าออกจาก Canvas
  const cropFaceFromCanvas = useCallback(async (sourceCanvas: HTMLCanvasElement, bbox: { x1: number; y1: number; x2: number; y2: number }): Promise<Blob> => {
    return new Promise((resolve, reject) => {
      const tempCanvas = document.createElement('canvas');
      const tempCtx = tempCanvas.getContext('2d');
      
      if (!tempCtx) {
        reject(new Error('Cannot create canvas context'));
        return;
      }

      // คำนวณขนาดที่จะตัด (เพิ่ม padding รอบใบหน้า)
      const padding = 0.2; // เพิ่ม 20% รอบใบหน้า
      const faceWidth = bbox.x2 - bbox.x1;
      const faceHeight = bbox.y2 - bbox.y1;
      const paddingX = faceWidth * padding;
      const paddingY = faceHeight * padding;
      
      const cropX = Math.max(0, bbox.x1 - paddingX);
      const cropY = Math.max(0, bbox.y1 - paddingY);
      const cropWidth = Math.min(sourceCanvas.width - cropX, faceWidth + (paddingX * 2));
      const cropHeight = Math.min(sourceCanvas.height - cropY, faceHeight + (paddingY * 2));
      
      // ตั้งค่า Canvas
      tempCanvas.width = cropWidth;
      tempCanvas.height = cropHeight;
      
      // วาดส่วนที่ตัดมาลงใน Canvas
      tempCtx.drawImage(
        sourceCanvas,
        cropX, cropY, cropWidth, cropHeight, // source
        0, 0, cropWidth, cropHeight // destination
      );
      
      // แปลง Canvas เป็น Blob
      tempCanvas.toBlob((blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error('Failed to create cropped image blob'));
        }
      }, 'image/jpeg', 0.9);
    });
  }, []);

  // ตรวจจับการปลอมแปลงจาก video frame (ใช้ Face Detection ก่อน)
  const detectSpoofing = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !isStreaming) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) return;

    // วาด video frame ลง canvas
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    try {
      canvas.toBlob(async (blob) => {
        if (!blob) return;

        try {
          // Step 1: รัน Face Detection ก่อน
          const faceFormData = new FormData();
          faceFormData.append('file', blob, 'frame.jpg');
          faceFormData.append('model_name', 'yolov11m');
          faceFormData.append('conf_threshold', '0.5');

          const faceResponse = await fetch('http://localhost:8080/api/face-detection/detect', {
            method: 'POST',
            body: faceFormData
          });

          if (!faceResponse.ok) {
            console.error('Face detection failed');
            return;
          }

          const faceResult = await faceResponse.json();
          const faces = faceResult.faces || [];

          if (faces.length === 0) {
            setResults([]);
            setOverallResult(null);
            return;
          }

          console.log(`🎯 พบใบหน้า ${faces.length} หน้า (Real-time)`);

          // Step 2: ประมวลผล Anti-Spoofing สำหรับใบหน้าแต่ละหน้า
          const antispoofingResults = [];

          for (let i = 0; i < faces.length; i++) {
            const face = faces[i];
            
            try {
              // ตัดใบหน้าออกจาก canvas
              const croppedBlob = await cropFaceFromCanvas(canvas, face.bbox);
              
              // ส่งใบหน้าที่ตัดแล้วไปตรวจสอบ Anti-Spoofing
              const spoofFormData = new FormData();
              spoofFormData.append('image', croppedBlob, `face_${i}.jpg`);
              spoofFormData.append('confidence_threshold', '0.5');

              const spoofResponse = await fetch('http://localhost:8080/api/anti-spoofing/detect-upload', {
                method: 'POST',
                body: spoofFormData
              });

              if (spoofResponse.ok) {
                const spoofResult = await spoofResponse.json();
                
                // รวมข้อมูลจาก Face Detection และ Anti-Spoofing
                const combinedResult = {
                  face_id: i + 1,
                  is_real: spoofResult.overall_result?.is_real || false,
                  confidence: spoofResult.overall_result?.confidence || 0,
                  spoofing_detected: spoofResult.overall_result?.spoofing_detected || false,
                  region: {
                    x: face.bbox.x1,
                    y: face.bbox.y1,
                    w: face.bbox.x2 - face.bbox.x1,
                    h: face.bbox.y2 - face.bbox.y1
                  }
                };

                antispoofingResults.push(combinedResult);
              }
            } catch (error) {
              console.error(`❌ เกิดข้อผิดพลาดกับใบหน้าที่ ${i + 1}:`, error);
            }
          }

          if (antispoofingResults.length > 0) {
            setResults(antispoofingResults);
            
            // สร้าง overall result
            const realFaces = antispoofingResults.filter(r => r.is_real).length;
            const fakeFaces = antispoofingResults.length - realFaces;
            
            setOverallResult({
              is_real: realFaces > fakeFaces,
              confidence: antispoofingResults.reduce((sum, r) => sum + r.confidence, 0) / antispoofingResults.length,
              spoofing_detected: fakeFaces > 0,
              real_faces: realFaces,
              fake_faces: fakeFaces
            });
            
            setDetectionCount(prev => prev + 1);
          } else {
            setResults([]);
            setOverallResult(null);
          }

        } catch (error) {
          console.error('Real-time detection error:', error);
        }
      }, 'image/jpeg', 0.8);
    } catch (error) {
      console.error('Canvas processing error:', error);
    }
  }, [isStreaming, cropFaceFromCanvas]);
  // เริ่ม/หยุด real-time detection
  const toggleDetection = useCallback(() => {
    if (isDetecting) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      setIsDetecting(false);
      setResults([]);
      setOverallResult(null);
      setFps(0);
    } else {
      setIsDetecting(true);
      
      // คำนวณ FPS
      let frameCount = 0;
      const startTime = Date.now();
      
      // ตรวจจับทุก 500ms (2 FPS) เพราะ anti-spoofing ใช้เวลานานกว่า
      intervalRef.current = setInterval(() => {
        detectSpoofing();
        frameCount++;
        
        // อัพเดท FPS ทุก 2 วินาที
        const elapsed = (Date.now() - startTime) / 1000;
        if (elapsed >= 2) {
          setFps(Math.round(frameCount / elapsed * 10) / 10); // ปัดเศษ 1 ตำแหน่ง
        }
      }, 500);
    }
  }, [isDetecting, detectSpoofing]);// วาดผลลัพธ์บน canvas (ใช้พิกัดจาก Face Detection)
  const drawResults = useCallback(() => {
    if (!canvasRef.current || !videoRef.current) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) return;

    // วาด video frame
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // วาดผลลัพธ์แต่ละใบหน้า (ใช้พิกัดจาก Face Detection)
    results.forEach((result) => {
      const { x, y, w, h } = result.region;
      const isReal = result.is_real;
      const confidence = result.confidence;
      
      // กรอบสี - เขียวถ้าจริง, แดงถ้าปลอม
      ctx.strokeStyle = isReal ? '#00ff00' : '#ff0000';
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, w, h);
      
      // พื้นหลังข้อความ
      const bgColor = isReal ? 'rgba(0, 255, 0, 0.9)' : 'rgba(255, 0, 0, 0.9)';
      ctx.fillStyle = bgColor;
      const textY = y > 40 ? y - 40 : y + h + 5;
      ctx.fillRect(x, textY, 220, 35);
      
      // ข้อความ
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 16px Arial';
      const status = isReal ? 'REAL FACE ✓' : 'FAKE/SPOOF ⚠️';
      ctx.fillText(`${status}`, x + 5, textY + 22);
      
      ctx.font = 'bold 12px Arial';
      ctx.fillText(`${(confidence * 100).toFixed(1)}%`, x + 5, textY + 35);
    });

    // แสดงผลรวม
    if (overallResult) {
      const status = overallResult.is_real ? 'REAL PERSON' : 'SPOOFING DETECTED';
      const color = overallResult.is_real ? 'rgba(0, 255, 0, 0.9)' : 'rgba(255, 0, 0, 0.9)';
      
      ctx.fillStyle = color;
      ctx.fillRect(10, 10, 250, 40);
      
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 16px Arial';
      ctx.fillText(status, 15, 35);
    }
  }, [results, overallResult]);

  // อัพเดท canvas เมื่อมีผลลัพธ์
  useEffect(() => {
    if (isDetecting) {
      const animationFrame = requestAnimationFrame(drawResults);
      return () => cancelAnimationFrame(animationFrame);
    }
  }, [isDetecting, results, overallResult, drawResults]);

  // เริ่มกล้องเมื่อเปิด modal
  useEffect(() => {
    if (visible) {
      startCamera();
    } else {
      stopCamera();
    }

    return () => {
      stopCamera();
    };
  }, [visible, startCamera, stopCamera]);

  return (
    <Modal
      title={
        <Space>
          <SafetyOutlined />
          Real-time Anti-Spoofing Detection
          {isDetecting && <Badge status="processing" text="กำลังตรวจสอบ..." />}
        </Space>
      }
      open={visible}
      onCancel={onClose}
      width={800}
      footer={[
        <Button key="close" onClick={onClose}>
          ปิด
        </Button>
      ]}
      destroyOnHidden
    >
      <div className="space-y-4">
        {error && (
          <Alert
            message="ข้อผิดพลาด"
            description={error}
            type="error"
            icon={<WarningOutlined />}
            showIcon
          />
        )}

        {/* ส่วนควบคุม */}
        <Card size="small">
          <Space wrap>
            <Button
              type={isStreaming ? "default" : "primary"}
              icon={<CameraOutlined />}
              onClick={startCamera}
              disabled={isStreaming}
            >
              เริ่มกล้อง
            </Button>

            <Button
              type="default"
              icon={<StopOutlined />}
              onClick={stopCamera}
              disabled={!isStreaming}
            >
              หยุดกล้อง
            </Button>

            <Button
              type={isDetecting ? "default" : "primary"}
              danger={isDetecting}
              icon={isDetecting ? <LoadingOutlined spin /> : <SafetyOutlined />}
              onClick={toggleDetection}
              disabled={!isStreaming}
            >
              {isDetecting ? 'หยุดตรวจสอบ' : 'เริ่มตรวจสอบการปลอมแปลง'}
            </Button>
          </Space>
        </Card>

        {/* สถิติและผลลัพธ์ */}
        {isDetecting && (
          <Card size="small">
            <Space split={<span>|</span>} wrap>
              <Text>
                <strong>FPS:</strong> {fps}
              </Text>
              <Text>
                <strong>ใบหน้าที่ตรวจพบ:</strong> {results.length}
              </Text>
              <Text>
                <strong>การตรวจสอบทั้งหมด:</strong> {detectionCount}
              </Text>
              {overallResult && (
                <Text style={{ color: overallResult.is_real ? '#52c41a' : '#ff4d4f' }}>
                  <strong>
                    {overallResult.is_real ? 
                      <><CheckCircleOutlined /> REAL PERSON</> : 
                      <><CloseCircleOutlined /> SPOOFING DETECTED</>
                    }
                  </strong>
                </Text>
              )}
            </Space>
          </Card>
        )}

        {/* Video และ Canvas */}
        <div className="relative bg-black rounded-lg overflow-hidden">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-auto"
            style={{ maxHeight: '400px' }}
          />
          
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 w-full h-full"
            style={{ 
              display: isDetecting ? 'block' : 'none',
              maxHeight: '400px'
            }}
          />
        </div>

        {/* คำแนะนำ */}
        <Alert
          message="คำแนะนำ"
          description="1. กดปุ่ม 'เริ่มกล้อง' เพื่อเปิดกล้อง 2. กดปุ่ม 'เริ่มตรวจสอบการปลอมแปลง' เพื่อตรวจสอบว่าเป็นใบหน้าจริงหรือภาพปลอม"
          type="info"
          showIcon
        />
      </div>
    </Modal>
  );
};

export default RealTimeAntiSpoofing;
