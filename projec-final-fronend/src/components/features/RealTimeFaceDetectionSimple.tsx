import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Modal, Button, Space, Alert, Card, Typography, Badge } from 'antd';
import { 
  CameraOutlined, 
  StopOutlined, 
  EyeOutlined,
  LoadingOutlined,
  WarningOutlined
} from '@ant-design/icons';

const { Text } = Typography;

interface RealTimeFaceDetectionProps {
  visible: boolean;
  onClose: () => void;
}

interface DetectedFace {
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    confidence: number;
  };
}

const RealTimeFaceDetection: React.FC<RealTimeFaceDetectionProps> = ({ 
  visible, 
  onClose 
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [faces, setFaces] = useState<DetectedFace[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState<number>(0);
  const [detectionCount, setDetectionCount] = useState<number>(0);
  const [lastDetectionTime, setLastDetectionTime] = useState<number>(0);
  const [isDetectionInProgress, setIsDetectionInProgress] = useState<boolean>(false);

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
    setFaces([]);
    setFps(0);
    setDetectionCount(0);
    setLastDetectionTime(0);
    setIsDetectionInProgress(false);
  }, []);

  // ตรวจจับใบหน้าจาก video frame
  const detectFaces = useCallback(async () => {
    // ป้องกันการเรียกซ้อนกัน
    if (!videoRef.current || !canvasRef.current || !isStreaming || isDetectionInProgress) {
      return;
    }

    setIsDetectionInProgress(true);
    const detectionStartTime = performance.now();

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) {
      setIsDetectionInProgress(false);
      return;
    }

    // วาด video frame ลง canvas
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    try {
      // แปลง canvas เป็น blob เพื่อส่งไปยัง API
      canvas.toBlob(async (blob) => {
        if (!blob) {
          setIsDetectionInProgress(false);
          return;
        }

        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');
        formData.append('model_name', 'yolov11m');
        formData.append('conf_threshold', '0.3');
        formData.append('max_faces', '10');
        formData.append('min_quality_threshold', '20.0');

        try {
          const response = await fetch('http://localhost:8080/api/face-detection/detect-fast', {
            method: 'POST',
            body: formData
          });

          if (response.ok) {
            const result = await response.json();
            if (result.success && result.faces && result.faces.length > 0) {
              setFaces(result.faces);
            } else {
              setFaces([]);
            }
            
            setDetectionCount(prev => prev + 1);
            
            // คำนวณ FPS อย่างแม่นยำ
            const detectionEndTime = performance.now();
            setLastDetectionTime(prev => {
              if (prev > 0) {
                const timeDiff = (detectionEndTime - prev) / 1000; // วินาที
                
                // ป้องกันค่าผิดปกติ (น้อยกว่า 50ms หรือมากกว่า 5 วินาที)
                if (timeDiff > 0.05 && timeDiff < 5) {
                  const currentFps = 1 / timeDiff;
                  
                  // กรองค่า FPS ที่ผิดปกติ (น้อยกว่า 1 หรือมากกว่า 30)
                  if (currentFps >= 1 && currentFps <= 30) {
                    setFps(Number(currentFps.toFixed(1)));
                  }
                }
              }
              return detectionEndTime;
            });
            
          } else {
            console.error('Face detection failed:', response.status);
          }
        } catch (err) {
          console.error('Face detection error:', err);
        } finally {
          setIsDetectionInProgress(false);
        }
      }, 'image/jpeg', 0.8);
    } catch (err) {
      console.error('Canvas error:', err);
      setIsDetectionInProgress(false);
    }
  }, [isStreaming, isDetectionInProgress]);

  // เริ่ม/หยุด real-time detection
  const toggleDetection = useCallback(() => {
    if (isDetecting) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      setIsDetecting(false);
      setFaces([]);
      setFps(0);
      setDetectionCount(0);
      setLastDetectionTime(0);
      setIsDetectionInProgress(false);
    } else {
      setIsDetecting(true);
      setDetectionCount(0);
      setIsDetectionInProgress(false);
      
      // เริ่มต้นด้วย interval 150ms (6.7 FPS) สำหรับความเสถียร
      intervalRef.current = setInterval(() => {
        detectFaces();
      }, 150);
    }
  }, [isDetecting, detectFaces]);

  // วาดกรอบใบหน้าบน canvas
  const drawFaceBoxes = useCallback(() => {
    if (!canvasRef.current || !videoRef.current) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) return;

    // วาด video frame
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // วาดกรอบใบหน้า
    faces.forEach((face, index) => {
      const { x1, y1, x2, y2, confidence } = face.bbox;
      
      // กรอบสีเขียว
      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      
      // ข้อความ confidence
      ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
      ctx.fillRect(x1, y1 - 30, 120, 25);
      
      ctx.fillStyle = '#000000';
      ctx.font = 'bold 14px Arial';
      ctx.fillText(`Face ${index + 1}: ${(confidence * 100).toFixed(1)}%`, x1 + 5, y1 - 10);
    });
  }, [faces]);

  // อัพเดท canvas เมื่อมีการตรวจจับใบหน้า
  useEffect(() => {
    if (isDetecting) {
      const animationFrame = requestAnimationFrame(drawFaceBoxes);
      return () => cancelAnimationFrame(animationFrame);
    }
  }, [isDetecting, faces, drawFaceBoxes]);

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
          <CameraOutlined />
          Real-time Face Detection
          {isDetecting && <Badge status="processing" text="กำลังตรวจจับ..." />}
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

        {/* สถิติประสิทธิภาพ */}
        {isDetecting && (
          <Card size="small" title="สถิติการทำงาน">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Space>
                <Text strong>FPS:</Text>
                <Text style={{ color: fps > 10 ? '#52c41a' : fps > 5 ? '#faad14' : '#f5222d' }}>
                  {fps}
                </Text>
                <Text type="secondary">|</Text>
                <Text strong>ใบหน้าที่พบ:</Text>
                <Text>{faces.length}</Text>
                <Text type="secondary">|</Text>
                <Text strong>การตรวจจับทั้งหมด:</Text>
                <Text>{detectionCount}</Text>
              </Space>
              <Space>
                <Text strong>สถานะ:</Text>
                {isDetectionInProgress ? (
                  <Badge status="processing" text="กำลังประมวลผล..." />
                ) : (
                  <Badge status="success" text="พร้อม" />
                )}
              </Space>
            </Space>
          </Card>
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
              icon={isDetecting ? <StopOutlined /> : <EyeOutlined />}
              onClick={toggleDetection}
              disabled={!isStreaming}
              loading={isDetectionInProgress}
            >
              {isDetecting ? 'หยุดตรวจจับ' : 'เริ่มตรวจจับ'}
            </Button>

            {isDetecting && (
              <Badge 
                count={isDetectionInProgress ? <LoadingOutlined /> : faces.length} 
                showZero 
                color={faces.length > 0 ? '#52c41a' : '#d9d9d9'}
              >
                <Text type="secondary">สถานะการตรวจจับ</Text>
              </Badge>
            )}
          </Space>
        </Card>

        {/* Video และ Canvas */}
        <Card title="กล้องและการตรวจจับ" size="small">
          <div style={{ position: 'relative', display: 'inline-block' }}>
            <video
              ref={videoRef}
              autoPlay
              muted
              style={{
                width: '100%',
                maxWidth: '640px',
                height: 'auto',
                border: '2px solid #d9d9d9',
                borderRadius: '8px'
              }}
            />
            <canvas
              ref={canvasRef}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none'
              }}
            />
          </div>
        </Card>

        {/* คำแนะนำ */}
        <Alert
          message="คำแนะนำ"
          description="1. กดปุ่ม 'เริ่มกล้อง' เพื่อเปิดกล้อง 2. กดปุ่ม 'เริ่มตรวจจับ' เพื่อเริ่มการตรวจจับใบหน้าแบบ real-time"
          type="info"
          showIcon
        />
      </div>
    </Modal>
  );
};

export default RealTimeFaceDetection;
