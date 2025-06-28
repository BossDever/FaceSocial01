'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { 
  Card, 
  Button, 
  Typography, 
  Alert, 
  Space, 
  Row, 
  Col,
  Statistic,
  Tag,
  Avatar,
  List,
  Badge,
  Switch,
  Slider,
  Select,
  Progress,
  Tooltip
} from 'antd';
import { 
  PlayCircleOutlined,
  PauseCircleOutlined,
  CameraOutlined,
  UserOutlined,
  SafetyOutlined,
  EyeOutlined,
  SettingOutlined,
  AlertOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { Option } = Select;

interface DetectedFace {
  bbox: [number, number, number, number]; // x1, y1, x2, y2
  confidence: number;
  quality_score?: number;
}

interface FaceDetectionResult {
  faces: DetectedFace[];
  processing_time_ms: number;
  image_shape: [number, number, number];
  model_used: string;
  error?: string;
}

interface RecognitionResult {
  success: boolean;
  person_id?: string;
  confidence?: number;
  similarity?: number;
  is_registered?: boolean;
  user_info?: {
    id: string;
    username: string;
    fullName: string;
    email: string;
  };
  best_match?: {
    person_id: string;
    confidence: number;
    similarity: number;
  };
}

interface CroppedFace {
  id: string;
  imageData: string; // base64
  bbox: [number, number, number, number];
  confidence: number;
  quality_score: number;
  timestamp: number;
  recognition?: RecognitionResult;
  isProcessing?: boolean;
}

const RealTimeCCTV: React.FC = () => {
  // Core States
  const [isStreaming, setIsStreaming] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [isRecognitionEnabled, setIsRecognitionEnabled] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Performance States
  const [fps, setFps] = useState(0);
  const [detectionLatency, setDetectionLatency] = useState(0);
  const [recognitionLatency, setRecognitionLatency] = useState(0);
  const [totalDetections, setTotalDetections] = useState(0);
  const [totalRecognitions, setTotalRecognitions] = useState(0);
  
  // Detection Settings
  const [confidence, setConfidence] = useState(0.3);
  const [maxFaces, setMaxFaces] = useState(10);
  const [minQuality, setMinQuality] = useState(20);
  const [modelName, setModelName] = useState('yolov11m');
  
  // Face Data
  const [croppedFaces, setCroppedFaces] = useState<CroppedFace[]>([]);
  const [recognizedUsers, setRecognizedUsers] = useState<Set<string>>(new Set());
  
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const detectionIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const fpsIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const fpsCountRef = useRef(0);
  const isProcessingRef = useRef(false);

  // API URLs
  const apiUrl = process.env.NODE_ENV === 'production' 
    ? process.env.NEXT_PUBLIC_FACE_API_URL || 'http://localhost:8080'
    : 'http://localhost:8080';

  // Initialize Camera
  const initCamera = useCallback(async () => {
    try {
      setError(null);
      console.log('📷 Starting CCTV camera...');
      
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        
        videoRef.current.onloadedmetadata = () => {
          setIsStreaming(true);
          setupCanvas();
        };
      }
    } catch (err) {
      console.error('❌ Camera error:', err);
      setError('ไม่สามารถเข้าถึงกล้องได้ กรุณาอนุญาตการใช้งานกล้อง');
    }
  }, []);

  // Stop Camera
  const stopCamera = useCallback(() => {
    console.log('🛑 Stopping CCTV...');
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    
    if (fpsIntervalRef.current) {
      clearInterval(fpsIntervalRef.current);
      fpsIntervalRef.current = null;
    }
    
    setIsStreaming(false);
    setIsDetecting(false);
  }, []);

  // Setup Canvas
  const setupCanvas = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || !overlayCanvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const overlayCanvas = overlayCanvasRef.current;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    overlayCanvas.width = video.videoWidth;
    overlayCanvas.height = video.videoHeight;
  }, []);

  // Convert canvas to blob
  const canvasToBlob = useCallback((canvas: HTMLCanvasElement): Promise<Blob> => {
    return new Promise((resolve) => {
      canvas.toBlob((blob) => {
        resolve(blob!);
      }, 'image/jpeg', 0.8);
    });
  }, []);

  // Crop face from image
  const cropFace = useCallback((
    canvas: HTMLCanvasElement, 
    bbox: [number, number, number, number],
    padding: number = 20
  ): string => {
    const [x1, y1, x2, y2] = bbox;
    const width = x2 - x1;
    const height = y2 - y1;
    
    // Add padding
    const padX = Math.max(0, x1 - padding);
    const padY = Math.max(0, y1 - padding);
    const padWidth = Math.min(canvas.width - padX, width + padding * 2);
    const padHeight = Math.min(canvas.height - padY, height + padding * 2);
    
    // Create crop canvas
    const cropCanvas = document.createElement('canvas');
    const cropCtx = cropCanvas.getContext('2d')!;
    cropCanvas.width = padWidth;
    cropCanvas.height = padHeight;
    
    // Draw cropped region
    const ctx = canvas.getContext('2d')!;
    const imageData = ctx.getImageData(padX, padY, padWidth, padHeight);
    cropCtx.putImageData(imageData, 0, 0);
    
    return cropCanvas.toDataURL('image/jpeg', 0.8);
  }, []);

  // Perform Face Recognition
  const performFaceRecognition = useCallback(async (faceData: string): Promise<RecognitionResult | null> => {
    try {
      const recognitionStart = performance.now();
      
      const formData = new FormData();
      
      // Convert base64 to blob
      const response = await fetch(faceData);
      const blob = await response.blob();
      formData.append('image', blob, 'face.jpg');
      
      const apiResponse = await fetch(`${apiUrl}/api/face-recognition/recognize`, {
        method: 'POST',
        body: formData,
      });
      
      const recognitionEnd = performance.now();
      setRecognitionLatency(recognitionEnd - recognitionStart);
      
      if (!apiResponse.ok) {
        throw new Error(`HTTP error! status: ${apiResponse.status}`);
      }
      
      const result = await apiResponse.json();
      setTotalRecognitions(prev => prev + 1);
      
      return result;
    } catch (error) {
      console.error('❌ Recognition error:', error);
      return null;
    }
  }, [apiUrl]);

  // Perform Face Detection
  const performDetection = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !overlayCanvasRef.current || isProcessingRef.current) {
      return;
    }
    
    isProcessingRef.current = true;
    
    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const overlayCanvas = overlayCanvasRef.current;
      const ctx = canvas.getContext('2d')!;
      const overlayCtx = overlayCanvas.getContext('2d')!;
      
      if (video.readyState < 2) {
        isProcessingRef.current = false;
        return;
      }
      
      // Capture frame
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(video, 0, 0);
      
      // Clear overlay
      overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
      
      const detectionStart = performance.now();
      
      // Convert to blob and send for detection
      const blob = await canvasToBlob(canvas);
      
      const formData = new FormData();
      formData.append('image', blob, 'frame.jpg');
      formData.append('conf', confidence.toString());
      formData.append('max_faces', maxFaces.toString());
      formData.append('min_quality', minQuality.toString());
      formData.append('model_name', modelName);
      
      const response = await fetch(`${apiUrl}/api/face-detection/detect-fast`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result: FaceDetectionResult = await response.json();
      
      const detectionEnd = performance.now();
      setDetectionLatency(detectionEnd - detectionStart);
      setTotalDetections(prev => prev + 1);
      
      // Update FPS counter
      fpsCountRef.current += 1;
      
      if (result.faces && result.faces.length > 0) {
        // Draw bounding boxes
        overlayCtx.strokeStyle = '#00ff00';
        overlayCtx.lineWidth = 2;
        overlayCtx.font = '14px Arial';
        overlayCtx.fillStyle = '#00ff00';
        
        const newCroppedFaces: CroppedFace[] = [];
        
        for (const face of result.faces) {
          const [x1, y1, x2, y2] = face.bbox;
          
          // Draw bounding box
          overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          
          // Draw confidence
          const confText = `${(face.confidence * 100).toFixed(0)}%`;
          const qualityText = face.quality_score ? ` Q:${face.quality_score.toFixed(0)}` : '';
          const text = confText + qualityText;
          
          overlayCtx.fillText(text, x1, y1 - 5);
          
          // Crop face
          const croppedData = cropFace(canvas, face.bbox);
          
          const croppedFace: CroppedFace = {
            id: `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            imageData: croppedData,
            bbox: face.bbox,
            confidence: face.confidence,
            quality_score: face.quality_score || 0,
            timestamp: Date.now(),
            isProcessing: isRecognitionEnabled
          };
          
          newCroppedFaces.push(croppedFace);
          
          // Perform recognition if enabled
          if (isRecognitionEnabled) {
            performFaceRecognition(croppedData).then((recognition) => {
              if (recognition) {
                setCroppedFaces(prev => 
                  prev.map(cf => 
                    cf.id === croppedFace.id 
                      ? { ...cf, recognition, isProcessing: false }
                      : cf
                  )
                );
                
                // Add to recognized users set
                if (recognition.success && recognition.user_info) {
                  setRecognizedUsers(prev => new Set(prev).add(recognition.user_info!.username));
                }
              } else {
                setCroppedFaces(prev => 
                  prev.map(cf => 
                    cf.id === croppedFace.id 
                      ? { ...cf, isProcessing: false }
                      : cf
                  )
                );
              }
            });
          }
        }
        
        // Add new cropped faces (keep last 20)
        setCroppedFaces(prev => {
          const updated = [...newCroppedFaces, ...prev];
          return updated.slice(0, 20);
        });
      }
      
    } catch (error) {
      console.error('❌ Detection error:', error);
    } finally {
      isProcessingRef.current = false;
    }
  }, [confidence, maxFaces, minQuality, modelName, apiUrl, isRecognitionEnabled, canvasToBlob, cropFace, performFaceRecognition]);

  // Start Detection
  const startDetection = useCallback(() => {
    if (detectionIntervalRef.current) return;
    
    setIsDetecting(true);
    
    // Start detection loop
    detectionIntervalRef.current = setInterval(performDetection, 200); // 5 FPS detection
    
    // Start FPS counter
    fpsCountRef.current = 0;
    fpsIntervalRef.current = setInterval(() => {
      setFps(fpsCountRef.current * 2); // Update every 500ms, so multiply by 2
      fpsCountRef.current = 0;
    }, 500);
  }, [performDetection]);

  // Stop Detection
  const stopDetection = useCallback(() => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    
    if (fpsIntervalRef.current) {
      clearInterval(fpsIntervalRef.current);
      fpsIntervalRef.current = null;
    }
    
    setIsDetecting(false);
    setFps(0);
  }, []);

  // Handle start/stop
  const handleStart = async () => {
    if (!isStreaming) {
      await initCamera();
    } else if (!isDetecting) {
      startDetection();
    } else {
      stopDetection();
    }
  };

  const handleStop = () => {
    stopCamera();
    setCroppedFaces([]);
    setRecognizedUsers(new Set());
    setTotalDetections(0);
    setTotalRecognitions(0);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  return (
    <div style={{ padding: '20px' }}>
      <Title level={2}>
        <EyeOutlined /> Real-time CCTV Face Detection & Recognition
      </Title>
      
      {error && (
        <Alert 
          message="เกิดข้อผิดพลาด" 
          description={error} 
          type="error" 
          closable 
          onClose={() => setError(null)}
          style={{ marginBottom: 16 }}
        />
      )}

      <Row gutter={[16, 16]}>
        {/* Main Video Feed */}
        <Col xs={24} lg={16}>
          <Card 
            title="กล้อง CCTV"
            extra={
              <Space>
                <Badge 
                  status={isStreaming ? "processing" : "default"} 
                  text={isStreaming ? "เชื่อมต่อแล้ว" : "ไม่ได้เชื่อมต่อ"} 
                />
                <Badge 
                  status={isDetecting ? "processing" : "default"} 
                  text={isDetecting ? "กำลังตรวจจับ" : "หยุดการตรวจจับ"} 
                />
              </Space>
            }
          >
            <div style={{ position: 'relative', textAlign: 'center' }}>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{
                  width: '100%',
                  maxWidth: '800px',
                  height: 'auto',
                  backgroundColor: '#000',
                  display: isStreaming ? 'block' : 'none'
                }}
              />
              
              {/* Overlay Canvas for Bounding Boxes */}
              <canvas
                ref={overlayCanvasRef}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: '50%',
                  transform: 'translateX(-50%)',
                  width: '100%',
                  maxWidth: '800px',
                  height: 'auto',
                  pointerEvents: 'none',
                  display: isStreaming ? 'block' : 'none'
                }}
              />
              
              {/* Hidden Canvas for Processing */}
              <canvas ref={canvasRef} style={{ display: 'none' }} />
              
              {!isStreaming && (
                <div style={{ 
                  padding: '100px 20px', 
                  backgroundColor: '#f5f5f5',
                  borderRadius: '8px',
                  color: '#666'
                }}>
                  <CameraOutlined style={{ fontSize: '48px', marginBottom: '16px' }} />
                  <div>กดปุ่มเริ่มต้นเพื่อเปิดกล้อง CCTV</div>
                </div>
              )}
            </div>
            
            <Space style={{ marginTop: 16, width: '100%', justifyContent: 'center' }}>
              <Button
                type="primary"
                icon={isDetecting ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
                onClick={handleStart}
                loading={!isStreaming && isDetecting}
              >
                {!isStreaming ? 'เริ่มต้น CCTV' : isDetecting ? 'หยุดการตรวจจับ' : 'เริ่มการตรวจจับ'}
              </Button>
              
              <Button onClick={handleStop} disabled={!isStreaming}>
                หยุด CCTV
              </Button>
            </Space>
          </Card>
        </Col>

        {/* Performance Stats */}
        <Col xs={24} lg={8}>
          <Card title="สถิติประสิทธิภาพ" size="small">
            <Row gutter={16}>
              <Col span={12}>
                <Statistic
                  title="FPS"
                  value={fps}
                  suffix=""
                  valueStyle={{ color: fps > 5 ? '#3f8600' : '#cf1322' }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="Detection (ms)"
                  value={detectionLatency.toFixed(0)}
                  suffix=""
                  valueStyle={{ color: detectionLatency < 100 ? '#3f8600' : '#cf1322' }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="Recognition (ms)"
                  value={recognitionLatency.toFixed(0)}
                  suffix=""
                  valueStyle={{ color: recognitionLatency < 200 ? '#3f8600' : '#cf1322' }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="ตรวจจับทั้งหมด"
                  value={totalDetections}
                  suffix=""
                />
              </Col>
            </Row>
          </Card>

          {/* Settings */}
          <Card title="การตั้งค่า" size="small" style={{ marginTop: 16 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text>โมเดล:</Text>
                <Select
                  value={modelName}
                  onChange={setModelName}
                  style={{ width: '100%', marginTop: 4 }}
                  disabled={isDetecting}
                >
                  <Option value="yolov11m">YOLOv11m (เร็ว)</Option>
                  <Option value="yolov11n">YOLOv11n (เล็ก)</Option>
                </Select>
              </div>
              
              <div>
                <Text>Confidence: {confidence}</Text>
                <Slider
                  min={0.1}
                  max={0.9}
                  step={0.1}
                  value={confidence}
                  onChange={setConfidence}
                  disabled={isDetecting}
                />
              </div>
              
              <div>
                <Text>จำนวนใบหน้าสูงสุด: {maxFaces}</Text>
                <Slider
                  min={1}
                  max={20}
                  step={1}
                  value={maxFaces}
                  onChange={setMaxFaces}
                  disabled={isDetecting}
                />
              </div>
              
              <div>
                <Text>คุณภาพขั้นต่ำ: {minQuality}</Text>
                <Slider
                  min={0}
                  max={100}
                  step={10}
                  value={minQuality}
                  onChange={setMinQuality}
                  disabled={isDetecting}
                />
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text>Face Recognition:</Text>
                <Switch
                  checked={isRecognitionEnabled}
                  onChange={setIsRecognitionEnabled}
                  checkedChildren="เปิด"
                  unCheckedChildren="ปิด"
                />
              </div>
            </Space>
          </Card>

          {/* Recognized Users */}
          {recognizedUsers.size > 0 && (
            <Card title="ผู้ใช้ที่จดจำได้" size="small" style={{ marginTop: 16 }}>
              <Space wrap>
                {Array.from(recognizedUsers).map(username => (
                  <Tag key={username} color="green" icon={<UserOutlined />}>
                    {username}
                  </Tag>
                ))}
              </Space>
            </Card>
          )}
        </Col>

        {/* Cropped Faces */}
        <Col xs={24}>
          <Card title="ใบหน้าที่ตรวจจับได้" size="small">
            {croppedFaces.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px', color: '#666' }}>
                ยังไม่มีการตรวจจับใบหน้า
              </div>
            ) : (
              <div style={{ 
                display: 'flex', 
                flexWrap: 'wrap', 
                gap: '8px', 
                maxHeight: '300px', 
                overflowY: 'auto' 
              }}>
                {croppedFaces.map((face) => (
                  <div
                    key={face.id}
                    style={{
                      position: 'relative',
                      border: '2px solid #d9d9d9',
                      borderRadius: '8px',
                      overflow: 'hidden',
                      width: '120px',
                      backgroundColor: '#fff'
                    }}
                  >
                    <img
                      src={face.imageData}
                      alt="Detected Face"
                      style={{
                        width: '100%',
                        height: '120px',
                        objectFit: 'cover'
                      }}
                    />
                    
                    <div style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      right: 0,
                      background: 'rgba(0,0,0,0.7)',
                      color: 'white',
                      padding: '2px 4px',
                      fontSize: '10px'
                    }}>
                      {(face.confidence * 100).toFixed(0)}% Q:{face.quality_score.toFixed(0)}
                    </div>
                    
                    {face.isProcessing && (
                      <div style={{
                        position: 'absolute',
                        bottom: 0,
                        left: 0,
                        right: 0,
                        background: 'rgba(24, 144, 255, 0.8)',
                        color: 'white',
                        padding: '2px 4px',
                        fontSize: '10px',
                        textAlign: 'center'
                      }}>
                        <Progress percent={50} showInfo={false} size="small" />
                        กำลังจดจำ...
                      </div>
                    )}
                    
                    {face.recognition && (
                      <div style={{
                        position: 'absolute',
                        bottom: 0,
                        left: 0,
                        right: 0,
                        background: face.recognition.success 
                          ? 'rgba(82, 196, 26, 0.9)' 
                          : 'rgba(245, 34, 45, 0.9)',
                        color: 'white',
                        padding: '2px 4px',
                        fontSize: '10px',
                        textAlign: 'center'
                      }}>
                        {face.recognition.success ? (
                          <>
                            <CheckCircleOutlined /> {face.recognition.user_info?.username || 'จดจำได้'}
                          </>
                        ) : (
                          <>
                            <AlertOutlined /> ไม่จดจำ
                          </>
                        )}
                      </div>
                    )}
                    
                    <div style={{
                      position: 'absolute',
                      top: '20px',
                      right: '4px',
                      background: 'rgba(0,0,0,0.7)',
                      color: 'white',
                      padding: '1px 3px',
                      fontSize: '9px',
                      borderRadius: '2px'
                    }}>
                      <ClockCircleOutlined /> {new Date(face.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default RealTimeCCTV;
