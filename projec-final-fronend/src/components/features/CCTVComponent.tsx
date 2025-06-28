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
  Badge,
  Switch,
  Slider,
  Select,
  Progress
} from 'antd';
import { 
  PlayCircleOutlined,
  PauseCircleOutlined,
  CameraOutlined,
  UserOutlined,
  EyeOutlined,
  SettingOutlined,
  AlertOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { Option } = Select;

interface DetectedFace {
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    confidence: number;
  };
  confidence?: number;
  quality_score?: number;
}

interface CroppedFace {
  id: string;
  imageData: string;
  bbox: {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    confidence: number;
  };
  confidence: number;
  quality_score: number;
  timestamp: number;
  recognition?: any;
  isProcessing?: boolean;
}

const CCTVComponent: React.FC = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [isRecognitionEnabled, setIsRecognitionEnabled] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const [detectionLatency, setDetectionLatency] = useState(0);
  const [totalDetections, setTotalDetections] = useState(0);
  const [confidence, setConfidence] = useState(0.3);
  const [maxFaces, setMaxFaces] = useState(10);
  const [modelName, setModelName] = useState('yolov11m');
  const [croppedFaces, setCroppedFaces] = useState<CroppedFace[]>([]);
  const [recognizedUsers, setRecognizedUsers] = useState<Set<string>>(new Set());

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const detectionIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const fpsIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const isProcessingRef = useRef(false);

  const apiUrl = 'http://localhost:8080';

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
  const canvasToBlob = useCallback((canvas: HTMLCanvasElement): Promise<Blob> => {
    return new Promise((resolve) => {
      canvas.toBlob((blob) => {
        resolve(blob!);
      }, 'image/jpeg', 0.8);
    });
  }, []);
  const cropFace = useCallback((
    canvas: HTMLCanvasElement, 
    bbox: {
      x1: number;
      y1: number;
      x2: number;
      y2: number;
      confidence: number;
    },
    padding: number = 20
  ): string => {
    const { x1, y1, x2, y2 } = bbox;
    const width = x2 - x1;
    const height = y2 - y1;
    
    const padX = Math.max(0, x1 - padding);
    const padY = Math.max(0, y1 - padding);
    const padWidth = Math.min(canvas.width - padX, width + padding * 2);
    const padHeight = Math.min(canvas.height - padY, height + padding * 2);
    
    const cropCanvas = document.createElement('canvas');
    const cropCtx = cropCanvas.getContext('2d')!;
    cropCanvas.width = padWidth;
    cropCanvas.height = padHeight;
    
    const ctx = canvas.getContext('2d')!;
    const imageData = ctx.getImageData(padX, padY, padWidth, padHeight);
    cropCtx.putImageData(imageData, 0, 0);
    
    return cropCanvas.toDataURL('image/jpeg', 0.8);
  }, []);

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
      
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(video, 0, 0);
      
      overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
      
      const detectionStart = performance.now();
      
      const blob = await canvasToBlob(canvas);
        const formData = new FormData();
      formData.append('file', blob, 'frame.jpg');
      formData.append('conf_threshold', confidence.toString());
      formData.append('max_faces', maxFaces.toString());
      formData.append('min_quality_threshold', '20');
      formData.append('model_name', modelName);
      
      const response = await fetch(`${apiUrl}/api/face-detection/detect-fast`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      
      const detectionEnd = performance.now();
      setDetectionLatency(detectionEnd - detectionStart);
      setTotalDetections(prev => prev + 1);
        if (result.success && result.faces && result.faces.length > 0) {
        overlayCtx.strokeStyle = '#00ff00';
        overlayCtx.lineWidth = 2;
        overlayCtx.font = '14px Arial';
        overlayCtx.fillStyle = '#00ff00';
        
        const newCroppedFaces: CroppedFace[] = [];
        
        for (const face of result.faces) {
          const { x1, y1, x2, y2, confidence } = face.bbox;
          
          overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            const confText = `${((confidence || 0) * 100).toFixed(0)}%`;
          const qualityText = face.quality_score ? ` Q:${face.quality_score.toFixed(0)}` : '';
          const text = confText + qualityText;
          
          overlayCtx.fillText(text, x1, y1 - 5);
          
          const croppedData = cropFace(canvas, face.bbox);
            const croppedFace: CroppedFace = {
            id: `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            imageData: croppedData,
            bbox: face.bbox,
            confidence: confidence || 0,
            quality_score: face.quality_score || 0,
            timestamp: Date.now(),
            isProcessing: isRecognitionEnabled
          };
          
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
                );                  // Add to recognized users set
                if (recognition.success && recognition.user_info && recognition.user_info.username) {
                  setRecognizedUsers(prev => new Set(prev).add(recognition.user_info.username));
                } else if (recognition.success && recognition.best_match && recognition.best_match.person_name) {
                  // Fallback: use person_name from face API
                  setRecognizedUsers(prev => new Set(prev).add(recognition.best_match.person_name));
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
          
          newCroppedFaces.push(croppedFace);
        }
        
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
  }, [confidence, maxFaces, modelName, apiUrl, isRecognitionEnabled, canvasToBlob, cropFace]);

  const startDetection = useCallback(() => {
    if (detectionIntervalRef.current) return;
    
    setIsDetecting(true);
    
    detectionIntervalRef.current = setInterval(performDetection, 200);
    
    let fpsCount = 0;
    fpsIntervalRef.current = setInterval(() => {
      setFps(fpsCount * 2);
      fpsCount = 0;
    }, 500);
    
    const fpsTicker = setInterval(() => {
      fpsCount += 1;
    }, 200);
    
    setTimeout(() => clearInterval(fpsTicker), 30000);
  }, [performDetection]);

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
  };
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

  // Get user info from database by person_id (UUID)
  const getUserInfo = useCallback(async (personId: string) => {
    try {
      console.log('🔍 Getting user info for person_id:', personId);
      
      // Method 1: Try Face API first
      const faceApiResponse = await fetch(`${apiUrl}/api/face-recognition/person/${personId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (faceApiResponse.ok) {
        const faceData = await faceApiResponse.json();
        console.log('👤 Face API user data:', faceData);
        
        if (faceData.success && faceData.person_name) {
          // Method 2: Get full user data from our database using the person_name (username)
          const token = localStorage.getItem('token');
          const dbResponse = await fetch(`/api/friends/search-users?q=${faceData.person_name}`, {
            headers: {
              'Authorization': `Bearer ${token}`
            }
          });

          if (dbResponse.ok) {
            const dbResult = await dbResponse.json();
            if (dbResult.success && dbResult.data.length > 0) {
              // Find exact username match
              const user = dbResult.data.find((u: any) => u.username === faceData.person_name);
              console.log('✅ Found user in database:', user);
              return user || dbResult.data[0];
            }
          }

          // Fallback: Create user object from Face API data
          return {
            id: personId,
            username: faceData.person_name,
            fullName: faceData.person_name,
            firstName: faceData.person_name?.split(' ')[0] || '',
            lastName: faceData.person_name?.split(' ').slice(1).join(' ') || '',
            avatarUrl: null
          };
        }
      }

      console.log('❌ Could not find user info for person_id:', personId);
      return null;
    } catch (error) {
      console.error('Error fetching user info:', error);
      return null;
    }
  }, [apiUrl]);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);
  // Perform Face Recognition
  const performFaceRecognition = useCallback(async (faceDataUrl: string): Promise<any> => {
    try {
      // Convert data URL to base64
      const base64Data = faceDataUrl.split(',')[1];
      
      const response = await fetch(`${apiUrl}/api/face-recognition/recognize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          face_image_base64: base64Data,
          model_name: 'facenet',
          top_k: 5,
          similarity_threshold: 0.5
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
        // If recognition successful, get user info
      if (result.success && result.best_match && result.best_match.person_id) {
        const userInfo = await getUserInfo(result.best_match.person_id);
        if (userInfo) {
          return {
            ...result,
            user_info: {
              id: userInfo.id,
              username: userInfo.username,
              fullName: userInfo.fullName,
              email: userInfo.email
            }
          };
        } else {
          // If getUserInfo fails, still return result with best_match info
          return result;
        }
      }
      
      return result;
    } catch (error) {
      console.error('❌ Recognition error:', error);
      return null;
    }
  }, [apiUrl, getUserInfo]);

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
                  title="ตรวจจับทั้งหมด"
                  value={totalDetections}
                  suffix=""
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="จดจำได้"
                  value={recognizedUsers.size}
                  suffix=""
                />
              </Col>
            </Row>
          </Card>

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
                    />                    <div style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      right: 0,
                      background: 'rgba(0,0,0,0.7)',
                      color: 'white',
                      padding: '2px 4px',
                      fontSize: '10px'
                    }}>
                      {(isNaN(face.confidence || 0) ? 0 : ((face.confidence || 0) * 100)).toFixed(0)}% Q:{(isNaN(face.quality_score || 0) ? 0 : (face.quality_score || 0)).toFixed(0)}
                    </div>
                    
                    {/* แสดงข้อมูลการจดจำ */}
                    {face.recognition && (
                      <div style={{
                        position: 'absolute',
                        bottom: 0,
                        left: 0,
                        right: 0,
                        background: face.recognition.success ? 'rgba(0,128,0,0.8)' : 'rgba(255,0,0,0.8)',
                        color: 'white',
                        padding: '2px 4px',
                        fontSize: '9px',
                        textAlign: 'center'
                      }}>                        {face.recognition.success && face.recognition.user_info 
                          ? `${face.recognition.user_info.username}` 
                          : face.recognition.success && face.recognition.best_match && face.recognition.best_match.person_name
                          ? `${face.recognition.best_match.person_name}`
                          : face.recognition.success && face.recognition.best_match
                          ? `ID: ${face.recognition.best_match.person_id.substring(0, 8)}...`
                          : 'Unknown'
                        }
                        {face.recognition.success && face.recognition.best_match && (
                          <div style={{ fontSize: '8px', opacity: 0.8 }}>
                            Sim: {(face.recognition.best_match.similarity * 100).toFixed(0)}%
                          </div>
                        )}
                      </div>
                    )}
                    
                    {/* แสดงสถานะการประมวลผล */}
                    {face.isProcessing && (
                      <div style={{
                        position: 'absolute',
                        bottom: 0,
                        left: 0,
                        right: 0,
                        background: 'rgba(255,165,0,0.8)',
                        color: 'white',
                        padding: '2px 4px',
                        fontSize: '9px',
                        textAlign: 'center'
                      }}>
                        Processing...
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

export default CCTVComponent;
