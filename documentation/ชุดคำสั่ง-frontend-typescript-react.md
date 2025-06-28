# ชุดคำสั่ง: Frontend TypeScript React
## การพัฒนา Frontend ด้วย TypeScript และ React

### 📋 สารบัญ
2.1 [ภาพรวมระบบ Frontend](#21-ภาพรวมระบบ-frontend)
2.2 [Face Authentication System](#22-face-authentication-system)
2.3 [Real-time Face Detection](#23-real-time-face-detection)
2.4 [Social Media Features](#24-social-media-features)
2.5 [Custom Hooks และ APIs](#25-custom-hooks-และ-apis)
2.6 [UI Components และ Design](#26-ui-components-และ-design)
2.7 [State Management](#27-state-management)
2.8 [Performance Optimization](#28-performance-optimization)

---

## 2.1 ภาพรวมระบบ Frontend

ระบบ Frontend ที่พัฒนาด้วย TypeScript, React และ Next.js รองรับการใช้งาน Face Recognition และ Social Media Platform แบบครบวงจร

### 🏗️ สถาปัตยกรรม Frontend
- **Authentication**: Face Login และ Traditional Login
- **Social Platform**: Posts, Comments, Likes, Face Tagging  
- **Real-time Features**: Live face detection และ WebSocket
- **UI/UX**: Modern responsive design ด้วย Ant Design

---

## 2.2 Face Authentication System

### 2.2.1 Face Login Modal Component

```tsx
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
  Col 
} from 'antd';
import { 
  CameraOutlined, 
  ScanOutlined, 
  CheckCircleOutlined, 
  ExclamationCircleOutlined,
  UserOutlined,
  SafetyOutlined 
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

  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const scanIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const TARGET_SCANS = 20;
  const SCAN_INTERVAL = 500; // ms
  const CONFIDENCE_THRESHOLD = 0.8;
  const MAX_SPOOFING_ALLOWED = 3;

  // Initialize Camera
  const initCamera = useCallback(async () => {
    try {
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
        setStep('scanning');
      }
    } catch (error) {
      console.error('Camera initialization failed:', error);
      setError('ไม่สามารถเข้าถึงกล้องได้ กรุณาอนุญาตการใช้งานกล้อง');
      onError?.('Camera access denied');
    }
  }, [onError]);

  // Start Face Scanning
  const startScanning = useCallback(() => {
    if (!isStreaming || isScanning) return;

    setIsScanning(true);
    setScanProgress(0);
    setScanResults([]);
    setSpoofingCount(0);
    setError(null);

    scanIntervalRef.current = setInterval(async () => {
      await performScan();
    }, SCAN_INTERVAL);
  }, [isStreaming, isScanning]);

  // Perform Single Scan
  const performScan = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) return;

    // Capture frame
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    // Convert to base64
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    const base64Data = imageData.split(',')[1];

    try {
      // Send for analysis
      const response = await fetch('/api/face/analyze-login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_base64: base64Data,
          anti_spoofing: true,
          face_recognition: true
        })
      });

      const result = await response.json();

      if (result.success) {
        const scanResult: ScanResult = {
          id: Date.now().toString(),
          isReal: result.anti_spoofing?.is_real || false,
          identity: result.face_recognition?.best_match?.person_name || null,
          confidence: result.face_recognition?.best_match?.confidence || 0,
          quality: result.face_detection?.quality_score || 0,
          timestamp: Date.now(),
          similarity: result.face_recognition?.best_match?.similarity || 0
        };

        setScanResults(prev => [...prev, scanResult]);

        if (!scanResult.isReal) {
          setSpoofingCount(prev => prev + 1);
        }

        // Update progress
        setScanProgress(prev => {
          const newProgress = Math.min(prev + (100 / TARGET_SCANS), 100);
          
          // Check if scanning is complete
          if (newProgress >= 100) {
            completeScan();
          }
          
          return newProgress;
        });
      }
    } catch (error) {
      console.error('Scan failed:', error);
    }
  }, []);

  // Complete Scanning Process
  const completeScan = useCallback(() => {
    if (scanIntervalRef.current) {
      clearInterval(scanIntervalRef.current);
      scanIntervalRef.current = null;
    }

    setIsScanning(false);
    setStep('processing');
    setIsProcessing(true);

    // Process results after a short delay
    setTimeout(() => {
      processResults();
    }, 1000);
  }, []);

  // Process Scan Results
  const processResults = useCallback(() => {
    const validScans = scanResults.filter(r => 
      r.isReal && r.confidence >= CONFIDENCE_THRESHOLD
    );

    if (spoofingCount > MAX_SPOOFING_ALLOWED) {
      setError('ตรวจพบการปลอมแปลง กรุณาลองใหม่อีกครั้ง');
      setStep('setup');
      setIsProcessing(false);
      return;
    }

    if (validScans.length < TARGET_SCANS * 0.6) {
      setError('การสแกนไม่เพียงพอ กรุณาลองใหม่อีกครั้ง');
      setStep('setup');
      setIsProcessing(false);
      return;
    }

    // Find the most frequent identity
    const identityStats = validScans.reduce((acc, result) => {
      if (result.identity) {
        acc[result.identity] = (acc[result.identity] || 0) + 1;
      }
      return acc;
    }, {} as Record<string, number>);

    const winnerIdentity = Object.entries(identityStats).reduce((winner, [identity, count]) => {
      return count > winner.count ? { identity, count } : winner;
    }, { identity: '', count: 0 });

    if (winnerIdentity.count >= validScans.length * 0.6) {
      // Success
      setStep('completed');
      setIsProcessing(false);
      
      onSuccess({
        identity: winnerIdentity.identity,
        confidence: winnerIdentity.count / validScans.length,
        scanResults: validScans,
        totalScans: scanResults.length
      });
    } else {
      setError('ไม่สามารถยืนยันตัวตนได้ กรุณาลองใหม่อีกครั้ง');
      setStep('setup');
      setIsProcessing(false);
    }
  }, [scanResults, spoofingCount, onSuccess]);

  return (
    <Modal
      title="เข้าสู่ระบบด้วยใบหน้า"
      open={visible}
      onCancel={onClose}
      footer={null}
      width={800}
      centered
    >
      <div style={{ textAlign: 'center' }}>
        {/* Video Preview */}
        <div style={{ position: 'relative', marginBottom: 20 }}>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{
              width: '100%',
              maxWidth: 640,
              height: 'auto',
              borderRadius: 8,
              border: '2px solid #d9d9d9'
            }}
          />
          <canvas ref={canvasRef} style={{ display: 'none' }} />
        </div>

        {/* Progress and Stats */}
        {isScanning && (
          <Card style={{ marginBottom: 20 }}>
            <Progress percent={Math.round(scanProgress)} />
            <Row gutter={16} style={{ marginTop: 16 }}>
              <Col span={8}>
                <Statistic
                  title="สแกนแล้ว"
                  value={scanResults.length}
                  suffix={`/ ${TARGET_SCANS}`}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="ความแม่นยำ"
                  value={scanResults.length > 0 ? 
                    Math.round((scanResults.filter(r => r.isReal).length / scanResults.length) * 100) : 0
                  }
                  suffix="%"
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="การปลอมแปลง"
                  value={spoofingCount}
                  valueStyle={{ color: spoofingCount > 0 ? '#cf1322' : '#3f8600' }}
                />
              </Col>
            </Row>
          </Card>
        )}

        {/* Action Buttons */}
        <Space>
          {step === 'setup' && (
            <Button
              type="primary"
              icon={<CameraOutlined />}
              size="large"
              onClick={initCamera}
              loading={isStreaming}
            >
              เริ่มกล้อง
            </Button>
          )}
          
          {step === 'scanning' && !isScanning && (
            <Button
              type="primary"
              icon={<ScanOutlined />}
              size="large"
              onClick={startScanning}
            >
              เริ่มสแกน
            </Button>
          )}
          
          {isProcessing && (
            <Button loading size="large">
              กำลังประมวลผล...
            </Button>
          )}
        </Space>

        {/* Error Display */}
        {error && (
          <Alert
            message={error}
            type="error"
            style={{ marginTop: 20 }}
            showIcon
          />
        )}
      </div>
    </Modal>
  );
};

export default FaceLoginModal;
```

## 2.3 Real-time Face Detection

### 2.3.1 ชุดคำสั่ง Real-time Face Detection Component

```tsx
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
  }, []);

  // ตรวจจับใบหน้าจาก video frame
  const detectFaces = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !isStreaming) {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) return;

    // Capture current frame
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    // Convert to base64
    const imageData = canvas.toDataURL('image/jpeg', 0.7);
    const base64Data = imageData.split(',')[1];

    try {
      const startTime = performance.now();
      
      const response = await fetch('/api/face/detect', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_base64: base64Data,
          model_name: 'yolov11m',
          conf_threshold: 0.5,
          max_faces: 10
        })
      });

      const result = await response.json();
      const endTime = performance.now();
      
      if (result.success && result.faces) {
        setFaces(result.faces);
        setDetectionCount(prev => prev + 1);
        
        // Calculate FPS
        const detectionTime = endTime - startTime;
        setFps(Math.round(1000 / detectionTime));
        
        // วาดกรอบรอบใบหน้าที่ตรวจพบ
        drawFaceBoxes(ctx, result.faces, canvas.width, canvas.height);
      }
    } catch (err) {
      console.error('Face detection error:', err);
    }
  }, [isStreaming]);

  // วาดกรอบรอบใบหน้า
  const drawFaceBoxes = useCallback((
    ctx: CanvasRenderingContext2D, 
    detectedFaces: DetectedFace[], 
    canvasWidth: number, 
    canvasHeight: number
  ) => {
    // Clear previous drawings
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);
    
    // Redraw video frame
    if (videoRef.current) {
      ctx.drawImage(videoRef.current, 0, 0, canvasWidth, canvasHeight);
    }

    // Draw face boxes
    detectedFaces.forEach((face, index) => {
      const { x1, y1, x2, y2, confidence } = face.bbox;
      
      // สีของกรอบขึ้นอยู่กับความมั่นใจ
      const color = confidence > 0.8 ? '#52c41a' : 
                   confidence > 0.6 ? '#faad14' : '#ff4d4f';
      
      // วาดกรอบ
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      
      // วาดข้อความแสดงความมั่นใจ
      ctx.fillStyle = color;
      ctx.font = '16px Arial';
      ctx.fillText(
        `Face ${index + 1}: ${(confidence * 100).toFixed(1)}%`,
        x1,
        y1 > 20 ? y1 - 5 : y1 + 20
      );
    });
  }, []);

  // เริ่มการตรวจจับต่อเนื่อง
  const startDetection = useCallback(() => {
    if (!isStreaming || isDetecting) return;
    
    setIsDetecting(true);
    setDetectionCount(0);
    
    intervalRef.current = setInterval(() => {
      detectFaces();
    }, 100); // ตรวจจับทุก 100ms
  }, [isStreaming, isDetecting, detectFaces]);

  // หยุดการตรวจจับ
  const stopDetection = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsDetecting(false);
    setFaces([]);
  }, []);

  // Cleanup เมื่อปิด modal
  useEffect(() => {
    if (!visible) {
      stopCamera();
    }
  }, [visible, stopCamera]);

  return (
    <Modal
      title="การตรวจจับใบหน้าแบบเรียลไทม์"
      open={visible}
      onCancel={onClose}
      footer={null}
      width={800}
      centered
    >
      <div style={{ textAlign: 'center' }}>
        {/* Video Display */}
        <div style={{ position: 'relative', marginBottom: 20 }}>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{
              width: '100%',
              maxWidth: 640,
              height: 'auto',
              borderRadius: 8,
              border: '2px solid #d9d9d9',
              display: isStreaming ? 'block' : 'none'
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
              pointerEvents: 'none',
              display: isDetecting ? 'block' : 'none'
            }}
          />
        </div>

        {/* Statistics */}
        <Card style={{ marginBottom: 20 }}>
          <Space size="large">
            <div>
              <Badge status={faces.length > 0 ? "success" : "default"} />
              <Text strong>ใบหน้าที่ตรวจพบ: {faces.length}</Text>
            </div>
            <div>
              <EyeOutlined style={{ marginRight: 8 }} />
              <Text>FPS: {fps}</Text>
            </div>
            <div>
              <Text>การตรวจจับ: {detectionCount}</Text>
            </div>
          </Space>
        </Card>

        {/* Control Buttons */}
        <Space>
          {!isStreaming ? (
            <Button
              type="primary"
              icon={<CameraOutlined />}
              onClick={startCamera}
              size="large"
            >
              เริ่มกล้อง
            </Button>
          ) : (
            <>
              {!isDetecting ? (
                <Button
                  type="primary"
                  icon={<EyeOutlined />}
                  onClick={startDetection}
                  size="large"
                >
                  เริ่มตรวจจับ
                </Button>
              ) : (
                <Button
                  danger
                  icon={<StopOutlined />}
                  onClick={stopDetection}
                  size="large"
                >
                  หยุดตรวจจับ
                </Button>
              )}
              
              <Button onClick={stopCamera} size="large">
                หยุดกล้อง
              </Button>
            </>
          )}
        </Space>

        {/* Error Display */}
        {error && (
          <Alert
            message={error}
            type="error"
            style={{ marginTop: 20 }}
            showIcon
          />
        )}
      </div>
    </Modal>
  );
};

export default RealTimeFaceDetection;
```

## 2.4 Social Media Features

### 2.4.1 ชุดคำสั่ง Post Creation with Face Tagging

```tsx
'use client';

import React, { useState } from 'react';
import {
  Modal,
  Form,
  Input,
  Button,
  Upload,
  Space,
  Switch,
  Tag,
  Avatar,
  message
} from 'antd';
import {
  PlusOutlined,
  EnvironmentOutlined,
  UserOutlined,
  TagOutlined
} from '@ant-design/icons';
import type { UploadProps, UploadFile } from 'antd';
import FaceTagModal from './FaceTagModal';

const { TextArea } = Input;

interface PostCreateModalProps {
  visible: boolean;
  onCancel: () => void;
  onSuccess: (post: any) => void;
}

interface TaggedUser {
  id: number;
  username: string;
  fullName: string;
  avatarUrl?: string;
  confidence: number;
  faceRegion: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

const PostCreateModal: React.FC<PostCreateModalProps> = ({
  visible,
  onCancel,
  onSuccess
}) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [selectedTags, setSelectedTags] = useState<TaggedUser[]>([]);
  const [showFaceTagModal, setShowFaceTagModal] = useState(false);
  const [uploadedImageFile, setUploadedImageFile] = useState<File | null>(null);

  // Handle form submission
  const handleSubmit = async (values: any) => {
    try {
      setLoading(true);
      
      const token = localStorage.getItem('token');
      if (!token) {
        message.error('กรุณาเข้าสู่ระบบก่อน');
        return;
      }

      // Step 1: Create the post
      const formData = new FormData();
      formData.append('content', values.content || '');
      formData.append('location', values.location || '');
      formData.append('isPublic', values.isPublic !== false ? 'true' : 'false');

      if (fileList[0]?.originFileObj) {
        formData.append('image', fileList[0].originFileObj);
      }

      const response = await fetch('/api/posts/create', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        
        // Step 2: Save face tags if available
        if (result.data && selectedTags.length > 0) {
          await saveFaceTags(result.data.id, selectedTags);
        }

        message.success('โพสต์สำเร็จ!');
        onSuccess(result.data);
        handleReset();
      } else {
        const error = await response.json();
        message.error(error.message || 'เกิดข้อผิดพลาดในการโพสต์');
      }
    } catch (error) {
      console.error('Post creation error:', error);
      message.error('เกิดข้อผิดพลาดในการโพสต์');
    } finally {
      setLoading(false);
    }
  };

  // Save face tags
  const saveFaceTags = async (postId: number, tags: TaggedUser[]) => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('/api/posts/face-tags', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          postId,
          tags: tags.map(tag => ({
            userId: tag.id,
            faceRegion: tag.faceRegion,
            confidence: tag.confidence
          }))
        })
      });

      if (!response.ok) {
        console.error('Failed to save face tags');
      }
    } catch (error) {
      console.error('Face tags save error:', error);
    }
  };

  // Handle image upload
  const handleUpload: UploadProps['customRequest'] = (options) => {
    const { file, onSuccess } = options;
    
    if (file instanceof File) {
      setUploadedImageFile(file);
      onSuccess?.('ok');
    }
  };

  const uploadProps: UploadProps = {
    customRequest: handleUpload,
    listType: 'picture-card',
    fileList,
    onChange: ({ fileList: newFileList }) => setFileList(newFileList),
    beforeUpload: (file) => {
      const isImage = file.type.startsWith('image/');
      if (!isImage) {
        message.error('สามารถอัปโหลดไฟล์รูปภาพเท่านั้น!');
      }
      const isLt5M = file.size / 1024 / 1024 < 5;
      if (!isLt5M) {
        message.error('รูปภาพต้องมีขนาดไม่เกิน 5MB!');
      }
      return isImage && isLt5M;
    },
    maxCount: 1,
  };

  // Handle face tagging
  const handleFaceTag = () => {
    if (!uploadedImageFile) {
      message.warning('กรุณาอัปโหลดรูปภาพก่อน');
      return;
    }
    setShowFaceTagModal(true);
  };

  // Handle face tag completion
  const handleFaceTagComplete = (tags: TaggedUser[]) => {
    setSelectedTags(tags);
    setShowFaceTagModal(false);
    message.success(`แท็กใบหน้าสำเร็จ: ${tags.length} คน`);
  };

  // Remove tag
  const removeTag = (tagId: number) => {
    setSelectedTags(prev => prev.filter(tag => tag.id !== tagId));
  };

  // Reset form
  const handleReset = () => {
    form.resetFields();
    setFileList([]);
    setSelectedTags([]);
    setUploadedImageFile(null);
  };

  const uploadButton = (
    <div>
      <PlusOutlined />
      <div style={{ marginTop: 8 }}>อัปโหลด</div>
    </div>
  );

  return (
    <>
      <Modal
        title="สร้างโพสต์ใหม่"
        open={visible}
        onCancel={onCancel}
        footer={null}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
        >
          {/* Content Input */}
          <Form.Item name="content" label="เนื้อหา">
            <TextArea
              rows={4}
              placeholder="คุณกำลังคิดอะไรอยู่?"
              maxLength={1000}
              showCount
            />
          </Form.Item>

          {/* Image Upload */}
          <Form.Item label="รูปภาพ">
            <Upload {...uploadProps}>
              {fileList.length >= 1 ? null : uploadButton}
            </Upload>
          </Form.Item>

          {/* Face Tagging */}
          {fileList.length > 0 && (
            <Form.Item label="แท็กใบหน้า">
              <Space direction="vertical" style={{ width: '100%' }}>
                <Button
                  type="dashed"
                  icon={<TagOutlined />}
                  onClick={handleFaceTag}
                  block
                >
                  แท็กใบหน้าในรูปภาพ
                </Button>
                
                {/* Display tagged users */}
                {selectedTags.length > 0 && (
                  <div>
                    <Text strong>ผู้ที่ถูกแท็ก:</Text>
                    <div style={{ marginTop: 8 }}>
                      {selectedTags.map(tag => (
                        <Tag
                          key={tag.id}
                          closable
                          onClose={() => removeTag(tag.id)}
                          style={{ marginBottom: 8 }}
                        >
                          <Avatar
                            size="small"
                            src={tag.avatarUrl}
                            icon={<UserOutlined />}
                            style={{ marginRight: 4 }}
                          />
                          {tag.fullName} ({(tag.confidence * 100).toFixed(1)}%)
                        </Tag>
                      ))}
                    </div>
                  </div>
                )}
              </Space>
            </Form.Item>
          )}

          {/* Location */}
          <Form.Item name="location" label="สถานที่">
            <Input
              prefix={<EnvironmentOutlined />}
              placeholder="เพิ่มสถานที่"
            />
          </Form.Item>

          {/* Privacy Setting */}
          <Form.Item name="isPublic" label="การแสดงผล" valuePropName="checked">
            <Switch
              defaultChecked
              checkedChildren="สาธารณะ"
              unCheckedChildren="เฉพาะเพื่อน"
            />
          </Form.Item>

          {/* Action Buttons */}
          <Form.Item>
            <Space>
              <Button onClick={onCancel}>
                ยกเลิก
              </Button>
              <Button
                type="primary"
                htmlType="submit"
                loading={loading}
              >
                โพสต์
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* Face Tag Modal */}
      <FaceTagModal
        visible={showFaceTagModal}
        imageFile={uploadedImageFile}
        onCancel={() => setShowFaceTagModal(false)}
        onComplete={handleFaceTagComplete}
      />
    </>
  );
};

export default PostCreateModal;
```

## 2.5 Custom Hooks และ APIs

### 2.5.1 ชุดคำสั่ง Custom Hooks สำหรับ Face Detection

```tsx
import { useState, useCallback, useRef } from 'react';

interface FaceDetectionResult {
  success: boolean;
  faces: any[];
  model_used: string;
  processing_time: number;
}

export const useFaceDetection = () => {
  const [isDetecting, setIsDetecting] = useState(false);
  const [results, setResults] = useState<FaceDetectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const detectFaces = useCallback(async (
    imageBase64: string,
    options: {
      model_name?: string;
      conf_threshold?: number;
      max_faces?: number;
    } = {}
  ) => {
    setIsDetecting(true);
    setError(null);

    try {
      const response = await fetch('/api/face/detect', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_base64: imageBase64,
          model_name: options.model_name || 'auto',
          conf_threshold: options.conf_threshold || 0.5,
          max_faces: options.max_faces || 10
        })
      });

      const result = await response.json();
      
      if (result.success) {
        setResults(result);
      } else {
        setError(result.message || 'การตรวจจับใบหน้าล้มเหลว');
      }
    } catch (err) {
      setError('เกิดข้อผิดพลาดในการเชื่อมต่อ');
      console.error('Face detection error:', err);
    } finally {
      setIsDetecting(false);
    }
  }, []);

  return {
    detectFaces,
    isDetecting,
    results,
    error,
    reset: () => {
      setResults(null);
      setError(null);
    }
  };
};
```

### 2.5.2 ชุดคำสั่ง API Service Functions

```tsx
// API Service for Face Recognition System
class FaceAPIService {
  private baseURL = '/api';

  // Face Detection
  async detectFaces(imageBase64: string, options: any = {}) {
    const response = await fetch(`${this.baseURL}/face/detect`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image_base64: imageBase64,
        ...options
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  // Face Recognition
  async recognizeFace(imageBase64: string, options: any = {}) {
    const response = await fetch(`${this.baseURL}/face/recognize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        face_image_base64: imageBase64,
        ...options
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  // Add Face to Gallery
  async addFace(imageBase64: string, personName: string, options: any = {}) {
    const response = await fetch(`${this.baseURL}/face/add-face`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        face_image_base64: imageBase64,
        person_name: personName,
        ...options
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  // Anti-Spoofing Detection
  async checkAntiSpoofing(imageBase64: string) {
    const response = await fetch(`${this.baseURL}/face/anti-spoofing`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image_base64: imageBase64
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }

  // Complete Analysis (Detection + Recognition + Anti-Spoofing)
  async analyzeImage(imageBase64: string, options: any = {}) {
    const response = await fetch(`${this.baseURL}/face/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image_base64: imageBase64,
        face_detection: true,
        face_recognition: true,
        anti_spoofing: true,
        ...options
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }
}

export const faceAPI = new FaceAPIService();
```

---

*เอกสารนี้แสดงชุดคำสั่งหลักสำหรับ Frontend TypeScript React ของระบบแพลตฟอร์มสื่อสังคมออนไลน์และการจดจำใบหน้า รวมถึงการยืนยันตัวตน การตรวจจับใบหน้าแบบเรียลไทม์ และการจัดการโซเชียลมีเดีย*
