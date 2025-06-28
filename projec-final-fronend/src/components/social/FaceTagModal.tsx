'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { 
  Modal, 
  Button, 
  Progress, 
  Typography, 
  Alert, 
  Space, 
  Card, 
  Row, 
  Col,
  Tag,
  Avatar,
  Checkbox,
  List,
  Empty
} from 'antd';
import { 
  CameraOutlined, 
  ScanOutlined, 
  CheckCircleOutlined, 
  ExclamationCircleOutlined,
  UserOutlined,
  EyeOutlined,
  TagOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;

interface FaceTagModalProps {
  visible: boolean;
  onClose: () => void;
  onTagsSelected: (tags: TaggedUser[]) => void;
  imageFile: File;
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

interface DetectedFace {
  region: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  confidence: number;
}

interface RecognitionResult {
  best_match?: {
    person_id: string;
    confidence: number;
    similarity: number;
  };
  success: boolean;
}

const FaceTagModal: React.FC<FaceTagModalProps> = ({
  visible,
  onClose,
  onTagsSelected,
  imageFile
}) => {
  const [step, setStep] = useState<'detecting' | 'recognizing' | 'selecting'>('detecting');
  const [progress, setProgress] = useState(0);
  const [detectedFaces, setDetectedFaces] = useState<DetectedFace[]>([]);
  const [recognizedUsers, setRecognizedUsers] = useState<TaggedUser[]>([]);
  const [selectedTags, setSelectedTags] = useState<TaggedUser[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  // API URL - Use the correct Face API port (8080, not 5000)
  const apiUrl = process.env.NODE_ENV === 'production' 
    ? process.env.NEXT_PUBLIC_FACE_API_URL || 'http://localhost:8080'
    : 'http://localhost:8080';

  // Convert image file to base64
  const convertToBase64 = useCallback((file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = (reader.result as string).split(',')[1];
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }, []);

  // Crop face from image
  const cropFaceFromImage = useCallback((
    imageElement: HTMLImageElement, 
    face: DetectedFace  ): Promise<Blob> => {
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d')!;
      
      // Validate face region data
      if (!face.region || typeof face.region.width !== 'number' || typeof face.region.height !== 'number') {
        console.error('Invalid face region:', face.region);
        // Use fallback values
        canvas.width = 100;
        canvas.height = 100;
        ctx.fillStyle = '#f0f0f0';
        ctx.fillRect(0, 0, 100, 100);
        canvas.toBlob((blob) => resolve(blob!), 'image/jpeg', 0.9);
        return;
      }
      
      // Set canvas size to face region
      canvas.width = Math.max(1, face.region.width);
      canvas.height = Math.max(1, face.region.height);
      
      // Draw cropped face
      ctx.drawImage(
        imageElement,
        face.region.x || 0, 
        face.region.y || 0, 
        face.region.width, 
        face.region.height,
        0, 0, 
        face.region.width, 
        face.region.height
      );
      
      canvas.toBlob((blob) => {
        resolve(blob!);
      }, 'image/jpeg', 0.9);
    });
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

  // Main face detection and recognition process
  const processFaceTagging = useCallback(async () => {
    try {
      setIsProcessing(true);
      setError(null);
      setProgress(10);

      // Step 1: Convert image to base64
      console.log('🔄 Converting image to base64...');
      const imageBase64 = await convertToBase64(imageFile);
      setProgress(20);      // Step 2: Detect faces (using FormData like FaceLoginModal)
      console.log('👁️ Detecting faces...');
      setStep('detecting');
      
      // Convert base64 back to blob for FormData upload (like FaceLoginModal)
      const imageBlob = await new Promise<Blob>((resolve) => {
        const byteCharacters = atob(imageBase64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        resolve(new Blob([byteArray], { type: 'image/jpeg' }));
      });

      const faceDetectionData = new FormData();
      faceDetectionData.append('file', imageBlob, 'face.jpg');
      faceDetectionData.append('model_name', 'yolov11m');
      faceDetectionData.append('conf_threshold', '0.5');
      faceDetectionData.append('max_faces', '50'); // Allow multiple faces for tagging
      
      const detectResponse = await fetch(`${apiUrl}/api/face-detection/detect`, {
        method: 'POST',
        body: faceDetectionData
      });

      if (!detectResponse.ok) {
        throw new Error('ไม่สามารถตรวจจับใบหน้าได้');
      }      const detectResult = await detectResponse.json();
      console.log('👁️ Detection result (full):', JSON.stringify(detectResult, null, 2));
      console.log('👁️ Raw faces data:', detectResult.faces);
      console.log('👁️ Faces array length:', detectResult.faces?.length);

      if (!detectResult.success || !detectResult.faces || detectResult.faces.length === 0) {
        console.warn('👁️ No faces detected or API returned error:', detectResult);
        throw new Error('ไม่พบใบหน้าในรูปภาพ');
      }

      // Filter out empty objects from faces array
      const validFaces = detectResult.faces.filter((face: any) => {
        const isValid = face && typeof face === 'object' && Object.keys(face).length > 0;
        if (!isValid) {
          console.warn('👁️ Skipping empty or invalid face object:', face);
        }
        return isValid;
      });

      console.log('👁️ Valid faces after filtering:', validFaces.length);

      if (validFaces.length === 0) {
        throw new Error('ไม่พบข้อมูลใบหน้าที่ถูกต้อง');
      }      // Debug: ดูโครงสร้างข้อมูลจริง
      console.log('👁️ First valid face structure:', JSON.stringify(validFaces[0], null, 2));
      if (validFaces[0]?.bbox) {
        console.log('👁️ First face bbox details:', validFaces[0].bbox);
        console.log('👁️ Bbox type:', typeof validFaces[0].bbox);
        console.log('👁️ Bbox is array:', Array.isArray(validFaces[0].bbox));
        console.log('👁️ Bbox keys:', Object.keys(validFaces[0].bbox || {}));
      }      const faces = validFaces.map((face: any, index: number) => {
        console.log(`👁️ Processing face ${index + 1}:`, JSON.stringify(face, null, 2));
        
        // Handle bbox format like FaceLoginModal (x1, y1, x2, y2)
        let region;
        
        if (face.bbox) {
          const bbox = face.bbox;
          console.log('👁️ Face bbox:', bbox);
          
          // Expected format: {x1, y1, x2, y2}
          if (bbox.x1 !== undefined && bbox.y1 !== undefined && bbox.x2 !== undefined && bbox.y2 !== undefined) {
            region = {
              x: Math.round(bbox.x1),
              y: Math.round(bbox.y1),
              width: Math.round(bbox.x2 - bbox.x1),
              height: Math.round(bbox.y2 - bbox.y1)
            };
          } else if (Array.isArray(bbox) && bbox.length >= 4) {
            // Fallback: Array format [x1, y1, x2, y2]
            region = {
              x: Math.round(bbox[0]),
              y: Math.round(bbox[1]),
              width: Math.round(bbox[2] - bbox[0]),
              height: Math.round(bbox[3] - bbox[1])
            };
          } else {
            console.warn('👁️ Unknown bbox format:', bbox);
            region = { x: 10, y: 10, width: 100, height: 100 };
          }
        } else {
          console.warn('👁️ Face object missing bbox, using fallback:', Object.keys(face));
          region = { x: 10, y: 10, width: 100, height: 100 };
        }

        // Ensure region has valid values
        region.x = Math.max(0, region.x);
        region.y = Math.max(0, region.y);
        region.width = Math.max(1, region.width);
        region.height = Math.max(1, region.height);

        console.log(`👁️ Mapped region for face ${index + 1}:`, region);
        
        return {
          region,
          confidence: face.confidence || face.score || 0.5
        };
      });

      console.log('👁️ Final processed faces:', faces.length);
      
      if (faces.length === 0) {
        throw new Error('ไม่สามารถประมวลผลข้อมูลใบหน้าได้');
      }

      setDetectedFaces(faces);
      setProgress(40);

      // Step 3: Load image for cropping
      console.log('📷 Loading image for face cropping...');
      const imageElement = new Image();
      await new Promise((resolve, reject) => {
        imageElement.onload = resolve;
        imageElement.onerror = reject;
        imageElement.src = URL.createObjectURL(imageFile);
      });

      setProgress(50);
      setStep('recognizing');

      // Step 4: Recognize each face
      console.log('🧠 Recognizing faces...');
      const recognizedUsers: TaggedUser[] = [];
      
      for (let i = 0; i < faces.length; i++) {
        const face = faces[i];
        setProgress(50 + (i / faces.length) * 40);

        try {
          // Crop face
          const croppedBlob = await cropFaceFromImage(imageElement, face);
          const croppedBase64 = await new Promise<string>((resolve) => {
            const reader = new FileReader();
            reader.onloadend = () => {
              const base64 = (reader.result as string).split(',')[1];
              resolve(base64);
            };
            reader.readAsDataURL(croppedBlob);
          });          // Recognize face
          const recognitionResponse = await fetch(`${apiUrl}/api/face-recognition/recognize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },            body: JSON.stringify({
              face_image_base64: croppedBase64,
              model_name: 'facenet',
              top_k: 5,
              similarity_threshold: 0.5
            })
          });

          if (recognitionResponse.ok) {
            const recognitionResult: RecognitionResult = await recognitionResponse.json();
            console.log(`🧠 Recognition result for face ${i + 1}:`, recognitionResult);            if (recognitionResult.success && recognitionResult.best_match) {
              const personId = recognitionResult.best_match.person_id;
              const confidence = recognitionResult.best_match.confidence || 0;

              // Get user info from database
              const userInfo = await getUserInfo(personId);
              if (userInfo) {
                // Check if this user is already recognized (avoid duplicates)
                const existingUser = recognizedUsers.find(u => u.id === userInfo.id);
                if (!existingUser) {
                  recognizedUsers.push({
                    id: userInfo.id,
                    username: userInfo.username,
                    fullName: userInfo.fullName || `${userInfo.firstName || ''} ${userInfo.lastName || ''}`.trim(),
                    avatarUrl: userInfo.avatarUrl,
                    confidence,
                    faceRegion: face.region
                  });
                  console.log(`✅ Recognized user: ${userInfo.fullName} (${(confidence * 100).toFixed(1)}%)`);
                }
              }
            } else {
              console.log(`ℹ️ Face ${i + 1}: No match found or confidence too low`);
            }
          }
        } catch (faceError) {
          console.error(`Error processing face ${i + 1}:`, faceError);
        }
      }

      setRecognizedUsers(recognizedUsers);
      setProgress(100);
      setStep('selecting');

      console.log('✅ Face tagging process completed:', recognizedUsers);

    } catch (error) {
      console.error('❌ Face tagging error:', error);
      setError(error instanceof Error ? error.message : 'เกิดข้อผิดพลาดในการประมวลผลใบหน้า');
    } finally {
      setIsProcessing(false);
    }
  }, [imageFile, convertToBase64, cropFaceFromImage, getUserInfo, apiUrl]);

  // Start processing when modal opens
  useEffect(() => {
    if (visible && imageFile) {
      processFaceTagging();
    }
  }, [visible, imageFile, processFaceTagging]);

  // Handle tag selection
  const handleTagToggle = (user: TaggedUser, checked: boolean) => {
    if (checked) {
      setSelectedTags(prev => [...prev, user]);
    } else {
      setSelectedTags(prev => prev.filter(tag => tag.id !== user.id));
    }
  };

  // Handle confirm tags
  const handleConfirmTags = () => {
    onTagsSelected(selectedTags);
    onClose();
  };

  // Reset state when modal closes
  const handleClose = () => {
    setStep('detecting');
    setProgress(0);
    setDetectedFaces([]);
    setRecognizedUsers([]);
    setSelectedTags([]);
    setError(null);
    setIsProcessing(false);
    onClose();
  };

  return (
    <Modal
      title={
        <Space>
          <TagOutlined />
          แท็กเพื่อนในรูปภาพ
        </Space>
      }
      open={visible}
      onCancel={handleClose}
      width={600}
      footer={
        step === 'selecting' ? [
          <Button key="cancel" onClick={handleClose}>
            ยกเลิก
          </Button>,
          <Button 
            key="confirm" 
            type="primary" 
            onClick={handleConfirmTags}
            disabled={selectedTags.length === 0}
          >
            แท็ก {selectedTags.length} คน
          </Button>
        ] : null
      }
      maskClosable={false}
    >
      <div className="space-y-4">
        {/* Progress */}
        <div>
          <div className="flex justify-between items-center mb-2">
            <Text strong>
              {step === 'detecting' && 'กำลังตรวจจับใบหน้า...'}
              {step === 'recognizing' && 'กำลังจดจำใบหน้า...'}
              {step === 'selecting' && 'เลือกเพื่อนที่ต้องการแท็ก'}
            </Text>
            <Text type="secondary">{progress}%</Text>
          </div>
          <Progress 
            percent={progress} 
            status={error ? 'exception' : 'active'}
            size="small"
          />
        </div>

        {/* Error */}
        {error && (
          <Alert
            message="เกิดข้อผิดพลาด"
            description={error}
            type="error"
            showIcon
            action={
              <Button size="small" onClick={processFaceTagging}>
                ลองใหม่
              </Button>
            }
          />
        )}

        {/* Detection Results */}
        {step === 'detecting' && detectedFaces.length > 0 && (
          <Card size="small">
            <Space>
              <EyeOutlined style={{ color: '#1890ff' }} />
              <Text>พบใบหน้า {detectedFaces.length} หน้า</Text>
            </Space>
          </Card>
        )}

        {/* Recognition Results */}
        {step === 'selecting' && (
          <div>
            {recognizedUsers.length > 0 ? (
              <div>
                <Text strong className="block mb-3">
                  จดจำได้ {recognizedUsers.length} คน (เลือกเพื่อนที่ต้องการแท็ก):
                </Text>
                <List
                  size="small"
                  dataSource={recognizedUsers}
                  renderItem={(user) => (
                    <List.Item>
                      <Checkbox
                        checked={selectedTags.some(tag => tag.id === user.id)}
                        onChange={(e) => handleTagToggle(user, e.target.checked)}
                      >
                        <Space>
                          <Avatar 
                            src={user.avatarUrl} 
                            icon={<UserOutlined />}
                            size="small"
                          />
                          <div>
                            <div className="font-medium">{user.fullName}</div>
                            <div className="text-xs text-gray-500">
                              @{user.username} • ความมั่นใจ {(user.confidence * 100).toFixed(1)}%
                            </div>
                          </div>
                        </Space>
                      </Checkbox>
                    </List.Item>
                  )}
                />
              </div>
            ) : (
              <Empty
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                description="ไม่พบใบหน้าที่จดจำได้"
              />
            )}
          </div>
        )}

        {/* Selected Tags Summary */}
        {selectedTags.length > 0 && (
          <Card size="small" className="bg-blue-50">
            <Text strong>แท็กที่เลือก:</Text>
            <div className="mt-2 space-x-1">
              {selectedTags.map(tag => (
                <Tag key={tag.id} color="blue">
                  {tag.fullName}
                </Tag>
              ))}
            </div>
          </Card>
        )}
      </div>
    </Modal>
  );
};

export default FaceTagModal;
