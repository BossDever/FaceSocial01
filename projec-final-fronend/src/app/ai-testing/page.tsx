'use client';

import React, { useState, useCallback, useEffect } from 'react';
import Link from 'next/link';
import { 
  Card, 
  Button, 
  Row, 
  Col, 
  Typography, 
  Tabs,
  Upload,
  Image,
  Alert,
  Progress,
  Space,
  Divider,
  Statistic,
  Tag,
  App
} from 'antd';
import { 
  EyeOutlined, 
  SafetyOutlined,
  UserOutlined,
  InboxOutlined,
  CameraOutlined
} from '@ant-design/icons';
import { useFaceRecognition } from '@/hooks/useFaceRecognition';
import ModelSpecificApiStatus from '@/components/features/ModelSpecificApiStatus';
import RealTimeFaceDetection from '@/components/features/RealTimeFaceDetectionSimple';
import RealTimeAntiSpoofing from '@/components/features/RealTimeAntiSpoofing';
import CCTVComponent from '@/components/features/CCTVComponent';
import { suppressAntdCompatibilityWarning } from '@/utils/suppressWarnings';
import type { 
  FaceDetectionResult, 
  AntiSpoofingResult, 
  AgeGenderResult
} from '@/types/api';

// Navigation Component
const NavigationBar = () => {
  return (
    <nav className="bg-white shadow-sm border-b">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <h1 className="text-2xl font-bold text-blue-600">🤖 FaceSocial</h1>
            </div>
          </div>          <div className="hidden md:block">
            <div className="ml-10 flex items-baseline space-x-4">
              <Link href="/" className="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium">หน้าแรก</Link>
              <Link href="/ai-testing" className="text-blue-600 hover:text-blue-800 px-3 py-2 rounded-md text-sm font-medium">AI Testing</Link>
              <Link href="/#features" className="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium">ฟีเจอร์</Link>
              <Link href="/#api" className="text-gray-700 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium">API</Link>
              <Link href="/login" className="text-blue-600 hover:text-blue-800 px-3 py-2 rounded-md text-sm font-medium">เข้าสู่ระบบ</Link>
              <Link href="/register" className="bg-blue-600 text-white hover:bg-blue-700 px-4 py-2 rounded-md text-sm font-medium">สมัครสมาชิก</Link>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

// Interface สำหรับ API response ของ Anti-Spoofing
interface AntiSpoofingApiResponse {
  overall_result?: {
    is_real: boolean;
    confidence: number;
    spoofing_detected?: boolean;
  };
  model?: string;
  processing_time?: number;
}

// Interface สำหรับ API response ของ Age Gender Analysis
interface AgeGenderApiResponse {
  analyses?: AgeGenderResult[];
  total_faces?: number;
  processing_time?: number;
}

const { Title, Paragraph, Text } = Typography;
const { Dragger } = Upload;

function AITestingPageContent() {
  const [activeTab, setActiveTab] = useState('face-detection');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [processedImageUrl, setProcessedImageUrl] = useState<string | null>(null);  const [realtimeModalVisible, setRealtimeModalVisible] = useState(false);
  const [realtimeOptimizedModalVisible, setRealtimeOptimizedModalVisible] = useState(false);
  const [antispoofingModalVisible, setAntispoofingModalVisible] = useState(false);
  
  // ใช้ App hook แทน static notification
  const { notification } = App.useApp();
  
  // Suppress Ant Design compatibility warnings
  useEffect(() => {
    suppressAntdCompatibilityWarning();
  }, []);
  
  // Face Recognition hooks
  const { 
    detectFaces, 
    detectSpoofing, 
    analyzeAgeGender,
    loading, 
    error, 
    results 
  } = useFaceRecognition();

  // Local state สำหรับ Anti-Spoofing results ที่อัปเดตแล้ว
  const [localAntiSpoofingResults, setLocalAntiSpoofingResults] = useState<AntiSpoofingResult[] | null>(null);
  
  // Local state สำหรับ Age Gender results ที่อัปเดตแล้ว
  const [localAgeGenderResults, setLocalAgeGenderResults] = useState<AgeGenderResult[] | null>(null);

  // วาดกรอบใบหน้าบนภาพ
  const drawFaceBoundingBoxes = useCallback((imageUrl: string, faces: FaceDetectionResult[]) => {
    return new Promise<string>((resolve) => {
      const img = document.createElement('img');
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = img.width;
        canvas.height = img.height;
        
        // วาดภาพต้นฉบับ
        ctx?.drawImage(img, 0, 0);
        
        // วาดกรอบใบหน้า
        faces.forEach((face, index) => {
          if (ctx && face.bbox) {
            const { x1, y1, x2, y2, confidence } = face.bbox;
            
            // วาดสี่เหลี่ยมกรอบใบหน้า
            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 20; // เพิ่มความหนาให้มากขึ้นอีก
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            
            // วาดพื้นหลังข้อความ
            ctx.fillStyle = 'rgba(0, 255, 0, 0.9)';
            ctx.fillRect(x1, y1 - 45, 280, 40); // เพิ่มขนาดกล่องข้อความ
            
            // เขียนข้อความ
            ctx.fillStyle = '#000000';
            ctx.font = 'bold 20px Arial'; // เพิ่มขนาดตัวอักษร
            ctx.fillText(`Face ${index + 1} (${(confidence * 100).toFixed(1)}%)`, x1 + 10, y1 - 18);
          }
        });
        
        // แปลง canvas เป็น URL
        const processedImageUrl = canvas.toDataURL('image/jpeg', 0.9);
        resolve(processedImageUrl);
      };
      img.src = imageUrl;
    });
  }, []);

  // Function to draw anti-spoofing results using face detection coordinates
  const drawAntiSpoofingWithFaceCoords = useCallback((imageUrl: string, results: AntiSpoofingResult[]) => {
    return new Promise<string>((resolve) => {
      const img = document.createElement('img');
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        if (!ctx) {
          resolve(imageUrl);
          return;
        }
        
        canvas.width = img.width;
        canvas.height = img.height;
        
        // วาดภาพต้นฉบับ
        ctx.drawImage(img, 0, 0);
        
        console.log('🖼️ Canvas size:', canvas.width, 'x', canvas.height);
        console.log('🎯 Anti-spoofing results with face coords:', results);
        
        // วาดผลลัพธ์แต่ละใบหน้า (ใช้พิกัดจาก Face Detection โดยตรง)
        results.forEach((result, index) => {
          const { x, y, w, h } = result.region;
          const isReal = result.is_real;
          const confidence = result.confidence;
          
          console.log(`📊 Face ${index + 1}: x=${x}, y=${y}, w=${w}, h=${h}, isReal=${isReal}, confidence=${confidence.toFixed(3)}`);
          
          // ใช้พิกัดโดยตรงจาก Face Detection (ไม่ต้อง scale)
          const finalX = x;
          const finalY = y;
          const finalW = w;
          const finalH = h;
          
          // กรอบสี - เขียวถ้าจริง, แดงถ้าปลอม
          ctx.strokeStyle = isReal ? '#00ff00' : '#ff0000';
          ctx.lineWidth = 20; // เหมือนกับ Face Detection
          ctx.strokeRect(finalX, finalY, finalW, finalH);
          
          // วาดพื้นหลังข้อความ
          const bgColor = isReal ? 'rgba(0, 255, 0, 0.9)' : 'rgba(255, 0, 0, 0.9)';
          ctx.fillStyle = bgColor;
          const textY = finalY > 45 ? finalY - 45 : finalY + finalH + 5;
          ctx.fillRect(finalX, textY, 380, 40);
          
          // เขียนข้อความ
          ctx.fillStyle = '#000000';
          ctx.font = 'bold 20px Arial';
          const status = isReal ? 'REAL FACE ✓' : 'FAKE/SPOOFING ⚠️';
          ctx.fillText(`${status} (${(confidence * 100).toFixed(1)}%)`, finalX + 10, textY + 25);
        });
        
        // แปลง canvas เป็น URL
        const processedImageUrl = canvas.toDataURL('image/jpeg', 0.9);
        resolve(processedImageUrl);
      };
      img.src = imageUrl;
    });
  }, []);

  // Function to draw age gender results using face detection coordinates
  const drawAgeGenderWithFaceCoords = useCallback((imageUrl: string, results: AgeGenderResult[]) => {
    return new Promise<string>((resolve) => {
      const img = document.createElement('img');
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        if (!ctx) {
          resolve(imageUrl);
          return;
        }
        
        canvas.width = img.width;
        canvas.height = img.height;
        
        // วาดภาพต้นฉบับ
        ctx.drawImage(img, 0, 0);
        
        console.log('🖼️ Canvas size:', canvas.width, 'x', canvas.height);
        console.log('🎯 Age Gender results with face coords:', results);
        
        // วาดผลลัพธ์แต่ละใบหน้า (ใช้พิกัดจาก Face Detection โดยตรง)
        results.forEach((result, index) => {
          const { x, y, w, h } = result.face_region;
          const age = result.age;
          const gender = result.gender;
          const genderConfidence = result.gender_confidence;
          
          console.log(`📊 Face ${index + 1}: x=${x}, y=${y}, w=${w}, h=${h}, age=${age}, gender=${gender}, confidence=${genderConfidence.toFixed(1)}%`);
          
          // ใช้พิกัดโดยตรงจาก Face Detection (ไม่ต้อง scale)
          const finalX = x;
          const finalY = y;
          const finalW = w;
          const finalH = h;
          
          // กรอบสี - ฟ้าสำหรับ Age Gender
          ctx.strokeStyle = '#1890ff';
          ctx.lineWidth = 20; // เหมือนกับ Face Detection
          ctx.strokeRect(finalX, finalY, finalW, finalH);
          
          // วาดพื้นหลังข้อความ
          const bgColor = 'rgba(24, 144, 255, 0.9)';
          ctx.fillStyle = bgColor;
          const textY = finalY > 60 ? finalY - 60 : finalY + finalH + 5;
          ctx.fillRect(finalX, textY, 400, 55);
          
          // เขียนข้อความ
          ctx.fillStyle = '#ffffff';
          ctx.font = 'bold 20px Arial';
          ctx.fillText(`Age: ${age}, Gender: ${gender}`, finalX + 10, textY + 25);
          
          ctx.font = 'bold 16px Arial';
          ctx.fillText(`Confidence: ${genderConfidence.toFixed(1)}%`, finalX + 10, textY + 45);
        });
        
        // แปลง canvas เป็น URL
        const processedImageUrl = canvas.toDataURL('image/jpeg', 0.9);
        resolve(processedImageUrl);
      };
      img.src = imageUrl;
    });
  }, []);

  // ฟังก์ชันตัดใบหน้าออกจากรูปภาพ
  const cropFaceFromImage = useCallback(async (originalFile: File, bbox: FaceDetectionResult['bbox']): Promise<File> => {
    return new Promise((resolve, reject) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const img = document.createElement('img');
      
      if (!ctx) {
        reject(new Error('Cannot create canvas context'));
        return;
      }
      
      img.onload = () => {
        // คำนวณขนาดที่จะตัด (เพิ่ม padding รอบใบหน้า)
        const padding = 0.2; // เพิ่ม 20% รอบใบหน้า
        const faceWidth = bbox.x2 - bbox.x1;
        const faceHeight = bbox.y2 - bbox.y1;
        const paddingX = faceWidth * padding;
        const paddingY = faceHeight * padding;
        
        const cropX = Math.max(0, bbox.x1 - paddingX);
        const cropY = Math.max(0, bbox.y1 - paddingY);
        const cropWidth = Math.min(img.width - cropX, faceWidth + (paddingX * 2));
        const cropHeight = Math.min(img.height - cropY, faceHeight + (paddingY * 2));
        
        console.log(`✂️ ตัดใบหน้า: (${cropX.toFixed(0)}, ${cropY.toFixed(0)}) ขนาด ${cropWidth.toFixed(0)}x${cropHeight.toFixed(0)}`);
        
        // ตั้งค่า Canvas
        canvas.width = cropWidth;
        canvas.height = cropHeight;
        
        // วาดส่วนที่ตัดมาลงใน Canvas
        ctx.drawImage(
          img,
          cropX, cropY, cropWidth, cropHeight, // source
          0, 0, cropWidth, cropHeight // destination
        );
        
        // แปลง Canvas เป็น Blob แล้วเป็น File
        canvas.toBlob((blob) => {
          if (blob) {
            const croppedFile = new File([blob], `cropped_${originalFile.name}`, {
              type: originalFile.type
            });
            resolve(croppedFile);
          } else {
            reject(new Error('Failed to create cropped image blob'));
          }
        }, originalFile.type, 0.9);
      };
      
      img.onerror = () => reject(new Error('Failed to load image'));
      img.src = URL.createObjectURL(originalFile);
    });
  }, []);
  const handleFileUpload = (file: File) => {
    setSelectedFile(file);
    setProcessedImageUrl(null); // รีเซ็ตภาพที่ประมวลผลแล้ว
    setLocalAntiSpoofingResults(null); // รีเซ็ต Anti-Spoofing results
    setLocalAgeGenderResults(null); // รีเซ็ต Age Gender results
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreviewUrl(e.target?.result as string);
    };
    reader.readAsDataURL(file);
    return false; // Prevent auto upload
  };

  // Handle face detection
  const handleDetectFaces = async () => {
    if (!selectedFile) {
      notification.error({
        message: 'ข้อผิดพลาด',
        description: 'กรุณาเลือกรูปภาพก่อน'
      });
      return;
    }

    try {
      setUploadLoading(true);
      console.log('🚀 เริ่มตรวจจับใบหน้า:', selectedFile.name, selectedFile.type, selectedFile.size);
      
      const result = await detectFaces(selectedFile);
      console.log('✅ ผลลัพธ์การตรวจจับใบหน้า:', result);
      console.log('🔍 ข้อมูล results.detection:', results.detection);
      console.log('🔍 จำนวนใบหน้า:', result?.length || 0);
      
      if (result && result.length > 0) {
        // วาดกรอบใบหน้าบนภาพ
        if (previewUrl) {
          const processedImage = await drawFaceBoundingBoxes(previewUrl, result);
          setProcessedImageUrl(processedImage);
        }
        
        notification.success({
          message: 'สำเร็จ',
          description: `ตรวจพบใบหน้า ${result.length} ใบ และวาดกรอบแล้ว`
        });
      } else {
        notification.warning({
          message: 'ไม่พบใบหน้า',
          description: 'ไม่สามารถตรวจพบใบหน้าในรูปภาพนี้'
        });
      }
    } catch (error) {
      console.error('❌ เกิดข้อผิดพลาด:', error);
      notification.error({
        message: 'เกิดข้อผิดพลาด',
        description: error instanceof Error ? error.message : 'ไม่สามารถตรวจจับใบหน้าได้'
      });
    } finally {
      setUploadLoading(false);
    }
  };

  // Handle spoofing detection with face detection first
  const handleDetectSpoofing = async () => {
    if (!selectedFile) {
      notification.error({
        message: 'ข้อผิดพลาด',
        description: 'กรุณาเลือกรูปภาพก่อน'
      });
      return;
    }

    try {
      setUploadLoading(true);
      setProcessedImageUrl(null);
      setLocalAntiSpoofingResults(null); // รีเซ็ต local state
      
      console.log('🚀 เริ่มตรวจสอบการปลอมแปลง:', selectedFile.name);
      
      // Step 1: รัน Face Detection ก่อนเพื่อหาตำแหน่งใบหน้า
      console.log('🔍 Step 1: ค้นหาใบหน้าด้วย Face Detection...');
      const faceDetectionResult = await detectFaces(selectedFile);
      
      if (!faceDetectionResult || faceDetectionResult.length === 0) {
        notification.warning({
          message: 'ไม่พบใบหน้า',
          description: 'ไม่สามารถตรวจพบใบหน้าในรูปภาพ กรุณาลองใช้รูปภาพที่มีใบหน้าชัดเจนกว่า'
        });
        return;
      }
      
      console.log(`✅ พบใบหน้า ${faceDetectionResult.length} หน้า:`, faceDetectionResult);
      
      // Step 2: ประมวลผล Anti-Spoofing สำหรับใบหน้าแต่ละหน้า
      console.log('�️ Step 2: ตรวจสอบการปลอมแปลงสำหรับแต่ละใบหน้า...');
      
      const antispoofingResults: AntiSpoofingResult[] = [];
      
      for (let i = 0; i < faceDetectionResult.length; i++) {
        const face = faceDetectionResult[i];
        console.log(`🎯 กำลังประมวลผลใบหน้าที่ ${i + 1}/${faceDetectionResult.length}`);
        
        try {
          // ตัดใบหน้าออกมาจากรูปเดิม
          const croppedFace = await cropFaceFromImage(selectedFile, face.bbox);
          
          // ส่งใบหน้าที่ตัดแล้วไปตรวจสอบ Anti-Spoofing
          const spoofingResult = await detectSpoofing(croppedFace);
          
          // รวมข้อมูลจาก Face Detection และ Anti-Spoofing
          const spoofingResponse = spoofingResult as AntiSpoofingApiResponse;
          const combinedResult: AntiSpoofingResult = {
            is_real: spoofingResponse?.overall_result?.is_real || false,
            confidence: spoofingResponse?.overall_result?.confidence || 0,
            model_used: spoofingResponse?.model || 'DeepFace Silent Face Anti-Spoofing',
            region: {
              x: face.bbox.x1,
              y: face.bbox.y1,
              w: face.bbox.x2 - face.bbox.x1,
              h: face.bbox.y2 - face.bbox.y1
            },
            processing_time: spoofingResponse?.processing_time || 0
          };
          
          antispoofingResults.push(combinedResult);
          console.log(`✅ ใบหน้าที่ ${i + 1} - สถานะ: ${combinedResult.is_real ? 'จริง' : 'ปลอม'} (${(combinedResult.confidence * 100).toFixed(1)}%)`);
          
        } catch (error) {
          console.error(`❌ เกิดข้อผิดพลาดกับใบหน้าที่ ${i + 1}:`, error);
          // เพิ่มผลลัพธ์ที่มี error
          antispoofingResults.push({
            is_real: false,
            confidence: 0,
            model_used: 'DeepFace Silent Face Anti-Spoofing',
            region: {
              x: face.bbox.x1,
              y: face.bbox.y1,
              w: face.bbox.x2 - face.bbox.x1,
              h: face.bbox.y2 - face.bbox.y1
            },
            processing_time: 0
          });
        }
      }
      
      console.log('🎉 ผลลัพธ์รวม Anti-Spoofing:', antispoofingResults);
      
      // Step 3: วาดกรอบตามตำแหน่งจาก Face Detection
      if (previewUrl && antispoofingResults.length > 0) {
        console.log('🎨 Step 3: วาดกรอบผลลัพธ์ตามพิกัด Face Detection...');
        const processedImage = await drawAntiSpoofingWithFaceCoords(previewUrl, antispoofingResults);
        setProcessedImageUrl(processedImage);
      }
      
      // อัปเดต local state สำหรับแสดงผลลัพธ์
      setLocalAntiSpoofingResults(antispoofingResults);
      
      // แสดงผลสรุป
      const realFaces = antispoofingResults.filter(r => r.is_real).length;
      const fakeFaces = antispoofingResults.length - realFaces;
      
      notification.success({
        message: 'สำเร็จ',
        description: `ตรวจสอบเสร็จสมบูรณ์: ${antispoofingResults.length} ใบหน้า (${realFaces} จริง, ${fakeFaces} ปลอม)`
      });
      
    } catch (error) {
      console.error('❌ เกิดข้อผิดพลาด:', error);
      notification.error({
        message: 'เกิดข้อผิดพลาด',
        description: error instanceof Error ? error.message : 'ไม่สามารถตรวจสอบการปลอมแปลงได้'
      });
    } finally {
      setUploadLoading(false);
    }
  };

  // Handle age gender analysis with face detection first
  const handleAnalyzeAgeGender = async () => {
    if (!selectedFile) {
      notification.error({
        message: 'ข้อผิดพลาด',
        description: 'กรุณาเลือกรูปภาพก่อน'
      });
      return;
    }

    try {
      setUploadLoading(true);
      setProcessedImageUrl(null);
      setLocalAgeGenderResults(null); // รีเซ็ต local state
      
      console.log('🚀 เริ่มวิเคราะห์อายุและเพศ:', selectedFile.name);
      
      // Step 1: รัน Face Detection ก่อนเพื่อหาตำแหน่งใบหน้า
      console.log('🔍 Step 1: ค้นหาใบหน้าด้วย Face Detection...');
      const faceDetectionResult = await detectFaces(selectedFile);
      
      if (!faceDetectionResult || faceDetectionResult.length === 0) {
        notification.warning({
          message: 'ไม่พบใบหน้า',
          description: 'ไม่สามารถตรวจพบใบหน้าในรูปภาพ กรุณาลองใช้รูปภาพที่มีใบหน้าชัดเจนกว่า'
        });
        return;
      }
      
      console.log(`✅ พบใบหน้า ${faceDetectionResult.length} หน้า:`, faceDetectionResult);
      
      // Step 2: ประมวลผล Age Gender สำหรับใบหน้าแต่ละหน้า
      console.log('🧠 Step 2: วิเคราะห์อายุและเพศสำหรับแต่ละใบหน้า...');
      
      const ageGenderResults: AgeGenderResult[] = [];
      
      for (let i = 0; i < faceDetectionResult.length; i++) {
        const face = faceDetectionResult[i];
        console.log(`🎯 กำลังประมวลผลใบหน้าที่ ${i + 1}/${faceDetectionResult.length}`);
        
        try {
          // ตัดใบหน้าออกมาจากรูปเดิม
          const croppedFace = await cropFaceFromImage(selectedFile, face.bbox);
          
          // ส่งใบหน้าที่ตัดแล้วไปวิเคราะห์ Age Gender
          const ageGenderResult = await analyzeAgeGender(croppedFace);
          
          // รวมข้อมูลจาก Face Detection และ Age Gender Analysis
          const ageGenderResponse = ageGenderResult as AgeGenderApiResponse;
          if (ageGenderResponse?.analyses && ageGenderResponse.analyses.length > 0) {
            const analysis = ageGenderResponse.analyses[0]; // เอาผลลัพธ์แรก
            
            const combinedResult: AgeGenderResult = {
              age: analysis.age,
              gender: analysis.gender,
              gender_confidence: analysis.gender_confidence,
              face_region: {
                x: face.bbox.x1,
                y: face.bbox.y1,
                w: face.bbox.x2 - face.bbox.x1,
                h: face.bbox.y2 - face.bbox.y1
              }
            };
            
            ageGenderResults.push(combinedResult);
            console.log(`✅ ใบหน้าที่ ${i + 1} - อายุ: ${combinedResult.age}, เพศ: ${combinedResult.gender} (${combinedResult.gender_confidence.toFixed(1)}%)`);
          }
          
        } catch (error) {
          console.error(`❌ เกิดข้อผิดพลาดกับใบหน้าที่ ${i + 1}:`, error);
          // เพิ่มผลลัพธ์ที่มี error
          ageGenderResults.push({
            age: 0,
            gender: 'Unknown',
            gender_confidence: 0,
            face_region: {
              x: face.bbox.x1,
              y: face.bbox.y1,
              w: face.bbox.x2 - face.bbox.x1,
              h: face.bbox.y2 - face.bbox.y1
            }
          });
        }
      }
      
      console.log('🎉 ผลลัพธ์รวม Age Gender:', ageGenderResults);
      
      // Step 3: วาดกรอบตามตำแหน่งจาก Face Detection
      if (previewUrl && ageGenderResults.length > 0) {
        console.log('🎨 Step 3: วาดกรอบผลลัพธ์ตามพิกัด Face Detection...');
        const processedImage = await drawAgeGenderWithFaceCoords(previewUrl, ageGenderResults);
        setProcessedImageUrl(processedImage);
      }
      
      // อัปเดต local state สำหรับแสดงผลลัพธ์
      setLocalAgeGenderResults(ageGenderResults);
      
      // แสดงผลสรุป
      const avgAge = Math.round(ageGenderResults.reduce((sum, r) => sum + r.age, 0) / ageGenderResults.length);
      const maleCount = ageGenderResults.filter(r => r.gender.toLowerCase().includes('m')).length;
      const femaleCount = ageGenderResults.length - maleCount;
      
      notification.success({
        message: 'สำเร็จ',
        description: `วิเคราะห์เสร็จสมบูรณ์: ${ageGenderResults.length} ใบหน้า (อายุเฉลี่ย: ${avgAge} ปี, ชาย: ${maleCount}, หญิง: ${femaleCount})`
      });
      
    } catch (error) {
      console.error('❌ เกิดข้อผิดพลาด:', error);
      notification.error({
        message: 'เกิดข้อผิดพลาด',
        description: error instanceof Error ? error.message : 'ไม่สามารถวิเคราะห์อายุและเพศได้'
      });
    } finally {
      setUploadLoading(false);
    }
  };

  // Upload props
  const uploadProps = {
    name: 'file',
    multiple: false,
    accept: 'image/*',
    beforeUpload: handleFileUpload,
    showUploadList: false,
  };

  const tabItems = [
    {
      key: 'face-detection',
      label: (
        <span className="text-lg font-semibold">
          <EyeOutlined className="mr-2 text-emerald-600" />
          🔍 Face Detection
        </span>
      ),
      children: (
        <div>
          <Card className="mb-4">
            <Title level={4}>การตรวจจับใบหน้า (Face Detection)</Title>
            <Paragraph>
              อัปโหลดรูปภาพเพื่อตรวจจับใบหน้าและวิเคราะห์คุณลักษณะใบหน้า
            </Paragraph>
            
            <Row gutter={[24, 24]}>
              <Col xs={24} md={12}>
                <Card title="อัปโหลดรูปภาพ" size="small">
                  <Dragger {...uploadProps}>
                    <p className="ant-upload-drag-icon">
                      <InboxOutlined />
                    </p>
                    <p className="ant-upload-text">คลิกหรือลากไฟล์มาวางที่นี่</p>
                    <p className="ant-upload-hint">
                      รองรับไฟล์ .jpg, .jpeg, .png
                    </p>
                  </Dragger>
                  
                  {selectedFile && (
                    <div className="mt-4">
                      <Alert 
                        message={`เลือกไฟล์: ${selectedFile.name}`} 
                        type="info" 
                        showIcon 
                      />
                    </div>
                  )}
                  
                  <div className="mt-4">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Button 
                        type="primary" 
                        onClick={handleDetectFaces}
                        loading={loading || uploadLoading}
                        disabled={!selectedFile}
                        icon={<EyeOutlined />}
                        size="large"
                        block
                      >
                        {loading ? 'กำลังประมวลผล...' : 'เริ่มตรวจจับใบหน้า'}
                      </Button>                      <Button 
                        type="primary" 
                        onClick={() => setRealtimeModalVisible(true)}
                        icon={<CameraOutlined />}
                        size="large"
                        block
                      >
                        📹 Real-time Face Detection (Working)
                      </Button>

                      <Button 
                        type="default" 
                        onClick={() => setRealtimeOptimizedModalVisible(true)}
                        icon={<CameraOutlined />}
                        size="large"
                        block
                        disabled
                      >
                        � Alternative (Under Development)
                      </Button>

                      {/* API Status for Face Detection */}
                      <div className="flex justify-center mt-2">
                        <ModelSpecificApiStatus 
                          type="face-detection"
                          checkInterval={5000}
                          size="small"
                          showText={true}
                        />
                      </div>
                    </Space>
                  </div>
                </Card>
              </Col>
              
              <Col xs={24} md={12}>
                <Card title="ตัวอย่างรูปภาพ" size="small">
                  {previewUrl ? (
                    <Image 
                      src={previewUrl} 
                      alt="Preview" 
                      style={{ maxWidth: '100%', maxHeight: '300px' }}
                    />
                  ) : (
                    <div className="text-center py-8 text-gray-400">
                      ยังไม่มีรูปภาพที่เลือก
                    </div>
                  )}
                </Card>
              </Col>
            </Row>
            
            {/* ผลลัพธ์การตรวจจับ */}
            {(results.detection && results.detection.length > 0) ? (
              <div className="mt-4">
                <Divider>ผลลัพธ์การตรวจจับใบหน้า</Divider>
                
                {/* แสดงภาพที่มีกรอบใบหน้า */}
                {processedImageUrl && (
                  <Card className="mb-4">
                    <div className="text-center">
                      <Text strong>ภาพที่มีกรอบใบหน้า:</Text>
                      <div className="mt-2">
                        <Image 
                          src={processedImageUrl} 
                          alt="ภาพที่มีกรอบใบหน้า" 
                          style={{ maxWidth: '100%', maxHeight: '400px' }}
                        />
                      </div>
                      <div className="mt-2">
                        <Button 
                          size="small" 
                          onClick={() => setProcessedImageUrl(null)}
                        >
                          ซ่อนภาพที่มีกรอบ
                        </Button>
                      </div>
                    </div>
                  </Card>
                )}
                
                <Card>
                  <Statistic 
                    title="จำนวนใบหน้าที่พบ" 
                    value={results.detection.length} 
                    prefix={<UserOutlined />}
                  />
                  
                  {results.detection.map((face, index) => (
                    <Card key={index} size="small" className="mt-2">
                      <Text strong>ใบหน้าที่ {index + 1}</Text>
                      <div className="mt-2">
                        <Tag color="blue">ความเชื่อมั่น: {(face.bbox?.confidence * 100).toFixed(1)}%</Tag>
                        <Tag color="green">คุณภาพ: {face.quality_score}%</Tag>
                        <Tag color="purple">โมเดล: {face.model_used}</Tag>
                        <Tag color="orange">ขนาด: {face.bbox?.width}×{face.bbox?.height}</Tag>
                        {face.bbox?.area && <Tag color="cyan">พื้นที่: {face.bbox.area.toLocaleString()}</Tag>}
                      </div>
                      <div className="mt-2">
                        <Text type="secondary">
                          ตำแหน่ง: ({face.bbox?.x1}, {face.bbox?.y1}) ถึง ({face.bbox?.x2}, {face.bbox?.y2})
                        </Text>
                      </div>
                      <div className="mt-2">
                        <Text type="secondary">
                          เวลาประมวลผล: {face.processing_time?.toFixed(1)}ms
                        </Text>
                      </div>
                    </Card>
                  ))}
                </Card>
              </div>
            ) : loading ? (
              <div className="mt-4">
                <Alert 
                  message="กำลังประมวลผล..." 
                  description="กรุณารอสักครู่" 
                  type="info" 
                  showIcon 
                />
              </div>
            ) : (
              <div className="mt-4">
                <Alert 
                  message="ยังไม่มีผลลัพธ์" 
                  description={`อัปโหลดรูปภาพและคลิก "เริ่มตรวจจับใบหน้า" เพื่อเริ่มการวิเคราะห์ (Debug: results.detection = ${JSON.stringify(results.detection)})`}
                  type="warning" 
                  showIcon 
                />
              </div>
            )}
            
            {error && (
              <Alert 
                message="เกิดข้อผิดพลาด" 
                description={error} 
                type="error" 
                showIcon 
                className="mt-4"
              />
            )}
          </Card>
        </div>
      )
    },
    {
      key: 'anti-spoofing',
      label: (
        <span className="text-lg font-semibold">
          <SafetyOutlined className="mr-2 text-rose-600" />
          🛡️ Anti-Spoofing
        </span>
      ),
      children: (
        <div>
          <Card className="mb-4">
            <Title level={4}>การตรวจจับการปลอมแปลง (Anti-Spoofing)</Title>
            <Paragraph>
              ตรวจสอบว่าใบหน้าในรูปภาพเป็นของจริงหรือปลอม/สปูฟ
            </Paragraph>
            
            <Row gutter={[24, 24]}>
              <Col xs={24} md={12}>
                <Card title="อัปโหลดรูปภาพ" size="small">
                  <Dragger {...uploadProps}>
                    <p className="ant-upload-drag-icon">
                      <InboxOutlined />
                    </p>
                    <p className="ant-upload-text">คลิกหรือลากไฟล์มาวางที่นี่</p>
                    <p className="ant-upload-hint">
                      รองรับไฟล์ .jpg, .jpeg, .png
                    </p>
                  </Dragger>
                  
                  {selectedFile && (
                    <div className="mt-4">
                      <Alert 
                        message={`เลือกไฟล์: ${selectedFile.name}`} 
                        type="info" 
                        showIcon 
                      />
                    </div>
                  )}
                  
                  <div className="mt-4">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Button 
                        type="primary" 
                        onClick={handleDetectSpoofing}
                        loading={loading || uploadLoading}
                        disabled={!selectedFile}
                        icon={<SafetyOutlined />}
                        size="large"
                        block
                      >
                        {loading ? 'กำลังประมวลผล...' : 'ตรวจสอบการปลอมแปลง'}
                      </Button>
                      
                      <Button 
                        type="default" 
                        onClick={() => setAntispoofingModalVisible(true)}
                        icon={<CameraOutlined />}
                        size="large"
                        block
                      >
                        เปิดกล้องตรวจจับแบบ Real-time
                      </Button>

                      {/* API Status for Anti-Spoofing */}
                      <div className="flex justify-center mt-2">
                        <ModelSpecificApiStatus 
                          type="anti-spoofing"
                          checkInterval={5000}
                          size="small"
                          showText={true}
                        />
                      </div>
                    </Space>
                  </div>
                </Card>
              </Col>
              
              <Col xs={24} md={12}>
                <Card title="ตัวอย่างรูปภาพ" size="small">
                  {previewUrl ? (
                    <Image 
                      src={previewUrl} 
                      alt="Preview" 
                      style={{ maxWidth: '100%', maxHeight: '300px' }}
                    />
                  ) : (
                    <div className="text-center py-8 text-gray-400">
                      ยังไม่มีรูปภาพที่เลือก
                    </div>
                  )}
                </Card>
              </Col>
            </Row>
            
            {/* ผลลัพธ์การตรวจสอบ */}
            {(localAntiSpoofingResults && localAntiSpoofingResults.length > 0) ? (
              <div className="mt-4">
                <Divider>ผลลัพธ์การตรวจสอบการปลอมแปลง</Divider>
                
                {/* แสดงภาพที่มีกรอบ */}
                {processedImageUrl && (
                  <Card className="mb-4">
                    <div className="text-center">
                      <Text strong>ภาพที่มีกรอบการตรวจสอบ:</Text>
                      <div className="mt-2">
                        <Image 
                          src={processedImageUrl} 
                          alt="ภาพที่มีกรอบการตรวจสอบ" 
                          style={{ maxWidth: '100%', maxHeight: '400px' }}
                        />
                      </div>
                      <div className="mt-2">
                        <Button 
                          size="small" 
                          onClick={() => setProcessedImageUrl(null)}
                        >
                          ซ่อนภาพที่มีกรอบ
                        </Button>
                      </div>
                    </div>
                  </Card>
                )}
                
                <Card>
                  <Statistic 
                    title="จำนวนใบหน้าที่ตรวจสอบ" 
                    value={localAntiSpoofingResults.length} 
                    prefix={<SafetyOutlined />}
                  />
                  
                  {localAntiSpoofingResults.map((result, index) => (
                    <Card key={index} size="small" className="mt-2">
                      <Text strong>ใบหน้าที่ {index + 1}</Text>
                      <div className="mt-2">
                        <Tag color={result.is_real ? "green" : "red"}>
                          สถานะ: {result.is_real ? "ใบหน้าจริง" : "ใบหน้าปลอม"}
                        </Tag>
                        <Tag color="blue">ความเชื่อมั่น: {(result.confidence * 100).toFixed(1)}%</Tag>
                        <Tag color="purple">โมเดล: {result.model_used}</Tag>
                        {result.region && (
                          <Tag color="orange">
                            ขนาด: {result.region.w}×{result.region.h}
                          </Tag>
                        )}
                      </div>
                      {result.region && (
                        <div className="mt-2">
                          <Text type="secondary">
                            ตำแหน่ง: ({result.region.x}, {result.region.y}) ขนาด: {result.region.w}×{result.region.h}
                          </Text>
                        </div>
                      )}
                      <div className="mt-2">
                        <Text type="secondary">
                          เวลาประมวลผล: {result.processing_time?.toFixed(1)}ms
                        </Text>
                      </div>
                    </Card>
                  ))}
                </Card>
              </div>
            ) : loading ? (
              <div className="mt-4">
                <Alert 
                  message="กำลังประมวลผล..." 
                  description="กรุณารอสักครู่" 
                  type="info" 
                  showIcon 
                />
              </div>
            ) : (
              <div className="mt-4">
                <Alert 
                  message="ยังไม่มีผลลัพธ์" 
                  description={`อัปโหลดรูปภาพและคลิก "ตรวจสอบการปลอมแปลง" เพื่อเริ่มการวิเคราะห์`}
                  type="warning" 
                  showIcon 
                />
              </div>
            )}
            
            {error && (
              <Alert 
                message="เกิดข้อผิดพลาด" 
                description={error} 
                type="error" 
                showIcon 
                className="mt-4"
              />
            )}
          </Card>
        </div>
      )
    },
    {
      key: 'age-gender',
      label: (
        <span className="text-lg font-semibold">
          <UserOutlined className="mr-2 text-violet-600" />
          👤 Age & Gender
        </span>
      ),
      children: (
        <div>
          <Card className="mb-4">
            <Title level={4}>การวิเคราะห์อายุและเพศ (Age & Gender Analysis)</Title>
            <Paragraph>
              วิเคราะห์อายุและเพศของใบหน้าในรูปภาพอย่างแม่นยำ
            </Paragraph>
            
            <Row gutter={[24, 24]}>
              <Col xs={24} md={12}>
                <Card title="อัปโหลดรูปภาพ" size="small">
                  <Dragger {...uploadProps}>
                    <p className="ant-upload-drag-icon">
                      <InboxOutlined />
                    </p>
                    <p className="ant-upload-text">คลิกหรือลากไฟล์มาวางที่นี่</p>
                    <p className="ant-upload-hint">
                      รองรับไฟล์ .jpg, .jpeg, .png
                    </p>
                  </Dragger>
                  
                  {selectedFile && (
                    <div className="mt-4">
                      <Alert 
                        message={`เลือกไฟล์: ${selectedFile.name}`} 
                        type="info" 
                        showIcon 
                      />
                    </div>
                  )}
                  
                  <div className="mt-4">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Button 
                        type="primary" 
                        onClick={handleAnalyzeAgeGender}
                        loading={loading || uploadLoading}
                        disabled={!selectedFile}
                        icon={<UserOutlined />}
                        size="large"
                        block
                      >
                        วิเคราะห์อายุและเพศ
                      </Button>
                      
                      {uploadLoading && (
                        <div className="text-center">
                          <Progress percent={75} size="small" />
                          <Text type="secondary">กำลังวิเคราะห์...</Text>
                        </div>
                      )}
                    </Space>
                  </div>
                </Card>
              </Col>
              
              <Col xs={24} md={12}>
                <Card title="ตัวอย่างรูปภาพ" size="small">
                  {previewUrl ? (
                    <div className="text-center">
                      <Image 
                        src={previewUrl} 
                        alt="รูปภาพตัวอย่าง" 
                        style={{ maxWidth: '100%', maxHeight: '300px' }}
                      />
                    </div>
                  ) : (
                    <div className="text-center py-8 text-gray-400">
                      ยังไม่มีรูปภาพที่เลือก
                    </div>
                  )}
                </Card>
              </Col>
            </Row>
            
            {/* ผลลัพธ์การวิเคราะห์อายุและเพศ */}
            {(localAgeGenderResults && localAgeGenderResults.length > 0) ? (
              <div className="mt-4">
                <Divider>ผลลัพธ์การวิเคราะห์อายุและเพศ</Divider>
                
                {/* แสดงภาพที่มีกรอบใบหน้า */}
                {processedImageUrl && (
                  <Card className="mb-4">
                    <div className="text-center">
                      <Text strong>ภาพที่มีข้อมูลอายุและเพศ:</Text>
                      <div className="mt-2">
                        <Image 
                          src={processedImageUrl} 
                          alt="ภาพที่มีข้อมูลอายุและเพศ" 
                          style={{ maxWidth: '100%', maxHeight: '400px' }}
                        />
                      </div>
                      <div className="mt-2">
                        <Button 
                          size="small" 
                          onClick={() => setProcessedImageUrl(null)}
                        >
                          ซ่อนภาพที่มีข้อมูล
                        </Button>
                      </div>
                    </div>
                  </Card>
                )}
                
                <Card>
                  <Row gutter={[16, 16]} className="mb-4">
                    <Col span={8}>
                      <Statistic 
                        title="จำนวนใบหน้าที่พบ" 
                        value={localAgeGenderResults.length} 
                        prefix={<UserOutlined />}
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic 
                        title="อายุเฉลี่ย" 
                        value={Math.round(localAgeGenderResults.reduce((sum, r) => sum + r.age, 0) / localAgeGenderResults.length)} 
                        suffix="ปี"
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic 
                        title="เพศที่พบมากที่สุด" 
                        value={localAgeGenderResults.filter(r => r.gender.toLowerCase().includes('m')).length > localAgeGenderResults.length / 2 ? 'ชาย' : 'หญิง'}
                      />
                    </Col>
                  </Row>
                  
                  {localAgeGenderResults.map((result, index) => (
                    <Card key={index} size="small" className="mt-2" style={{ backgroundColor: '#f8f9fa' }}>
                      <Text strong>ใบหน้าที่ {index + 1}</Text>
                      <div className="mt-2">
                        <Tag color="blue" style={{ fontSize: '14px', padding: '4px 8px' }}>
                          อายุ: {result.age} ปี
                        </Tag>
                        <Tag 
                          color={result.gender.toLowerCase().includes('m') ? 'geekblue' : 'magenta'} 
                          style={{ fontSize: '14px', padding: '4px 8px' }}
                        >
                          เพศ: {result.gender}
                        </Tag>
                        <Tag color="green" style={{ fontSize: '14px', padding: '4px 8px' }}>
                          ความเชื่อมั่น: {result.gender_confidence.toFixed(1)}%
                        </Tag>
                      </div>
                      <div className="mt-2">
                        <Text type="secondary">
                          ตำแหน่ง: ({result.face_region.x}, {result.face_region.y}) ขนาด: {result.face_region.w}×{result.face_region.h}
                        </Text>
                      </div>
                    </Card>
                  ))}
                </Card>
              </div>
            ) : loading ? (
              <div className="mt-4">
                <Alert 
                  message="กำลังประมวลผล..." 
                  description="กรุณารอสักครู่" 
                  type="info" 
                  showIcon 
                />
              </div>
            ) : (
              <div className="mt-4">
                <Alert 
                  message="ยังไม่มีผลลัพธ์" 
                  description={`อัปโหลดรูปภาพและคลิก "วิเคราะห์อายุและเพศ" เพื่อเริ่มการวิเคราะห์`}
                  type="warning" 
                  showIcon 
                />
              </div>
            )}
            
            {error && (
              <Alert 
                message="เกิดข้อผิดพลาด" 
                description={error} 
                type="error" 
                showIcon 
                className="mt-4"
              />
            )}
          </Card>        </div>
      )
    },
    {
      key: 'cctv',
      label: (
        <span className="text-lg font-semibold">
          <CameraOutlined className="mr-2 text-blue-600" />
          📹 Real-time CCTV
        </span>
      ),
      children: (
        <div>
          <Card className="mb-4">
            <Title level={4}>กล้องวงจรปิดแบบ Real-time (CCTV Face Recognition)</Title>
            <Paragraph>
              ระบบตรวจจับและจดจำใบหน้าแบบ real-time สำหรับกล้องวงจรปิด โดยใช้ YOLOv11m 
              พร้อมการตรวจจับใบหน้าและจดจำบุคคลที่ลงทะเบียนไว้ในระบบ
            </Paragraph>
            
            <Alert
              message="ฟีเจอร์ Real-time CCTV"
              description={
                <div>
                  <p>• ตรวจจับใบหน้าแบบ real-time ด้วย YOLOv11m</p>
                  <p>• จดจำบุคคลที่ลงทะเบียนไว้ในระบบ</p>
                  <p>• แสดงภาพใบหน้าที่ตัดออกมาแบบ real-time</p>
                  <p>• วิเคราะห์สถิติการตรวจจับและจดจำ</p>
                </div>
              }
              type="info"
              showIcon
              className="mb-4"
            />
              <CCTVComponent />
          </Card>
        </div>
      )
    }
  ];return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <div className="bg-white shadow-lg border-b border-gray-200">
        <div className="max-w-6xl mx-auto px-4 py-16">
          <div className="text-center">
            <div className="mb-6">
              <div className="w-32 h-32 bg-gradient-to-r from-emerald-500 via-cyan-500 to-violet-500 rounded-full mx-auto flex items-center justify-center shadow-2xl">
                <EyeOutlined className="text-6xl text-white" />
              </div>
            </div>
            <Title level={1} className="mb-4 text-6xl font-bold text-gray-800">
              🧪 AI Testing Lab
            </Title>
            <Paragraph className="text-2xl text-gray-600 max-w-4xl mx-auto leading-relaxed">
              ✨ ทดสอบความสามารถของโมเดล Computer Vision AI แบบ Real-time<br/>
              <span className="text-lg text-gray-500">🚀 YOLOv11 Detection • 🛡️ Anti-Spoofing CNN • 🔍 Age/Gender Analysis</span>
            </Paragraph>{loading && (
              <div className="mt-6 max-w-md mx-auto">
                <Progress 
                  percent={Math.random() * 100} 
                  status="active" 
                  strokeColor={{ '0%': '#10b981', '50%': '#06b6d4', '100%': '#8b5cf6' }}
                  format={() => '🔄 Processing with AI...'}
                  size={8}
                />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-4 py-8">
        <Row gutter={[24, 24]}>
          <Col span={24}>
            <Card className="shadow-2xl border-0 rounded-2xl overflow-hidden bg-white/95 backdrop-blur-sm">
              <Tabs 
                activeKey={activeTab}
                onChange={setActiveTab}
                items={tabItems}
                size="large"
                type="card"
              />
            </Card>
          </Col>
        </Row>        {/* Real-time Face Detection Modal */}
        <RealTimeFaceDetection
          visible={realtimeModalVisible}
          onClose={() => setRealtimeModalVisible(false)}
        />

        {/* Real-time Face Detection Modal (Main) */}
        <RealTimeFaceDetection
          visible={realtimeOptimizedModalVisible}
          onClose={() => setRealtimeOptimizedModalVisible(false)}
        />

        {/* Real-time Anti-Spoofing Modal */}
        <RealTimeAntiSpoofing
          visible={antispoofingModalVisible}
          onClose={() => setAntispoofingModalVisible(false)}
        />
      </div>
    </div>
  );
}

export default function AITestingPage() {
  return (
    <App>
      <div className="min-h-screen bg-gray-50">
        <NavigationBar />
        <div className="pt-4">
          <AITestingPageContent />
        </div>
      </div>
    </App>
  );
}
