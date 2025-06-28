import { NextRequest, NextResponse } from 'next/server';
import { compare } from 'bcryptjs';
import { v4 as uuidv4 } from 'uuid';
import { prisma } from '@/lib/prisma';

interface LoginRequestBody {
  email?: string;
  password?: string;
  username?: string;
  faceEmbedding?: number[];
  faceImageBase64?: string;
  loginMethod?: 'password' | 'face';
  method?: 'password' | 'face';
}

interface FaceAnalysisResult {
  success: boolean;
  faces?: Array<{
    detection_confidence: number;
    quality_score: number;
    has_identity: boolean;
    identity?: string;
    identity_name?: string;
    recognition_confidence: number;
    matches?: Array<{
      person_id: string;
      person_name: string;
      similarity: number;
      confidence: number;
      match_type: string;
    }>;
    best_match?: {
      person_id: string;
      person_name: string;
      confidence: number;
      similarity_score: number;
    };
  }>;
  message?: string;
  error?: string;
}

export async function POST(request: NextRequest) {
  try {
    const body: LoginRequestBody = await request.json();
    const { 
      email, 
      password, 
      username, 
      faceEmbedding, 
      faceImageBase64,
      loginMethod = 'password',
      method = 'password'
    } = body;

    // Determine actual login method
    const actualLoginMethod = loginMethod || method;
    let user = null;
    let similarity = 0;

    if (actualLoginMethod === 'password') {
      // Traditional password login
      if (!email && !username) {
        return NextResponse.json(
          { success: false, message: 'กรุณากรอกอีเมลหรือชื่อผู้ใช้' },
          { status: 400 }
        );
      }

      if (!password) {
        return NextResponse.json(
          { success: false, message: 'กรุณากรอกรหัสผ่าน' },
          { status: 400 }
        );
      }

      // Find user by email or username
      user = await prisma.user.findFirst({
        where: {
          OR: [
            { email: email },
            { username: username }
          ]
        }
      });

      if (!user) {
        return NextResponse.json(
          { success: false, message: 'ไม่พบผู้ใช้งาน' },
          { status: 401 }
        );
      }

      // Check password
      const isPasswordValid = await compare(password!, user.passwordHash);
      if (!isPasswordValid) {
        return NextResponse.json(
          { success: false, message: 'รหัสผ่านไม่ถูกต้อง' },
          { status: 401 }
        );
      }

    } else if (actualLoginMethod === 'face') {
      // Face recognition login using WORKING endpoint
      try {
        if (!faceImageBase64) {
          return NextResponse.json(
            { success: false, message: 'ไม่พบข้อมูลใบหน้าที่ถูกต้อง' },
            { status: 400 }
          );
        }

        console.log('🔍 Attempting face recognition login with working endpoint...');

        // Convert base64 to buffer for the working endpoint
        const imageBuffer = Buffer.from(faceImageBase64, 'base64');
        
        // Create FormData for the working endpoint
        const FormData = require('form-data');
        const formData = new FormData();
        formData.append('file', imageBuffer, {
          filename: 'login_face.jpg',
          contentType: 'image/jpeg'
        });
        formData.append('mode', 'full_analysis');
        formData.append('confidence_threshold', '0.5');
        formData.append('similarity_threshold', '0.6');
        formData.append('max_faces', '1');

        // Call the WORKING Face Analysis API endpoint
        const analysisResponse = await fetch('http://localhost:8080/api/face-analysis/analyze', {
          method: 'POST',
          body: formData,
          headers: formData.getHeaders()
        });

        if (!analysisResponse.ok) {
          console.error('❌ Face analysis API error:', analysisResponse.status);
          const errorText = await analysisResponse.text();
          console.error('Error details:', errorText);
          return NextResponse.json(
            { success: false, message: 'เกิดข้อผิดพลาดในการวิเคราะห์ใบหน้า' },
            { status: 500 }
          );
        }

        const analysisResult: FaceAnalysisResult = await analysisResponse.json();
        console.log('✅ Face analysis result:', {
          success: analysisResult.success,
          facesFound: analysisResult.faces?.length || 0,
          firstFace: analysisResult.faces?.[0] ? {
            hasIdentity: analysisResult.faces[0].has_identity,
            identity: analysisResult.faces[0].identity,
            confidence: analysisResult.faces[0].recognition_confidence,
            matchesCount: analysisResult.faces[0].matches?.length || 0
          } : null
        });

        // Check if face analysis was successful
        if (!analysisResult.success) {
          return NextResponse.json(
            { success: false, message: `การวิเคราะห์ใบหน้าล้มเหลว: ${analysisResult.error || analysisResult.message}` },
            { status: 500 }
          );
        }

        // Check if faces were detected
        if (!analysisResult.faces || analysisResult.faces.length === 0) {
          return NextResponse.json(
            { success: false, message: 'ไม่พบใบหน้าในภาพ กรุณาถ่ายภาพใหม่' },
            { status: 400 }
          );
        }

        const detectedFace = analysisResult.faces[0];
        
        // Check face quality
        if (detectedFace.detection_confidence < 0.8) {
          return NextResponse.json(
            { success: false, message: 'คุณภาพของใบหน้าไม่เพียงพอ กรุณาถ่ายภาพในที่ที่มีแสงดี' },
            { status: 400 }
          );
        }

        // Check if identity was found
        if (!detectedFace.has_identity || !detectedFace.identity) {
          return NextResponse.json(
            { success: false, message: 'ไม่พบใบหน้าที่จดทะเบียนในระบบ กรุณาลงทะเบียนก่อนเข้าสู่ระบบ' },
            { status: 401 }
          );
        }

        // Check recognition confidence
        if (detectedFace.recognition_confidence < 0.6) {
          return NextResponse.json(
            { success: false, message: `ความมั่นใจในการจดจำใบหน้าต่ำ (${(detectedFace.recognition_confidence * 100).toFixed(1)}%) กรุณาลองใหม่` },
            { status: 401 }
          );
        }

        // Get the recognized identity
        const recognizedIdentity = detectedFace.identity;
        similarity = detectedFace.recognition_confidence;

        console.log(`🎯 Face recognized as: ${recognizedIdentity} with confidence: ${(similarity * 100).toFixed(1)}%`);

        // Find the user in database by the recognized identity
        // The identity could be user ID, email, or username depending on how faces were registered
        user = await prisma.user.findFirst({
          where: {
            OR: [
              { id: recognizedIdentity },
              { email: recognizedIdentity },
              { username: recognizedIdentity }
            ]
          },
          include: {
            faceEmbeddings: {
              where: { isPrimary: true },
              take: 1
            }
          }
        });

        if (!user) {
          console.error(`❌ User not found for recognized identity: ${recognizedIdentity}`);
          
          // If direct match failed, try to find by person_id from matches
          if (detectedFace.matches && detectedFace.matches.length > 0) {
            const bestMatch = detectedFace.matches[0];
            user = await prisma.user.findUnique({
              where: { id: bestMatch.person_id },
              include: {
                faceEmbeddings: {
                  where: { isPrimary: true },
                  take: 1
                }
              }
            });
            
            if (user) {
              console.log(`✅ Found user via person_id: ${bestMatch.person_id}`);
              similarity = bestMatch.confidence;
            }
          }
          
          if (!user) {
            return NextResponse.json(
              { success: false, message: 'ไม่พบข้อมูลผู้ใช้ที่ตรงกับใบหน้าที่จดจำได้' },
              { status: 404 }
            );
          }
        }

        console.log(`✅ Successfully matched user: ${user.email} with confidence ${(similarity * 100).toFixed(1)}%`);

      } catch (error) {
        console.error('❌ Face recognition error:', error);
        return NextResponse.json(
          { success: false, message: 'เกิดข้อผิดพลาดในการจดจำใบหน้า กรุณาลองใหม่' },
          { status: 500 }
        );
      }
    } else {
      return NextResponse.json(
        { success: false, message: 'วิธีการเข้าสู่ระบบไม่ถูกต้อง' },
        { status: 400 }
      );
    }

    // Check if user account is active
    if (!user || !user.isActive) {
      return NextResponse.json(
        { success: false, message: 'บัญชีผู้ใช้ถูกระงับ กรุณาติดต่อผู้ดูแลระบบ' },
        { status: 403 }
      );
    }

    // Generate session token
    const token = `sess_${uuidv4()}_${Date.now()}`;

    // Update last login time
    await prisma.user.update({
      where: { id: user.id },
      data: { 
        lastLoginAt: new Date()
      }
    });

    console.log(`🎉 ${actualLoginMethod === 'face' ? 'Face' : 'Password'} login successful for user: ${user.email}`);

    // Prepare response data
    const responseData = {
      success: true,
      message: `เข้าสู่ระบบสำเร็จด้วย${actualLoginMethod === 'face' ? 'ใบหน้า' : 'รหัสผ่าน'}`,
      data: {
        token,
        user: {
          id: user.id,
          email: user.email,
          username: user.username,
          fullName: `${user.firstName} ${user.lastName}`,
          firstName: user.firstName,
          lastName: user.lastName,
          isActive: user.isActive,
          lastLoginAt: user.lastLoginAt,
          createdAt: user.createdAt
        },
        loginMethod: actualLoginMethod,
        ...(actualLoginMethod === 'face' && {
          similarity: similarity,
          confidencePercentage: `${(similarity * 100).toFixed(1)}%`
        })
      }
    };

    // Set secure cookie (optional)
    const response = NextResponse.json(responseData);
    response.cookies.set('auth-token', token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 60 * 60 * 24 * 7 // 7 days
    });

    return response;

  } catch (error) {
    console.error('❌ Login API error:', error);
    return NextResponse.json(
      { success: false, message: 'เกิดข้อผิดพลาดของเซิร์ฟเวอร์' },
      { status: 500 }
    );
  }
}
