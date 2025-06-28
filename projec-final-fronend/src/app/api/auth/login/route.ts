import { NextRequest, NextResponse } from 'next/server';
import { compare } from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { v4 as uuidv4 } from 'uuid';
import { prisma } from '@/lib/prisma';
import { getFaceApiUrl } from '@/lib/face-api';

interface LoginRequestBody {
  email?: string;
  password?: string;
  username?: string;
  userId?: string;
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
    console.log('🔍 Login API called');
    const body: LoginRequestBody = await request.json();
    console.log('📨 Login request body:', { 
      email: body.email, 
      username: body.username, 
      userId: body.userId,
      hasPassword: !!body.password,
      loginMethod: body.loginMethod || body.method 
    });
    
    const { 
      email, 
      password, 
      username, 
      userId,
      faceEmbedding, 
      faceImageBase64,
      loginMethod = 'password',
      method = 'password'
    } = body;

    // Determine actual login method
    const actualLoginMethod = loginMethod || method;
    let user = null;
    let similarity = 0;

    // Face Login with userId (จาก Face Recognition ที่ทำไว้แล้ว)
    if (actualLoginMethod === 'face' && userId && !faceImageBase64) {
      console.log('🎯 Face login with pre-verified userId:', userId);
      
      user = await prisma.user.findUnique({
        where: { id: userId },
        include: {
          faceEmbeddings: true
        }
      });

      if (!user) {
        return NextResponse.json(
          { success: false, message: 'ไม่พบผู้ใช้ในระบบ' },
          { status: 404 }
        );
      }

      similarity = 1.0; // 100% เพราะ Face Recognition ยืนยันแล้ว
      console.log('✅ Face login user found:', user.email);
    } else if (actualLoginMethod === 'password') {
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
      }      // Find user by email or username
      console.log('🔍 Looking for user with email/username:', email || username);
      user = await prisma.user.findFirst({
        where: {
          OR: [
            { email: email },
            { username: username }
          ]
        }
      });

      console.log('👤 User found:', user ? { id: user.id, username: user.username, email: user.email } : 'not found');

      if (!user) {
        console.log('❌ User not found');
        return NextResponse.json(
          { success: false, message: 'ไม่พบผู้ใช้งาน' },
          { status: 401 }
        );
      }

      // Check password
      console.log('🔐 Checking password...');
      const isPasswordValid = await compare(password!, user.passwordHash);
      console.log('🔐 Password valid:', isPasswordValid);
      
      if (!isPasswordValid) {
        console.log('❌ Invalid password');
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
        }        console.log('🔍 Attempting face recognition login...');

        // Use face recognition API to identify the person directly
        // Changed to use the base64 endpoint instead of form upload
        const recognitionData = {
          face_image_base64: faceImageBase64,
          model_name: 'facenet',
          top_k: 5,
          similarity_threshold: 0.5
        };

        const recognitionResponse = await fetch(`${getFaceApiUrl()}/api/face-recognition/recognize`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(recognitionData)
        });

        if (!recognitionResponse.ok) {
          console.error('❌ Face recognition failed:', recognitionResponse.status);
          const errorText = await recognitionResponse.text();
          console.error('Error details:', errorText);
          return NextResponse.json(
            { success: false, message: 'เกิดข้อผิดพลาดในการค้นหา' },
            { status: 500 }
          );
        }        const recognitionResult = await recognitionResponse.json();
        console.log('✅ Face recognition result:', {
          success: recognitionResult.success,
          matchesFound: recognitionResult.matches?.length || 0,
          bestMatch: recognitionResult.best_match
        });

        // Check if recognition was successful
        if (!recognitionResult.success) {
          return NextResponse.json(
            { success: false, message: 'เกิดข้อผิดพลาดในการจดจำใบหน้า' },
            { status: 400 }
          );
        }

        // Check if matches were found
        if (!recognitionResult.matches || recognitionResult.matches.length === 0) {
          return NextResponse.json(
            { success: false, message: 'ไม่พบใบหน้าที่ตรงกันในระบบ กรุณาลงทะเบียนก่อนเข้าสู่ระบบ' },
            { status: 401 }
          );
        }

        // Get the best match
        const bestMatch = recognitionResult.best_match || recognitionResult.matches[0];
        
        if (!bestMatch || !bestMatch.person_id) {
          return NextResponse.json(
            { success: false, message: 'ไม่พบใบหน้าที่ตรงกันในระบบ กรุณาลงทะเบียนก่อนเข้าสู่ระบบ' },
            { status: 401 }
          );
        }

        // Get the matched user ID and similarity
        const matchedUserId = bestMatch.person_id;
        similarity = bestMatch.confidence || bestMatch.similarity || 0;console.log(`🎯 Face matched with user: ${matchedUserId} with similarity: ${(similarity * 100).toFixed(1)}%`);

        // Find the user in database by the matched user ID
        user = await prisma.user.findFirst({
          where: {
            OR: [
              { id: matchedUserId },
              { email: matchedUserId },
              { username: matchedUserId }
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
          console.error(`❌ User not found for matched ID: ${matchedUserId}`);
          return NextResponse.json(
            { success: false, message: 'ไม่พบข้อมูลผู้ใช้ที่ตรงกับใบหน้าที่จดจำได้' },
            { status: 404 }
          );
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
    }    // Generate JWT token
    const token = jwt.sign(
      { userId: user.id, email: user.email },
      process.env.JWT_SECRET!,
      { expiresIn: '7d' }
    );

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
