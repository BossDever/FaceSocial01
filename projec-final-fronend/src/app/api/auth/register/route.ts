import { NextRequest, NextResponse } from 'next/server';
import { hash } from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { prisma } from '@/lib/prisma';
import { getFaceApiUrl } from '@/lib/face-api';

// Helper functions for registration cache
const registrationCache = new Set<string>();
async function checkDuplicateRegistration(registrationId: string): Promise<boolean> {
  if (!registrationId) return false;
  return registrationCache.has(registrationId);
}
async function markRegistrationCompleted(registrationId: string): Promise<void> {
  if (registrationId) {
    registrationCache.add(registrationId);
    setTimeout(() => {
      registrationCache.delete(registrationId);
    }, 3600000);
  }
}

// Face Registration API call with timeout & retry
const callFaceRegistrationAPI = async (faceApiData: any, maxRetries = 3) => {
  let lastError = null;
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      console.log(`🔄 Attempt ${attempt}/${maxRetries} - Calling Face Registration API...`);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 15000);
      const faceApiResponse = await fetch(`${getFaceApiUrl()}/api/face-recognition/register-multiple`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(faceApiData),
        signal: controller.signal
      });
      clearTimeout(timeoutId);
      console.log(`📡 API Response Status (Attempt ${attempt}):`, faceApiResponse.status);
      if (!faceApiResponse.ok) {
        const errorText = await faceApiResponse.text();
        throw new Error(`HTTP ${faceApiResponse.status}: ${errorText}`);
      }
      const faceApiResult = await faceApiResponse.json();
      console.log(`✅ Face Registration API Success (Attempt ${attempt}):`, faceApiResult);
      return { success: true, result: faceApiResult };
    } catch (error) {
      lastError = error;
      console.error(`❌ Attempt ${attempt}/${maxRetries} failed:`, error);
      if (attempt < maxRetries) {
        const waitTime = Math.pow(2, attempt) * 1000;
        console.log(`⏳ Waiting ${waitTime}ms before retry...`);
        await new Promise(resolve => setTimeout(resolve, waitTime));
      }
    }
  }
  return { success: false, error: lastError };
};

export async function POST(request: NextRequest) {
  const startTime = Date.now();
  const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  
  console.log(`🚀 [${requestId}] Registration request started`);
  
  try {
    const body = await request.json();
    console.log(`📝 [${requestId}] Request body received:`, {
      firstName: body.firstName,
      lastName: body.lastName,
      email: body.email,
      username: body.username,
      hasPassword: !!body.password,
      hasFaceEmbedding: !!body.faceEmbedding,
      hasFaceImageBase64: !!body.faceImageBase64,
      registrationId: body.registrationId
    });
    
    const {
      firstName,
      lastName,
      email,
      username,
      password,
      phone,
      dateOfBirth,
      faceEmbedding,
      faceImageUrl,
      faceImageBase64,
      qualityScore,
      detectionConfidence,
      registrationId // เพิ่ม unique ID
    } = body;

    // Validation
    if (!firstName || !lastName || !email || !username || !password) {
      return NextResponse.json(
        { success: false, message: 'กรุณากรอกข้อมูลที่จำเป็นให้ครบถ้วน' },
        { status: 400 }
      );
    }
    // Face data validation - need either faceEmbedding or faceImageBase64
    if (!faceEmbedding && !faceImageBase64) {
      return NextResponse.json(
        { success: false, message: 'ไม่พบข้อมูลใบหน้า กรุณาสแกนใบหน้าใหม่' },
        { status: 400 }
      );
    }

    // ตรวจสอบ duplicate registration ID
    const existingRegistration = await checkDuplicateRegistration(registrationId);
    if (existingRegistration) {
      return NextResponse.json(
        { success: false, message: 'การลงทะเบียนนี้ดำเนินการแล้ว' },
        { status: 409 }
      );
    }

    let finalEmbedding = faceEmbedding;
    let extractedFromBase64 = false;

    // If we have base64 but no embedding, extract embedding from image
    if (faceImageBase64 && !faceEmbedding) {
      console.log('Extracting embedding from base64 image...');
      
      try {
        const extractResponse = await fetch(`${getFaceApiUrl()}/api/face-recognition/extract-embedding`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },          body: JSON.stringify({
            face_image_base64: faceImageBase64,
            model_name: 'facenet'
          })
        });

        if (!extractResponse.ok) {
          return NextResponse.json(
            { success: false, message: 'ไม่สามารถประมวลผลใบหน้าได้ กรุณาลองใหม่' },
            { status: 400 }
          );
        }

        const extractResult = await extractResponse.json();
        if (!extractResult.success || !extractResult.embedding) {
          return NextResponse.json(
            { success: false, message: 'ไม่พบใบหน้าในภาพ กรุณาถ่ายใหม่' },
            { status: 400 }
          );
        }

        finalEmbedding = extractResult.embedding;
        extractedFromBase64 = true;
        console.log(`Embedding extracted: ${finalEmbedding.length} dimensions`);
        
      } catch (error) {
        console.error('Embedding extraction error:', error);
        return NextResponse.json(
          { success: false, message: 'เกิดข้อผิดพลาดในการประมวลผลใบหน้า' },
          { status: 500 }
        );
      }
    }

    // Validate embedding dimensions (should be 512 for most models)
    if (!finalEmbedding || !Array.isArray(finalEmbedding) || finalEmbedding.length === 0) {
      return NextResponse.json(
        { success: false, message: 'ข้อมูลใบหน้าไม่ถูกต้อง' },
        { status: 400 }
      );
    }

    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return NextResponse.json(
        { success: false, message: 'รูปแบบอีเมลไม่ถูกต้อง' },
        { status: 400 }
      );
    }

    // Username validation
    const usernameRegex = /^[a-zA-Z0-9_]{3,}$/;
    if (!usernameRegex.test(username)) {
      return NextResponse.json(
        { success: false, message: 'ชื่อผู้ใช้ต้องมีอย่างน้อย 3 ตัวอักษร และใช้ได้เฉพาะ a-z, A-Z, 0-9, _' },
        { status: 400 }
      );
    }

    // Password validation
    if (password.length < 6) {
      return NextResponse.json(
        { success: false, message: 'รหัสผ่านต้องมีอย่างน้อย 6 ตัวอักษร' },
        { status: 400 }
      );
    }
    // Check if email or username already exists using Prisma
    const existingEmailUser = await prisma.user.findUnique({
      where: { email: email.toLowerCase() }
    });
    const existingUsernameUser = await prisma.user.findUnique({
      where: { username: username.toLowerCase() }
    });
    
    let existingPhoneUser = null;
    if (phone) {
      existingPhoneUser = await prisma.user.findFirst({
        where: { phone: phone }
      });
    }

    if (existingEmailUser) {
      return NextResponse.json(
        { success: false, message: 'อีเมลนี้ถูกใช้งานแล้ว' },
        { status: 409 }
      );
    }

    if (existingUsernameUser) {
      return NextResponse.json(
        { success: false, message: 'ชื่อผู้ใช้นี้ถูกใช้งานแล้ว' },
        { status: 409 }
      );
    }

    if (existingPhoneUser) {
      return NextResponse.json(
        { success: false, message: 'เบอร์โทรศัพท์นี้ถูกใช้งานแล้ว' },
        { status: 409 }
      );
    }    // Hash password
    const passwordHash = await hash(password, 12);
    
    console.log(`🔄 [${requestId}] Creating user and face embedding in database...`);
    
    // Create user and face embedding using Prisma transaction (fast operations only)
    const dbResult = await prisma.$transaction(async (tx) => {
      // Create user
      const newUser = await tx.user.create({
        data: {
          username: username.toLowerCase(),
          email: email.toLowerCase(),
          passwordHash,
          firstName,
          lastName,
          phone: phone || null,
          dateOfBirth: dateOfBirth ? new Date(dateOfBirth) : null,
          isActive: true,
          isVerified: false,
        }
      });
      
      // Create face embedding in database
      const newEmbedding = await tx.faceEmbedding.create({
        data: {
          userId: newUser.id,
          embeddingModel: extractedFromBase64 ? 'adaface' : 'facenet',
          embeddingData: finalEmbedding,
          imageUrl: faceImageUrl,
          faceImageData: faceImageBase64, // Store base64 for login use
          qualityScore: qualityScore || 0,
          detectionConfidence: detectionConfidence || 0,
          isPrimary: true,
        }
      });
      
      return { user: newUser, embedding: newEmbedding };
    }, {
      timeout: 10000 // 10 second timeout for database operations only
    });
    
    console.log(`✅ [${requestId}] Database records created successfully`);
    
    // Now call Face API outside of transaction (this can take longer)
    console.log(`🔄 [${requestId}] Calling Face API...`);
    
    // Prepare Face API data
    let cleanBase64 = faceImageBase64 || '';
    if (cleanBase64.includes(',')) {
      cleanBase64 = cleanBase64.split(',')[1];
    }
    
    const faceApiData = {
      full_name: `${firstName} ${lastName}`,
      employee_id: dbResult.user.id.toString(),
      department: "General User",
      position: "User",
      model_name: 'facenet', // Always use facenet for consistency
      images: [cleanBase64],
      metadata: {
        user_id: dbResult.user.id.toString(),
        username: dbResult.user.username,
        email: dbResult.user.email,
        full_name: `${firstName} ${lastName}`,
        quality_score: qualityScore || 0,
        detection_confidence: detectionConfidence || 0,
        registration_type: "web_signup",
        registration_date: new Date().toISOString(),
        registration_id: registrationId
      }
    };
      // Call Face API with retry logic (outside transaction)
    const apiResult = await callFaceRegistrationAPI(faceApiData);
    
    // Update face embedding with API result (separate transaction)
    console.log(`🔄 [${requestId}] Updating face embedding with API result...`);
    
    try {
      await prisma.faceEmbedding.update({
        where: { id: dbResult.embedding.id },
        data: {
          landmarkData: apiResult.success ? {
            api_person_id: apiResult.result?.person_id || null,
            api_registration_success: true,
            registration_date: new Date().toISOString()
          } : {
            api_registration_success: false,
            api_error: typeof apiResult.error === 'object' && apiResult.error !== null 
              ? (apiResult.error as any).message || 'Unknown API error'
              : String(apiResult.error || 'Unknown API error'),
            registration_date: new Date().toISOString()
          }
        }
      });
    } catch (updateError) {
      console.error(`❌ [${requestId}] Failed to update face embedding with API result:`, updateError);
      // Don't fail the registration if we can't update the API result
    }
    
    const result = { ...dbResult, apiResult };    
    // If Face API failed, we still consider registration successful but warn user
    if (!apiResult.success) {
      console.warn(`⚠️ [${requestId}] Face API registration failed but user created: ${apiResult.error}`);
    }
    
    // Generate JWT token for auto-login
    const token = jwt.sign(
      { userId: result.user.id, email: result.user.email },
      process.env.JWT_SECRET!,
      { expiresIn: '7d' }
    );
    
    // Mark registration as completed
    await markRegistrationCompleted(registrationId);
    
    const processingTime = Date.now() - startTime;
    console.log(`✅ [${requestId}] Registration completed successfully in ${processingTime}ms`);
      return NextResponse.json({
      success: true,
      message: apiResult.success 
        ? 'สมัครสมาชิกสำเร็จ กรุณาตรวจสอบอีเมลเพื่อยืนยันบัญชี'
        : 'สมัครสมาชิกสำเร็จ แต่การลงทะเบียนใบหน้าไม่สมบูรณ์ คุณสามารถเข้าใช้งานได้ปกติ',
      data: {
        token,
        user: {
          id: result.user.id,
          username: result.user.username,
          email: result.user.email,
          firstName: result.user.firstName,
          lastName: result.user.lastName,
          fullName: `${result.user.firstName} ${result.user.lastName}`,
          isActive: result.user.isActive,
          createdAt: result.user.createdAt
        },
        faceRegistered: true,
        qualityScore: qualityScore,
        apiRegistration: result.apiResult.success,
        apiWarning: !result.apiResult.success ? 'Face API registration incomplete' : undefined
      }
    });} catch (error: any) {
    const processingTime = Date.now() - startTime;
    console.error(`❌ [${requestId}] Registration failed after ${processingTime}ms:`, error);
    
    // Detailed error handling
    let errorMessage = 'เกิดข้อผิดพลาดของระบบ กรุณาลองใหม่อีกครั้ง';
    let statusCode = 500;
    
    if (error.code === 'P2002') {
      errorMessage = 'ข้อมูลนี้ถูกใช้งานแล้ว';
      statusCode = 409;
    } else if (error.message) {
      // If error has a message, use it but make it user-friendly
      if (error.message.includes('duplicate key')) {
        errorMessage = 'ข้อมูลนี้ถูกใช้งานแล้ว';
        statusCode = 409;
      } else if (error.message.includes('timeout')) {
        errorMessage = 'การเชื่อมต่อขาดหาย กรุณาลองใหม่';
        statusCode = 408;
      } else if (error.message.includes('connection')) {
        errorMessage = 'ไม่สามารถเชื่อมต่อฐานข้อมูลได้';
        statusCode = 503;
      } else {
        errorMessage = 'เกิดข้อผิดพลาดของระบบ กรุณาลองใหม่อีกครั้ง';
      }
    }
    
    console.log(`📋 [${requestId}] Sending error response: ${statusCode} - ${errorMessage}`);
    
    return NextResponse.json(
      { 
        success: false, 
        message: errorMessage,
        requestId: requestId,
        error: process.env.NODE_ENV === 'development' ? error.message : undefined
      },
      { status: statusCode }
    );
  }
}

// GET endpoint to check if email/username is available
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const email = searchParams.get('email');
    const username = searchParams.get('username');

    if (!email && !username) {
      return NextResponse.json(
        { success: false, message: 'กรุณาระบุ email หรือ username' },
        { status: 400 }
      );
    }

    let existingUser = null;
    if (email) {
      existingUser = await prisma.user.findUnique({
        where: { email: email.toLowerCase() }
      });
    } else if (username) {
      existingUser = await prisma.user.findUnique({
        where: { username: username.toLowerCase() }
      });
    }

    return NextResponse.json({
      success: true,
      available: !existingUser,
      message: existingUser 
        ? (email ? 'อีเมลนี้ถูกใช้งานแล้ว' : 'ชื่อผู้ใช้นี้ถูกใช้งานแล้ว')
        : 'สามารถใช้งานได้'
    });

  } catch (error) {
    console.error('Check availability error:', error);
    return NextResponse.json(
      { success: false, message: 'เกิดข้อผิดพลาดในการตรวจสอบ' },
      { status: 500 }
    );
  }
}
