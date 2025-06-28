import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import jwt from 'jsonwebtoken';
import { writeFile, mkdir } from 'fs/promises';
import path from 'path';

const prisma = new PrismaClient();

export async function POST(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization');
    if (!authHeader?.startsWith('Bearer ')) {
      return NextResponse.json({ message: 'Unauthorized' }, { status: 401 });
    }

    const token = authHeader.substring(7);
    let decoded: { userId: string };
      try {
      decoded = jwt.verify(token, process.env.JWT_SECRET!) as { userId: string };
    } catch (jwtError) {
      console.error('JWT Error:', jwtError);
      return NextResponse.json({ message: 'Invalid token' }, { status: 401 });
    }

    // ตรวจสอบ Content-Type เพื่อจัดการ FormData หรือ JSON
    const contentType = request.headers.get('content-type') || '';
    let content: string = '';
    let image: File | null = null;
    let location: string | null = null;
    let isPublic: boolean = true;

    if (contentType.includes('multipart/form-data')) {
      // Handle FormData (for file uploads)
      const formData = await request.formData();
      content = formData.get('content') as string || '';
      image = formData.get('image') as File | null;
      location = formData.get('location') as string | null;
      isPublic = formData.get('isPublic') !== 'false';
    } else {
      // Handle JSON (for text-only posts)
      const jsonData = await request.json();
      content = jsonData.content || '';
      location = jsonData.location || null;
      isPublic = jsonData.isPublic !== false;
    }

    if (!content?.trim() && !image) {
      return NextResponse.json({ message: 'Content or image is required' }, { status: 400 });
    }

    let imageUrl: string | null = null;
    let detectedFaces: any[] = [];

    // อัปโหลดรูปถ้ามี
    if (image) {
      // Validate file type
      if (!image.type.startsWith('image/')) {
        return NextResponse.json({ message: 'Only image files are allowed' }, { status: 400 });
      }

      // Validate file size (10MB max)
      if (image.size > 10 * 1024 * 1024) {
        return NextResponse.json({ message: 'File size must be less than 10MB' }, { status: 400 });
      }

      const bytes = await image.arrayBuffer();
      const buffer = Buffer.from(bytes);

      // Create uploads directory
      const uploadsDir = path.join(process.cwd(), 'public', 'uploads', 'posts');
      await mkdir(uploadsDir, { recursive: true });

      // Generate unique filename
      const timestamp = Date.now();
      const fileExtension = path.extname(image.name);
      const filename = `${decoded.userId}-${timestamp}${fileExtension}`;
      const filepath = path.join(uploadsDir, filename);
      imageUrl = `/uploads/posts/${filename}`;

      // Save file
      await writeFile(filepath, buffer);      // เรียก Face Recognition API เพื่อตรวจจับใบหน้า
      try {
        const faceApiUrl = process.env.FACE_API_URL || 'http://host.docker.internal:8080';
        
        // Convert buffer to base64
        const base64Image = buffer.toString('base64');
        
        // ใช้ Face Detection API
        const faceResponse = await fetch(`${faceApiUrl}/api/face-detection/detect-base64`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            image_base64: base64Image,
            model_name: 'auto',
            conf_threshold: 0.5,
            max_faces: 10,
            return_landmarks: true
          })
        });

        if (faceResponse.ok) {
          const faceResult = await faceResponse.json();
          console.log('🔍 Face detection result:', faceResult);
          
          if (faceResult.success && faceResult.faces && faceResult.faces.length > 0) {
            detectedFaces = faceResult.faces;
            console.log(`🔍 Detected ${detectedFaces.length} faces in image`);
            
            // สำหรับแต่ละใบหน้าที่ตรวจพบ ลองจดจำว่าเป็นใครในระบบ
            for (let i = 0; i < detectedFaces.length; i++) {
              const face = detectedFaces[i];
              try {
                // ใช้ Face Recognition API เพื่อจดจำใบหน้า
                const recognitionResponse = await fetch(`${faceApiUrl}/api/face-recognition/recognize`, {
                  method: 'POST',
                  headers: {
                    'Content-Type': 'application/json'
                  },
                  body: JSON.stringify({
                    face_image_base64: base64Image,
                    model_name: 'facenet',
                    similarity_threshold: 0.7,
                    top_k: 3
                  })
                });

                if (recognitionResponse.ok) {
                  const recognitionResult = await recognitionResponse.json();
                  if (recognitionResult.success && recognitionResult.matches && recognitionResult.matches.length > 0) {
                    // เพิ่มข้อมูลการจดจำเข้าไปใน face object
                    detectedFaces[i].recognizedUsers = recognitionResult.matches.map((match: any) => ({
                      person_id: match.person_id,
                      person_name: match.person_name,
                      similarity: match.similarity,
                      confidence: match.confidence
                    }));
                  }
                }
              } catch (recognitionError) {
                console.error('Face recognition error for face', i, ':', recognitionError);
              }
            }
          }
        }
      } catch (faceError) {
        console.error('Face detection error:', faceError);
        // ไม่ให้ error ในการ detect faces หยุดการสร้างโพสต์
      }
    }

    // สร้างโพสต์
    const post = await prisma.post.create({
      data: {
        userId: decoded.userId,
        content: content?.trim() || '',
        imageUrl,
        location,
        isPublic
      },
      include: {
        user: {
          select: {
            id: true,
            username: true,
            firstName: true,
            lastName: true,
            profileImageUrl: true
          }
        },
        likes: true,
        comments: {
          include: {
            user: {
              select: {
                id: true,
                username: true,
                firstName: true,
                lastName: true,
                profileImageUrl: true
              }
            }
          }
        }
      }
    });

    return NextResponse.json({ 
      success: true, 
      post, 
      detectedFaces: detectedFaces.length > 0 ? detectedFaces : undefined
    });

  } catch (error) {
    console.error('Error creating post:', error);
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}
