import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import jwt from 'jsonwebtoken';
import { writeFile, mkdir } from 'fs/promises';
import path from 'path';

const prisma = new PrismaClient();

export async function POST(request: NextRequest) {
  try {
    console.log('🔄 Avatar upload API called');
    
    const authHeader = request.headers.get('authorization');
    if (!authHeader?.startsWith('Bearer ')) {
      console.log('❌ No authorization header');
      return NextResponse.json({ message: 'Unauthorized' }, { status: 401 });
    }

    const token = authHeader.substring(7);
    let decoded: { userId: string };
    
    try {
      decoded = jwt.verify(token, process.env.JWT_SECRET!) as { userId: string };
      console.log('✅ Token verified for user:', decoded.userId);
    } catch (jwtError) {
      console.log('❌ JWT verification failed:', jwtError);
      return NextResponse.json({ message: 'Invalid token' }, { status: 401 });
    }

    const formData = await request.formData();
    const file = formData.get('avatar') as File | null;

    if (!file) {
      console.log('❌ No file provided');
      return NextResponse.json(
        { message: 'No file provided' },
        { status: 400 }
      );
    }

    console.log('📁 File received:', file.name, file.type, file.size);

    // Validate file type
    if (!file.type.startsWith('image/')) {
      console.log('❌ Invalid file type:', file.type);
      return NextResponse.json(
        { message: 'Only image files are allowed' },
        { status: 400 }
      );
    }    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
      console.log('❌ File too large:', file.size, 'bytes');
      return NextResponse.json(
        { message: 'File size must be less than 10MB' },
        { status: 400 }
      );
    }

    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    // Create uploads directory if it doesn't exist
    const uploadsDir = path.join(process.cwd(), 'public', 'uploads', 'avatars');
    console.log('📂 Creating upload directory:', uploadsDir);
    await mkdir(uploadsDir, { recursive: true });

    // Generate unique filename
    const timestamp = Date.now();
    const fileExtension = path.extname(file.name);
    const filename = `${decoded.userId}-${timestamp}${fileExtension}`;
    const filepath = path.join(uploadsDir, filename);
    const profileImageUrl = `/uploads/avatars/${filename}`;

    console.log('💾 Saving file to:', filepath);
    console.log('🔗 Profile image URL:', profileImageUrl);

    // Save file
    await writeFile(filepath, buffer);    // Update user's profile picture in database
    console.log('🗄️ Updating database for user:', decoded.userId);
    const updatedUser = await prisma.user.update({
      where: { id: decoded.userId },
      data: { profileImageUrl },      select: {
        id: true,
        username: true,
        email: true,
        firstName: true,
        lastName: true,
        profileImageUrl: true,
        phone: true,
        dateOfBirth: true,
        isVerified: true,
        createdAt: true
      }
    });

    if (!updatedUser) {
      console.log('❌ User not found in database');
      return NextResponse.json({ message: 'User not found' }, { status: 404 });
    }
    
    // Format response
    const formattedUser = {
      id: updatedUser.id,
      username: updatedUser.username,
      email: updatedUser.email,
      firstName: updatedUser.firstName,
      lastName: updatedUser.lastName,
      fullName: `${updatedUser.firstName} ${updatedUser.lastName}`,
      profilePicture: updatedUser.profileImageUrl,
      phone: updatedUser.phone,
      dateOfBirth: updatedUser.dateOfBirth,
      isVerified: updatedUser.isVerified,
      createdAt: updatedUser.createdAt
    };

    console.log('✅ Avatar upload successful');
    return NextResponse.json({
      message: 'Profile picture updated successfully',
      user: formattedUser,
      profileImageUrl
    });

  } catch (error) {
    console.error('💥 Error uploading profile picture:', error);
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}
