import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import jwt from 'jsonwebtoken';

const prisma = new PrismaClient();

export async function POST(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization');
    if (!authHeader?.startsWith('Bearer ')) {
      return NextResponse.json({ message: 'Unauthorized' }, { status: 401 });
    }

    const token = authHeader.substring(7);
    const decoded = jwt.verify(token, process.env.JWT_SECRET!) as { userId: string };
    
    const { postId, tags } = await request.json();

    if (!postId || !Array.isArray(tags)) {
      return NextResponse.json({ message: 'Post ID and tags array are required' }, { status: 400 });
    }

    // ตรวจสอบว่าโพสต์เป็นของ user นี้หรือไม่
    const post = await prisma.post.findFirst({
      where: { 
        id: postId,
        userId: decoded.userId 
      }
    });

    if (!post) {
      return NextResponse.json({ message: 'Post not found or unauthorized' }, { status: 404 });
    }

    // ลบ face tags เก่าของโพสต์นี้
    await prisma.faceTag.deleteMany({
      where: { postId }
    });

    // สร้าง face tags ใหม่
    const faceTagsData = tags.map((tag: any) => ({
      postId,
      taggedUserId: tag.userId,
      taggerUserId: decoded.userId,
      x: tag.x || 0.5, // ตำแหน่ง X (0-1)
      y: tag.y || 0.5, // ตำแหน่ง Y (0-1)
      width: tag.width || 0.1, // ความกว้าง bounding box
      height: tag.height || 0.1, // ความสูง bounding box
      confidence: tag.confidence || 0.8,
      isConfirmed: false // รอการยืนยันจากผู้ถูกแท็ก
    }));

    if (faceTagsData.length > 0) {
      await prisma.faceTag.createMany({
        data: faceTagsData
      });

      // ส่งการแจ้งเตือนให้ผู้ถูกแท็ก
      const notifications = tags.map((tag: any) => ({
        userId: tag.userId,
        senderId: decoded.userId,
        type: 'FACE_TAG' as const,
        title: 'คุณถูกแท็กในรูปภาพ',
        message: `คุณถูกแท็กในโพสต์ของ ${post.userId}`,
        data: { postId, tagId: tag.userId }
      }));

      await prisma.notification.createMany({
        data: notifications
      });
    }

    return NextResponse.json({ 
      success: true, 
      message: 'Face tags created successfully',
      tagsCount: faceTagsData.length
    });

  } catch (error) {
    console.error('Error creating face tags:', error);
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}
