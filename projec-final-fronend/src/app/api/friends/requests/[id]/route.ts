import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import jwt from 'jsonwebtoken';

const prisma = new PrismaClient();

export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
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
      return NextResponse.json({ message: 'Invalid token' }, { status: 401 });
    }

    const { action } = await request.json();
    const resolvedParams = await params;
    const friendshipId = resolvedParams.id;

    if (!['accept', 'reject'].includes(action)) {
      return NextResponse.json({ message: 'Invalid action' }, { status: 400 });
    }

    // ตรวจสอบว่า friendship request นี้มีอยู่และเป็นของ user ที่ได้รับคำขอ
    const friendship = await prisma.friendship.findFirst({
      where: {
        id: friendshipId,
        friendId: decoded.userId,
        status: 'PENDING'
      }
    });

    if (!friendship) {
      return NextResponse.json({ 
        message: 'Friend request not found or already processed' 
      }, { status: 404 });
    }

    if (action === 'accept') {
      // อัปเดตสถานะเป็น ACCEPTED
      await prisma.friendship.update({
        where: { id: friendshipId },
        data: { status: 'ACCEPTED' }
      });

      // สร้าง notification สำหรับผู้ส่งคำขอ
      await prisma.notification.create({
        data: {
          userId: friendship.userId,
          senderId: decoded.userId,
          type: 'FRIEND_ACCEPTED',
          title: 'ยอมรับคำขอเป็นเพื่อน',
          message: 'ยอมรับคำขอเป็นเพื่อนของคุณ',
          data: { friendshipId: friendship.id }
        }
      });

      return NextResponse.json({ 
        success: true, 
        message: 'Friend request accepted' 
      });
    } else {
      // ลบ friendship request
      await prisma.friendship.delete({
        where: { id: friendshipId }
      });

      return NextResponse.json({ 
        success: true, 
        message: 'Friend request rejected' 
      });
    }

  } catch (error) {
    console.error('Error handling friend request:', error);
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}
