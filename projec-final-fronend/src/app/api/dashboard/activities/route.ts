import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import jwt from 'jsonwebtoken';

const prisma = new PrismaClient();

export async function GET(request: NextRequest) {
  try {
    // Get token from header
    const authHeader = request.headers.get('authorization');
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return NextResponse.json({ error: 'Token required' }, { status: 401 });
    }

    const token = authHeader.substring(7);
    
    // Verify token
    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'your-secret-key') as { userId: string };
    const userId = decoded.userId;

    // Get recent activities
    const recentNotifications = await prisma.notification.findMany({
      where: { userId },
      orderBy: { createdAt: 'desc' },
      take: 5,
      include: {
        sender: {
          select: {
            id: true,
            firstName: true,
            lastName: true,
            username: true
          }
        }
      }
    });

    // Transform notifications to activities
    const activities = recentNotifications.map(notification => {
      let icon = '';
      let text = '';
      
      switch (notification.type) {
        case 'POST_LIKE':
          icon = 'heart';
          text = `${notification.sender?.firstName || 'ใครบางคน'} ถูกใจโพสต์ของคุณ`;
          break;
        case 'POST_COMMENT':
          icon = 'message';
          text = `${notification.sender?.firstName || 'ใครบางคน'} แสดงความคิดเห็นในโพสต์ของคุณ`;
          break;
        case 'FRIEND_REQUEST':
          icon = 'user-add';
          text = `${notification.sender?.firstName || 'ใครบางคน'} ส่งคำขอเป็นเพื่อน`;
          break;
        case 'FRIEND_ACCEPTED':
          icon = 'team';
          text = `${notification.sender?.firstName || 'ใครบางคน'} ยอมรับคำขอเป็นเพื่อน`;
          break;
        case 'FACE_TAG':
          icon = 'camera';
          text = `${notification.sender?.firstName || 'ใครบางคน'} แท็กคุณในรูปภาพ`;
          break;
        default:
          icon = 'bell';
          text = notification.message;
      }

      // Calculate time ago
      const now = new Date();
      const createdAt = new Date(notification.createdAt);
      const diffInMs = now.getTime() - createdAt.getTime();
      const diffInHours = Math.floor(diffInMs / (1000 * 60 * 60));
      const diffInDays = Math.floor(diffInHours / 24);
      
      let timeAgo = '';
      if (diffInDays > 0) {
        timeAgo = `${diffInDays} วันที่แล้ว`;
      } else if (diffInHours > 0) {
        timeAgo = `${diffInHours} ชั่วโมงที่แล้ว`;
      } else {
        const diffInMinutes = Math.floor(diffInMs / (1000 * 60));
        timeAgo = diffInMinutes > 0 ? `${diffInMinutes} นาทีที่แล้ว` : 'เมื่อสักครู่';
      }

      return {
        id: notification.id,
        icon,
        text,
        time: timeAgo,
        isRead: notification.isRead
      };
    });

    return NextResponse.json(activities);

  } catch (error) {
    console.error('Error fetching recent activities:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
