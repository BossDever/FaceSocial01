import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import jwt from 'jsonwebtoken';

const prisma = new PrismaClient();

export async function GET(request: NextRequest) {
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

    const { searchParams } = new URL(request.url);
    const limit = parseInt(searchParams.get('limit') || '20');
    const unreadOnly = searchParams.get('unread') === 'true';    let notifications;
    if (unreadOnly) {
      notifications = await prisma.$queryRawUnsafe(`
        SELECT 
          n.id,
          n.type,
          n.title,
          n.message,
          n.data,
          n.is_read as "isRead",
          n.created_at as "createdAt",
          json_build_object(
            'id', s.id,
            'username', s.username,
            'firstName', s.first_name,
            'lastName', s.last_name,
            'profileImageUrl', s.profile_image_url
          ) as sender
        FROM notifications n
        LEFT JOIN users s ON n.sender_id = s.id
        WHERE n.user_id = $1 AND n.is_read = false
        ORDER BY n.created_at DESC
        LIMIT $2
      `, decoded.userId, limit) as any[];
    } else {
      notifications = await prisma.$queryRawUnsafe(`
        SELECT 
          n.id,
          n.type,
          n.title,
          n.message,
          n.data,
          n.is_read as "isRead",
          n.created_at as "createdAt",
          json_build_object(
            'id', s.id,
            'username', s.username,
            'firstName', s.first_name,
            'lastName', s.last_name,
            'profileImageUrl', s.profile_image_url
          ) as sender
        FROM notifications n
        LEFT JOIN users s ON n.sender_id = s.id
        WHERE n.user_id = $1
        ORDER BY n.created_at DESC
        LIMIT $2
      `, decoded.userId, limit) as any[];
    }

    return NextResponse.json(notifications);

  } catch (error) {
    console.error('Error fetching notifications:', error);
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function PUT(request: NextRequest) {
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

    const { notificationIds, markAsRead } = await request.json();

    if (!notificationIds || !Array.isArray(notificationIds)) {
      return NextResponse.json({ message: 'Invalid notification IDs' }, { status: 400 });
    }    // อัปเดตสถานะการอ่าน
    await prisma.$executeRawUnsafe(`
      UPDATE notifications 
      SET is_read = $1
      WHERE id = ANY($2::uuid[])
      AND user_id = $3
    `, markAsRead !== false, notificationIds, decoded.userId);

    return NextResponse.json({ 
      success: true, 
      message: `Notifications marked as ${markAsRead !== false ? 'read' : 'unread'}` 
    });

  } catch (error) {
    console.error('Error updating notifications:', error);
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}
