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

    const resolvedParams = await params;
    const notificationId = resolvedParams.id;

    // Mark notification as read
    const notification = await prisma.notification.update({
      where: {
        id: notificationId,
        userId: decoded.userId // Ensure user can only update their own notifications
      },
      data: {
        isRead: true
      }
    });

    return NextResponse.json({ success: true, notification });

  } catch (error) {
    console.error('Error marking notification as read:', error);
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}
