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
    const decoded = jwt.verify(token, process.env.JWT_SECRET!) as { userId: string };    const unreadCount = await prisma.$queryRawUnsafe(`
      SELECT COUNT(*)::int as count
      FROM notifications 
      WHERE user_id = $1 
      AND is_read = false
    `, decoded.userId) as { count: number }[];

    return NextResponse.json({ 
      unreadCount: unreadCount[0]?.count || 0 
    });

  } catch (error) {
    console.error('Error fetching unread count:', error);
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}
