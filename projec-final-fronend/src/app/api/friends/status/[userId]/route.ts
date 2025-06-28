import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import jwt from 'jsonwebtoken';

const prisma = new PrismaClient();

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ userId: string }> }
) {
  try {
    const authHeader = request.headers.get('authorization');
    if (!authHeader?.startsWith('Bearer ')) {
      return NextResponse.json({ message: 'Unauthorized' }, { status: 401 });
    }

    const token = authHeader.substring(7);
    const decoded = jwt.verify(token, process.env.JWT_SECRET!) as { userId: string };
    const currentUserId = decoded.userId;
    const resolvedParams = await params;
    const targetUserId = resolvedParams.userId;// Check friendship status
    const friendship = await prisma.$queryRawUnsafe(`
      SELECT status, user_id, friend_id
      FROM friendships      WHERE (user_id = $1 AND friend_id = $2)
         OR (user_id = $2 AND friend_id = $1)
      LIMIT 1
    `, currentUserId, targetUserId) as { status: string; user_id: string; friend_id: string }[];if (!friendship || friendship.length === 0) {
      return NextResponse.json({
        status: 'none',
        isRequestSent: false,
        isRequestReceived: false
      });
    }

    const friendshipRecord = friendship[0];
    const status = {
      status: friendshipRecord.status.toLowerCase(),
      isRequestSent: friendshipRecord.user_id === currentUserId && friendshipRecord.status === 'PENDING',
      isRequestReceived: friendshipRecord.friend_id === currentUserId && friendshipRecord.status === 'PENDING'
    };

    return NextResponse.json(status);

  } catch (error) {
    console.error('Error fetching friendship status:', error);
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}
