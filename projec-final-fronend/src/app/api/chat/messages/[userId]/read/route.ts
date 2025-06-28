import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import jwt from 'jsonwebtoken';

const prisma = new PrismaClient();

// PUT - Mark messages as read
export async function PUT(
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
    const resolvedParams = await params;
    const senderId = resolvedParams.userId;// Update last_read_at for the user in the conversation
    await prisma.$queryRaw`
      UPDATE conversation_participants 
      SET last_read_at = NOW()
      WHERE user_id = ${decoded.userId}::uuid 
        AND conversation_id IN (
          SELECT c.id
          FROM conversations c
          JOIN conversation_participants cp1 ON c.id = cp1.conversation_id 
            AND cp1.user_id = ${decoded.userId}::uuid
          JOIN conversation_participants cp2 ON c.id = cp2.conversation_id 
            AND cp2.user_id = ${senderId}::uuid
          WHERE c.type = 'direct'
        )
    `;

    return NextResponse.json({ message: 'Messages marked as read' });
  } catch (error) {
    console.error('Error marking messages as read:', error);
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}
