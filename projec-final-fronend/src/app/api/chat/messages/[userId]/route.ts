import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import * as jwt from 'jsonwebtoken';

const prisma = new PrismaClient();

// GET - ดึงข้อความระหว่าง 2 คน
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
    let decoded: { userId: string };
    
    try {
      decoded = jwt.verify(token, process.env.JWT_SECRET!) as { userId: string };
    } catch (jwtError) {
      return NextResponse.json({ message: 'Invalid token' }, { status: 401 });
    }

    const { userId: otherUserId } = await params;    // ดึงข้อความระหว่าง 2 คน
    const messages = await (prisma as any).message.findMany({
      where: {
        OR: [
          {
            senderId: decoded.userId,
            receiverId: otherUserId
          },
          {
            senderId: otherUserId,
            receiverId: decoded.userId
          }
        ]
      },
      orderBy: {
        createdAt: 'asc'
      },      include: {
        sender: {
          select: {
            id: true,
            username: true,
            firstName: true,
            lastName: true,
            profileImageUrl: true
          }
        },
        receiver: {
          select: {
            id: true,
            username: true,
            firstName: true,
            lastName: true,
            profileImageUrl: true
          }
        }
      }
    });    // Transform messages to match frontend interface
    const transformedMessages = messages.map((msg: any) => ({
      id: msg.id,
      sender_id: msg.senderId,
      receiver_id: msg.receiverId,
      content: msg.content,
      created_at: msg.createdAt,
      sender: {
        ...msg.sender,
        profilePicture: msg.sender.profileImageUrl
      },
      receiver: {
        ...msg.receiver,
        profilePicture: msg.receiver.profileImageUrl
      }
    }));

    // อัพเดทสถานะว่าอ่านแล้ว
    await (prisma as any).message.updateMany({
      where: {
        senderId: otherUserId,
        receiverId: decoded.userId,
        read: false
      },
      data: {
        read: true
      }
    });

    return NextResponse.json(transformedMessages);
  } catch (error) {
    console.error('Error fetching messages:', error);
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}
