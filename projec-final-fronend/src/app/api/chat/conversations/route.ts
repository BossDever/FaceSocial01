import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import * as jwt from 'jsonwebtoken';

const prisma = new PrismaClient();

export async function GET(request: NextRequest) {
  try {
    // Get token from Authorization header
    const authHeader = request.headers.get('authorization');
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return NextResponse.json({ error: 'ไม่พบ token การตรวจสอบสิทธิ์' }, { status: 401 });
    }

    const token = authHeader.substring(7);
    const decoded = jwt.verify(token, process.env.JWT_SECRET!) as { userId: string };
    const currentUserId = decoded.userId;    // Get all messages where the current user is sender or receiver
    const messages = await (prisma as any).message.findMany({
      where: {
        OR: [
          { senderId: currentUserId },
          { receiverId: currentUserId }
        ]
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
      },
      orderBy: {
        createdAt: 'desc'
      }
    });

    // Group messages by conversation (other user)
    const conversationsMap = new Map();

    for (const message of messages) {
      const otherUser = message.senderId === currentUserId ? message.receiver : message.sender;
      const conversationKey = otherUser.id;

      if (!conversationsMap.has(conversationKey)) {
        // Transform message to match frontend interface
        const transformedMessage = {
          id: message.id,
          sender_id: message.senderId,
          receiver_id: message.receiverId,
          content: message.content,
          created_at: message.createdAt
        };        conversationsMap.set(conversationKey, {
          id: conversationKey,
          otherUser: {
            ...otherUser,
            profilePicture: otherUser.profileImageUrl, // Map to expected field name
            isOnline: false, // TODO: Implement real online status
            lastSeen: null   // TODO: Implement last seen
          },
          lastMessage: transformedMessage,
          unreadCount: 0, // TODO: Implement unread count logic
          messages: [transformedMessage]
        });
      }
    }

    // Convert map to array
    const conversations = Array.from(conversationsMap.values());

    return NextResponse.json(conversations);

  } catch (error) {
    console.error('Error fetching conversations:', error);
    
    if (error instanceof jwt.JsonWebTokenError) {
      return NextResponse.json({ error: 'Token ไม่ถูกต้อง' }, { status: 401 });
    }

    return NextResponse.json(
      { error: 'เกิดข้อผิดพลาดในการดึงข้อมูลการสนทนา' },
      { status: 500 }
    );
  }
}
