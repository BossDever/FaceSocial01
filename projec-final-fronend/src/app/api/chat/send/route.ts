import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import * as jwt from 'jsonwebtoken';
import { sendToUser } from '../stream/route';

const prisma = new PrismaClient();

export async function POST(request: NextRequest) {
  try {
    // Get token from Authorization header
    const authHeader = request.headers.get('authorization');
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const token = authHeader.substring(7);
    const decoded = jwt.verify(token, process.env.JWT_SECRET!) as { userId: string };
    
    const { receiverId, content } = await request.json();

    if (!receiverId || !content) {
      return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
    }

    // Save message to database
    const savedMessage = await (prisma as any).message.create({
      data: {
        senderId: decoded.userId,
        receiverId: receiverId,
        content: content
      },      include: {
        sender: {
          select: {
            id: true,
            username: true,
            firstName: true,
            lastName: true,
            profileImageUrl: true
          }
        }
      }
    });    // Send real-time notification to receiver
    sendToUser(receiverId, {
      type: 'new-message',
      message: {
        id: savedMessage.id,
        sender_id: savedMessage.senderId,
        receiver_id: savedMessage.receiverId,
        content: savedMessage.content,
        created_at: savedMessage.createdAt,
        sender: {
          ...savedMessage.sender,
          profilePicture: savedMessage.sender.profileImageUrl
        }
      }
    });

    // Also send confirmation to sender
    sendToUser(decoded.userId, {
      type: 'message-sent',
      message: {
        id: savedMessage.id,
        sender_id: savedMessage.senderId,
        receiver_id: savedMessage.receiverId,
        content: savedMessage.content,
        created_at: savedMessage.createdAt,
        sender: {
          ...savedMessage.sender,
          profilePicture: savedMessage.sender.profileImageUrl
        }
      }
    });

    return NextResponse.json({
      id: savedMessage.id,
      sender_id: savedMessage.senderId,
      receiver_id: savedMessage.receiverId,
      content: savedMessage.content,
      created_at: savedMessage.createdAt
    });

  } catch (error) {
    console.error('Error sending message:', error);
    
    if (error instanceof jwt.JsonWebTokenError) {
      return NextResponse.json({ error: 'Invalid token' }, { status: 401 });
    }

    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
