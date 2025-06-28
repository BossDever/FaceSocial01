import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import jwt from 'jsonwebtoken';

const prisma = new PrismaClient();

// POST - Send a new message
export async function POST(request: NextRequest) {
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
    
    const body = await request.json();
    
    // Support both senderId (from Socket.IO) and current user (from frontend)
    const senderId = body.senderId || decoded.userId;
    const { receiverId, content } = body;

    if (!content?.trim() || !receiverId) {
      return NextResponse.json({ 
        message: 'Content and receiver ID are required' 
      }, { status: 400 });
    }

    // Check if receiver exists
    const receiver = await prisma.$queryRawUnsafe(`
      SELECT id FROM users WHERE id = $1
    `, receiverId) as { id: string }[];

    if (!receiver || receiver.length === 0) {
      return NextResponse.json({ message: 'Receiver not found' }, { status: 404 });
    }

    // Send message directly (no conversations table needed)
    const result = await prisma.$queryRawUnsafe(`
      INSERT INTO messages (id, sender_id, receiver_id, content, read, created_at, updated_at)
      VALUES (gen_random_uuid()::text, $1, $2, $3, false, NOW(), NOW())
      RETURNING id, sender_id, receiver_id, content, read, created_at
    `, senderId, receiverId, content.trim()) as any[];

    const newMessage = result[0];

    // Get sender info for response
    const senderInfo = await prisma.$queryRawUnsafe(`
      SELECT username, first_name, last_name, profile_image_url
      FROM users WHERE id = $1
    `, senderId) as any[];

    // Format response to match frontend expectations
    const response = {
      id: newMessage.id,
      content: newMessage.content,
      sender_id: newMessage.sender_id,
      receiver_id: newMessage.receiver_id,
      read: newMessage.read,
      created_at: newMessage.created_at,
      sender: {
        id: newMessage.sender_id,
        username: senderInfo[0]?.username || '',
        firstName: senderInfo[0]?.first_name || '',
        lastName: senderInfo[0]?.last_name || '',
        profilePicture: senderInfo[0]?.profile_image_url || null
      }
    };

    return NextResponse.json(response, { status: 201 });
  } catch (error) {
    console.error('Error sending message:', error);
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}
