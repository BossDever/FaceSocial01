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
    const decoded = jwt.verify(token, process.env.JWT_SECRET!) as { userId: string };
    const userId = decoded.userId;

    // Get unique conversations from messages table
    const conversations = await prisma.$queryRaw`
      WITH conversation_users AS (
        SELECT DISTINCT
          CASE 
            WHEN sender_id = ${userId}::uuid THEN receiver_id
            ELSE sender_id
          END as other_user_id
        FROM messages
        WHERE sender_id = ${userId}::uuid OR receiver_id = ${userId}::uuid
      ),
      latest_messages AS (
        SELECT 
          cu.other_user_id,
          m.content as last_message_content,
          m.created_at as last_message_time,
          ROW_NUMBER() OVER (PARTITION BY cu.other_user_id ORDER BY m.created_at DESC) as rn
        FROM conversation_users cu
        JOIN messages m ON (
          (m.sender_id = ${userId}::uuid AND m.receiver_id = cu.other_user_id) OR 
          (m.sender_id = cu.other_user_id AND m.receiver_id = ${userId}::uuid)
        )
      ),
      unread_counts AS (
        SELECT 
          sender_id as other_user_id,
          COUNT(*) as unread_count
        FROM messages
        WHERE receiver_id = ${userId}::uuid AND read = false
        GROUP BY sender_id
      )
      SELECT 
        u.id as user_id,
        u.username,
        u.first_name,
        u.last_name,
        u.profile_image_url,
        u.is_online,
        u.last_seen,
        lm.last_message_content,
        lm.last_message_time,
        COALESCE(uc.unread_count, 0) as unread_count
      FROM conversation_users cu
      JOIN users u ON cu.other_user_id = u.id
      LEFT JOIN latest_messages lm ON cu.other_user_id = lm.other_user_id AND lm.rn = 1
      LEFT JOIN unread_counts uc ON cu.other_user_id = uc.other_user_id
      ORDER BY COALESCE(lm.last_message_time, '1970-01-01'::timestamp) DESC
    `;

    // Transform the result
    const formattedConversations = (conversations as any[]).map((conv: any) => ({
      id: `conv-${conv.user_id}`,
      user: {
        id: conv.user_id,
        username: conv.username,
        firstName: conv.first_name,
        lastName: conv.last_name,
        profilePicture: conv.profile_image_url,
        isOnline: conv.is_online,
        lastSeen: conv.last_seen
      },
      lastMessage: conv.last_message_content ? {
        content: conv.last_message_content,
        createdAt: conv.last_message_time
      } : undefined,
      unreadCount: parseInt(conv.unread_count) || 0
    }));

    return NextResponse.json(formattedConversations);

  } catch (error) {
    console.error('Error fetching conversations:', error);
    return NextResponse.json(
      { message: 'Internal server error', error: (error as Error).message },
      { status: 500 }
    );
  }
}
