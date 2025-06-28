import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import jwt from 'jsonwebtoken';

const prisma = new PrismaClient();

export async function GET(request: NextRequest) {
  try {
    // Get token from header
    const authHeader = request.headers.get('authorization');
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return NextResponse.json({ error: 'Token required' }, { status: 401 });
    }

    const token = authHeader.substring(7);
    
    // Verify token
    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'your-secret-key') as { userId: string };
    const userId = decoded.userId;    // Get user's stats
    const [
      totalPosts,
      totalLikes,
      totalComments,
      totalFriends,
      totalMessages
    ] = await Promise.all([
      // Count user's posts
      prisma.post.count({
        where: { userId: userId }
      }),
      
      // Count likes on user's posts
      prisma.like.count({
        where: {
          post: {
            userId: userId
          }
        }
      }),
      
      // Count comments on user's posts
      prisma.comment.count({
        where: {
          post: {
            userId: userId
          }
        }
      }),
      
      // Count user's friends (both directions)
      prisma.friendship.count({
        where: {
          OR: [
            { userId: userId, status: 'ACCEPTED' },
            { friendId: userId, status: 'ACCEPTED' }
          ]
        }
      }),
      
      // Count messages sent by user
      prisma.message.count({
        where: { senderId: userId }
      })
    ]);

    return NextResponse.json({
      totalPosts,
      totalLikes,
      totalComments,
      totalFriends,
      totalMessages
    });

  } catch (error) {
    console.error('Error fetching dashboard stats:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
