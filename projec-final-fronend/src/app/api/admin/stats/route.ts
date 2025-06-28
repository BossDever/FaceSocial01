import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import { verifyAdminAccess } from '../utils/auth';

const prisma = new PrismaClient();

export async function GET(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization');
    
    const authResult = await verifyAdminAccess(authHeader);
    if (!authResult.success) {
      return NextResponse.json(
        { message: authResult.error },
        { status: authResult.error === 'Unauthorized' ? 401 : 403 }
      );
    }

    // Get system statistics
    const [
      totalUsers,
      activeUsers,
      totalPosts,
      totalMessages,
      onlineUsers
    ] = await Promise.all([
      prisma.user.count(),
      prisma.user.count({ where: { isActive: true } }),
      (prisma as any).post.count().catch(() => 0), // Handle if posts table doesn't exist
      (prisma as any).message.count().catch(() => 0), // Handle if message table doesn't exist
      prisma.user.count({ where: { lastLoginAt: { gte: new Date(Date.now() - 24 * 60 * 60 * 1000) } } }) // Users active in last 24h
    ]);

    return NextResponse.json({
      totalUsers,
      activeUsers,
      totalPosts,
      totalMessages,
      onlineUsers
    });

  } catch (error) {
    console.error('Error fetching admin stats:', error);
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}
