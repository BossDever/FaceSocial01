import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';

export async function GET() {
  try {
    // Get basic user info (without sensitive data)
    const users = await prisma.user.findMany({
      select: {
        id: true,
        email: true,
        username: true,
        firstName: true,
        lastName: true,
        isActive: true,
        createdAt: true,
        lastLoginAt: true,
        faceEmbeddings: {
          select: {
            id: true,
            isPrimary: true,
            createdAt: true
          }
        }
      },
      orderBy: {
        createdAt: 'desc'
      }
    });

    // Get total counts
    const stats = {
      totalUsers: users.length,
      activeUsers: users.filter(u => u.isActive).length,
      usersWithFaceEmbeddings: users.filter(u => u.faceEmbeddings.length > 0).length,
      recentLogins: users.filter(u => u.lastLoginAt && 
        new Date(u.lastLoginAt) > new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)).length
    };

    return NextResponse.json({
      success: true,
      data: {
        users: users.map(user => ({
          id: user.id,
          email: user.email,
          username: user.username,
          fullName: `${user.firstName} ${user.lastName}`,
          isActive: user.isActive,
          createdAt: user.createdAt,
          lastLoginAt: user.lastLoginAt,
          hasFaceEmbeddings: user.faceEmbeddings.length > 0,
          faceEmbeddingCount: user.faceEmbeddings.length
        })),
        statistics: stats
      }
    });

  } catch (error) {
    console.error('Database users query error:', error);
    return NextResponse.json(
      { 
        success: false, 
        message: 'เกิดข้อผิดพลาดในการดึงข้อมูลผู้ใช้',
        error: error instanceof Error ? error.message : 'Unknown error' 
      },
      { status: 500 }
    );
  }
}
