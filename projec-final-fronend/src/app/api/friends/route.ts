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
    let decoded: { userId: string };
    
    try {
      decoded = jwt.verify(token, process.env.JWT_SECRET!) as { userId: string };
    } catch (jwtError) {
      return NextResponse.json({ message: 'Invalid token' }, { status: 401 });
    }    // ดึงรายการเพื่อนทั้งหมด
    const friendships = await prisma.friendship.findMany({
      where: {
        OR: [
          { userId: decoded.userId, status: 'ACCEPTED' },
          { friendId: decoded.userId, status: 'ACCEPTED' }
        ]
      },
      include: {
        user: {
          select: {
            id: true,
            username: true,
            firstName: true,
            lastName: true,
            profilePicture: true
          }
        },
        friend: {
          select: {
            id: true,
            username: true,
            firstName: true,
            lastName: true,
            profilePicture: true
          }
        }
      }
    });

    // แปลงข้อมูลเพื่อให้ได้เพื่อนที่ไม่ใช่ตัวเอง
    const friends = friendships.map((friendship: any) => {
      const friend = friendship.userId === decoded.userId 
        ? friendship.friend 
        : friendship.user;
      
      return {
        id: friendship.id,
        user1Id: friendship.userId,
        user2Id: friendship.friendId,
        createdAt: friendship.createdAt,
        user1: friendship.user,
        user2: friendship.friend,
        friend: friend
      };
    });

    return NextResponse.json(friends);

  } catch (error) {
    console.error('Error fetching friends:', error);
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}
