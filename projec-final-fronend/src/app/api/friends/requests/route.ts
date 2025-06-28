import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import jwt from 'jsonwebtoken';

const prisma = new PrismaClient();

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

    const { friendId } = await request.json();

    if (!friendId) {
      return NextResponse.json({ message: 'Friend ID is required' }, { status: 400 });
    }

    if (friendId === decoded.userId) {
      return NextResponse.json({ message: 'Cannot send friend request to yourself' }, { status: 400 });
    }

    // ตรวจสอบว่าเป็นเพื่อนกันแล้วหรือยัง
    const existingFriendship = await prisma.friendship.findFirst({
      where: {
        OR: [
          { userId: decoded.userId, friendId: friendId },
          { userId: friendId, friendId: decoded.userId }
        ]
      }
    });

    if (existingFriendship) {
      return NextResponse.json({ 
        message: existingFriendship.status === 'ACCEPTED' 
          ? 'Already friends' 
          : 'Friend request already sent' 
      }, { status: 400 });
    }

    // สร้าง friend request
    const friendship = await prisma.friendship.create({
      data: {
        userId: decoded.userId,
        friendId: friendId,
        status: 'PENDING'
      }
    });

    // สร้าง notification
    await prisma.notification.create({
      data: {
        userId: friendId,
        senderId: decoded.userId,
        type: 'FRIEND_REQUEST',
        title: 'คำขอเป็นเพื่อน',
        message: 'ส่งคำขอเป็นเพื่อนถึงคุณ',
        data: { friendshipId: friendship.id }
      }
    });

    return NextResponse.json({ 
      success: true, 
      message: 'Friend request sent successfully',
      friendship 
    });

  } catch (error) {
    console.error('Error sending friend request:', error);
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}

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
    }    const { searchParams } = new URL(request.url);
    
    // ดึงคำขอที่ส่งไป
    const sentRequests = await prisma.friendship.findMany({
      where: {
        userId: decoded.userId,
        status: 'PENDING'
      },
      include: {
        friend: {
          select: {
            id: true,
            username: true,
            firstName: true,
            lastName: true,
            profilePicture: true
          }
        }
      },
      orderBy: { createdAt: 'desc' }
    });

    // ดึงคำขอที่ได้รับ
    const receivedRequests = await prisma.friendship.findMany({
      where: {
        friendId: decoded.userId,
        status: 'PENDING'
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
        }
      },
      orderBy: { createdAt: 'desc' }
    });

    // แปลงข้อมูลให้ตรงกับ interface
    const sent = sentRequests.map(req => ({
      id: req.id,
      senderId: req.userId,
      receiverId: req.friendId,
      status: req.status,
      createdAt: req.createdAt,
      sender: null,
      receiver: req.friend
    }));

    const received = receivedRequests.map(req => ({
      id: req.id,
      senderId: req.userId,
      receiverId: req.friendId,
      status: req.status,
      createdAt: req.createdAt,
      sender: req.user,
      receiver: null
    }));

    return NextResponse.json({ sent, received });

  } catch (error) {
    console.error('Error fetching friend requests:', error);
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}
