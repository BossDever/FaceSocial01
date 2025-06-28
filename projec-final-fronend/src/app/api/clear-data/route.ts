import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import * as jwt from 'jsonwebtoken';

const prisma = new PrismaClient();

export async function DELETE(request: NextRequest) {
  try {
    // Get token from Authorization header
    const authHeader = request.headers.get('authorization');
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return NextResponse.json({ error: 'ไม่พบ token การตรวจสอบสิทธิ์' }, { status: 401 });
    }

    const token = authHeader.substring(7);
    const decoded = jwt.verify(token, process.env.JWT_SECRET!) as { userId: string };

    // Check if user is admin (optional security check)
    const user = await (prisma as any).user.findUnique({
      where: { id: decoded.userId },
      select: { username: true }
    });

    if (!user || user.username !== 'admin01') {
      return NextResponse.json({ error: 'ไม่มีสิทธิ์ในการลบข้อมูล' }, { status: 403 });
    }

    // Clear various data tables
    await (prisma as any).message.deleteMany({});
    await (prisma as any).friendship.deleteMany({});
    
    // You can add more cleanup operations here as needed
    
    return NextResponse.json({ 
      message: 'ลบข้อมูลสำเร็จ',
      cleared: ['messages', 'friendships']
    });

  } catch (error) {
    console.error('Error clearing data:', error);
    
    if (error instanceof jwt.JsonWebTokenError) {
      return NextResponse.json({ error: 'Token ไม่ถูกต้อง' }, { status: 401 });
    }

    return NextResponse.json(
      { error: 'เกิดข้อผิดพลาดในการลบข้อมูล' },
      { status: 500 }
    );
  }
}
