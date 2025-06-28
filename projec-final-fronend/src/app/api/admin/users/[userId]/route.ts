import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import { verifyAdminAccess } from '../../utils/auth';
import bcrypt from 'bcryptjs';

const prisma = new PrismaClient();

// PUT - Update user (Admin only)
export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ userId: string }> }
) {
  try {
    const authHeader = request.headers.get('authorization');
    
    const authResult = await verifyAdminAccess(authHeader);
    if (!authResult.success) {
      return NextResponse.json(
        { message: authResult.error },
        { status: authResult.error === 'Unauthorized' ? 401 : 403 }
      );
    }

    const { userId } = await params;
    const body = await request.json();
    
    const {
      firstName,
      lastName,
      username,
      email,
      phone,
      isActive,
      isVerified,
      password
    } = body;

    // Prepare update data
    const updateData: any = {
      firstName,
      lastName,
      username,
      email,
      phone,
      isActive,
      isVerified
    };

    // Hash password if provided
    if (password && password.trim()) {
      const saltRounds = 10;
      updateData.passwordHash = await bcrypt.hash(password, saltRounds);
    }

    // Update user
    const updatedUser = await prisma.user.update({
      where: { id: userId },
      data: updateData,
      select: {
        id: true,
        username: true,
        email: true,
        firstName: true,
        lastName: true,
        phone: true,
        isActive: true,
        isVerified: true,
        profileImageUrl: true,
        createdAt: true,
        lastLoginAt: true
      }
    });

    return NextResponse.json(updatedUser);

  } catch (error) {
    console.error('Error updating user:', error);
    
    if (error instanceof Error && error.message.includes('Unique constraint')) {
      return NextResponse.json(
        { message: 'ชื่อผู้ใช้หรืออีเมลนี้ถูกใช้แล้ว' },
        { status: 400 }
      );
    }
    
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}

// DELETE - Delete user (Admin only)
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ userId: string }> }
) {
  try {
    const authHeader = request.headers.get('authorization');
    
    const authResult = await verifyAdminAccess(authHeader);
    if (!authResult.success) {
      return NextResponse.json(
        { message: authResult.error },
        { status: authResult.error === 'Unauthorized' ? 401 : 403 }
      );
    }

    const { userId } = await params;
    
    // Don't allow admin to delete themselves
    if (userId === authResult.user?.id) {
      return NextResponse.json(
        { message: 'ไม่สามารถลบบัญชีของตัวเองได้' },
        { status: 400 }
      );
    }

    // Delete user (this will cascade delete related data)
    await prisma.user.delete({
      where: { id: userId }
    });

    return NextResponse.json({ message: 'ลบผู้ใช้สำเร็จ' });

  } catch (error) {
    console.error('Error deleting user:', error);
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}
