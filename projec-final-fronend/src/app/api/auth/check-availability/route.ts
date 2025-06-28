import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { field, value } = body;

    if (!field || !value) {
      return NextResponse.json({
        success: false,
        message: 'กรุณาระบุฟิลด์และค่าที่ต้องการตรวจสอบ'
      }, { status: 400 });
    }

    // Simulate API delay for better UX
    await new Promise(resolve => setTimeout(resolve, 300));

    let isAvailable = true;
    let message = '';

    switch (field) {
      case 'email':
        const emailUser = await prisma.user.findUnique({
          where: { email: value.toLowerCase() }
        });
        isAvailable = !emailUser;
        message = emailUser ? 'อีเมลนี้ถูกใช้งานแล้ว' : 'อีเมลพร้อมใช้งาน';
        break;

      case 'username':
        const usernameUser = await prisma.user.findUnique({
          where: { username: value.toLowerCase() }
        });
        isAvailable = !usernameUser;
        message = usernameUser ? 'ชื่อผู้ใช้นี้ถูกใช้งานแล้ว' : 'ชื่อผู้ใช้พร้อมใช้งาน';
        break;

      case 'phone':
        if (!value.trim()) {
          isAvailable = true;
          message = 'เบอร์โทรศัพท์พร้อมใช้งาน';
          break;
        }
        const phoneUser = await prisma.user.findFirst({
          where: { phone: value }
        });
        isAvailable = !phoneUser;
        message = phoneUser ? 'เบอร์โทรศัพท์นี้ถูกใช้งานแล้ว' : 'เบอร์โทรศัพท์พร้อมใช้งาน';
        break;

      case 'fullName':
        // Check if combination of firstName and lastName exists
        const [firstName, lastName] = value.split('|');
        if (!firstName || !lastName) {
          isAvailable = true;
          message = 'ชื่อ-นามสกุลพร้อมใช้งาน';
          break;
        }
        const nameUser = await prisma.user.findFirst({
          where: {
            firstName: firstName.trim(),
            lastName: lastName.trim()
          }
        });
        isAvailable = !nameUser;
        message = nameUser ? 'ชื่อ-นามสกุลนี้ถูกใช้งานแล้ว' : 'ชื่อ-นามสกุลพร้อมใช้งาน';
        break;

      default:
        return NextResponse.json({
          success: false,
          message: 'ฟิลด์ที่ระบุไม่ถูกต้อง'
        }, { status: 400 });
    }

    return NextResponse.json({
      success: true,
      available: isAvailable,
      message: message,
      field: field,
      value: value
    });

  } catch (error) {
    console.error('Availability check error:', error);
    return NextResponse.json({
      success: false,
      message: 'เกิดข้อผิดพลาดในการตรวจสอบข้อมูล'
    }, { status: 500 });
  }
}
