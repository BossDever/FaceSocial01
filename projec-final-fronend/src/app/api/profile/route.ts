import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import jwt from 'jsonwebtoken';

const prisma = new PrismaClient();

export async function PUT(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization');
    if (!authHeader?.startsWith('Bearer ')) {
      return NextResponse.json({ message: 'Unauthorized' }, { status: 401 });
    }

    const token = authHeader.substring(7);
    const decoded = jwt.verify(token, process.env.JWT_SECRET!) as { userId: string };

    const body = await request.json();
    const { firstName, lastName, bio, location, website, phone } = body;

    // Validate required fields
    if (!firstName || !lastName) {
      return NextResponse.json(
        { message: 'First name and last name are required' },
        { status: 400 }
      );
    }

    // Update user profile in database
    await prisma.$queryRaw`
      UPDATE users 
      SET 
        first_name = ${firstName},
        last_name = ${lastName},
        bio = ${bio || null},
        location = ${location || null},
        website = ${website || null},
        phone = ${phone || null},
        updated_at = NOW()
      WHERE id = ${decoded.userId}::uuid
    `;

    // Fetch updated user data
    const updatedUser = await prisma.$queryRaw`
      SELECT 
        id,
        username,
        email,
        first_name,
        last_name,
        profile_image_url,
        bio,
        location,
        website,
        phone,
        date_of_birth,
        is_verified,
        created_at,
        updated_at
      FROM users
      WHERE id = ${decoded.userId}::uuid
    ` as any[];

    if (updatedUser.length === 0) {
      return NextResponse.json({ message: 'User not found' }, { status: 404 });
    }

    const user = updatedUser[0];
    
    // Format response
    const formattedUser = {
      id: user.id,
      username: user.username,
      email: user.email,
      firstName: user.first_name,
      lastName: user.last_name,
      fullName: `${user.first_name} ${user.last_name}`,
      profilePicture: user.profile_image_url,
      bio: user.bio,
      location: user.location,
      website: user.website,
      phone: user.phone,
      dateOfBirth: user.date_of_birth,
      isVerified: user.is_verified,
      createdAt: user.created_at,
      updatedAt: user.updated_at
    };

    return NextResponse.json({
      message: 'Profile updated successfully',
      user: formattedUser
    });

  } catch (error) {
    console.error('Error updating profile:', error);
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}
