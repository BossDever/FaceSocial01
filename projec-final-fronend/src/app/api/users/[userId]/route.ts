import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import jwt from 'jsonwebtoken';

const prisma = new PrismaClient();

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ userId: string }> }
) {
  try {
    const authHeader = request.headers.get('authorization');
    if (!authHeader?.startsWith('Bearer ')) {
      return NextResponse.json({ message: 'Unauthorized' }, { status: 401 });
    }

    const token = authHeader.substring(7);
    const decoded = jwt.verify(token, process.env.JWT_SECRET!) as { userId: string };

    const resolvedParams = await params;
    const userId = resolvedParams.userId;    const user = await prisma.$queryRawUnsafe(`
      SELECT id, username, first_name, last_name, profile_image_url, bio, created_at
      FROM users 
      WHERE id = $1 OR username = $1
    `, userId) as {
      id: string;
      username: string;
      first_name: string;
      last_name: string;
      profile_image_url: string | null;
      bio: string | null;
      created_at: Date;
    }[];

    if (!user || user.length === 0) {
      return NextResponse.json({ message: 'User not found' }, { status: 404 });
    }

    // Transform to match frontend expectations
    const transformedUser = {
      id: user[0].id,
      username: user[0].username,
      firstName: user[0].first_name,
      lastName: user[0].last_name,
      profilePicture: user[0].profile_image_url,
      bio: user[0].bio,
      createdAt: user[0].created_at
    };

    return NextResponse.json(transformedUser);

  } catch (error) {
    console.error('Error fetching user:', error);
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}
