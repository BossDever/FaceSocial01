import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import jwt from 'jsonwebtoken';

const prisma = new PrismaClient();

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const authHeader = request.headers.get('authorization');
    if (!authHeader?.startsWith('Bearer ')) {
      return NextResponse.json({ message: 'Unauthorized' }, { status: 401 });
    }

    const token = authHeader.substring(7);
    const decoded = jwt.verify(token, process.env.JWT_SECRET!) as { userId: string };
    const resolvedParams = await params;
    const postId = resolvedParams.id;// Check if post exists
    const post = await prisma.$queryRaw`
      SELECT id FROM posts WHERE id = ${postId}::uuid
    ` as { id: string }[];    if (!post || post.length === 0) {
      return NextResponse.json({ message: 'Post not found' }, { status: 404 });
    }

    // Check if user already liked the post
    const existingLike = await prisma.$queryRaw`
      SELECT id FROM post_likes 
      WHERE user_id = ${decoded.userId}::uuid AND post_id = ${postId}::uuid
    ` as { id: string }[];

    if (existingLike.length > 0) {
      // Unlike the post
      await prisma.$queryRaw`
        DELETE FROM post_likes 
        WHERE user_id = ${decoded.userId}::uuid AND post_id = ${postId}::uuid
      `;
      return NextResponse.json({ message: 'Post unliked' });
    } else {
      // Like the post
      await prisma.$queryRaw`
        INSERT INTO post_likes (user_id, post_id)
        VALUES (${decoded.userId}::uuid, ${postId}::uuid)
      `;
      return NextResponse.json({ message: 'Post liked' });
    }
  } catch (error) {
    console.error('Error toggling like:', error);
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}
