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
    const postId = resolvedParams.id;

    const { content } = await request.json();

    if (!content?.trim()) {
      return NextResponse.json({ message: 'Content is required' }, { status: 400 });
    }

    // Check if post exists
    const post = await prisma.$queryRaw`
      SELECT id FROM posts WHERE id = ${postId}::uuid
    ` as { id: string }[];

    if (!post || post.length === 0) {
      return NextResponse.json({ message: 'Post not found' }, { status: 404 });
    }

    // Create comment
    await prisma.$queryRaw`
      INSERT INTO post_comments (user_id, post_id, content)
      VALUES (${decoded.userId}::uuid, ${postId}::uuid, ${content.trim()})
    `;

    return NextResponse.json({ message: 'Comment added successfully' }, { status: 201 });
  } catch (error) {
    console.error('Error adding comment:', error);
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}
