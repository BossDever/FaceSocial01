import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import jwt from 'jsonwebtoken';
import { writeFile, mkdir } from 'fs/promises';
import path from 'path';

const prisma = new PrismaClient();

// GET - Fetch all posts with pagination
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
      console.error('JWT verification failed:', jwtError);
      return NextResponse.json({ message: 'Invalid token' }, { status: 401 });
    }const url = new URL(request.url);
    const page = parseInt(url.searchParams.get('page') || '1');
    const limit = parseInt(url.searchParams.get('limit') || '10');
    const userId = url.searchParams.get('userId'); // Add userId filter
    const skip = (page - 1) * limit;    let posts;
    if (userId) {
      posts = await prisma.$queryRawUnsafe(`
        SELECT 
          p.id,
          p.user_id as "userId",
          p.content,
          p.image_url as "imageUrl",
          p.created_at as "createdAt",
          json_build_object(
            'id', u.id,
            'username', u.username,
            'firstName', u.first_name,
            'lastName', u.last_name,
            'profilePicture', u.profile_image_url
          ) as user,
          COALESCE(l.likes_count, 0) as "likesCount",
          COALESCE(c.comments_count, 0) as "commentsCount"
        FROM posts p
        JOIN users u ON p.user_id = u.id
        LEFT JOIN (
          SELECT post_id, COUNT(*)::int as likes_count
          FROM likes 
          GROUP BY post_id
        ) l ON p.id = l.post_id
        LEFT JOIN (
          SELECT post_id, COUNT(*)::int as comments_count
          FROM comments 
          GROUP BY post_id
        ) c ON p.id = c.post_id
        WHERE p.user_id = $1
        ORDER BY p.created_at DESC
        LIMIT $2 OFFSET $3
      `, userId, limit, skip) as any[];
    } else {
      posts = await prisma.$queryRawUnsafe(`
        SELECT 
          p.id,
          p.user_id as "userId",
          p.content,
          p.image_url as "imageUrl",
          p.created_at as "createdAt",
          json_build_object(
            'id', u.id,
            'username', u.username,
            'firstName', u.first_name,
            'lastName', u.last_name,
            'profilePicture', u.profile_image_url
          ) as user,
          COALESCE(l.likes_count, 0) as "likesCount",
          COALESCE(c.comments_count, 0) as "commentsCount"
        FROM posts p
        JOIN users u ON p.user_id = u.id
        LEFT JOIN (
          SELECT post_id, COUNT(*)::int as likes_count
          FROM likes 
          GROUP BY post_id
        ) l ON p.id = l.post_id
        LEFT JOIN (
          SELECT post_id, COUNT(*)::int as comments_count
          FROM comments 
          GROUP BY post_id
        ) c ON p.id = c.post_id
        ORDER BY p.created_at DESC
        LIMIT $1 OFFSET $2
      `, limit, skip) as any[];
    }

    // Format response
    const formattedPosts = posts.map((post: any) => ({
      id: post.id,
      userId: post.userId,
      content: post.content,
      imageUrl: post.imageUrl,
      createdAt: post.createdAt,
      user: post.user,
      likes: [],
      comments: [],
      _count: {
        likes: Number(post.likesCount) || 0,
        comments: Number(post.commentsCount) || 0
      }
    }));

    return NextResponse.json(formattedPosts);
  } catch (error) {
    console.error('Error fetching posts:', error);
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}

// POST - Create a new post
export async function POST(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization');
    if (!authHeader?.startsWith('Bearer ')) {
      return NextResponse.json({ message: 'Unauthorized' }, { status: 401 });
    }

    const token = authHeader.substring(7);
    const decoded = jwt.verify(token, process.env.JWT_SECRET!) as { userId: string };

    const contentType = request.headers.get('content-type');
    let content: string;
    let imageFile: File | null = null;
    let imageUrl: string | undefined;

    if (contentType?.includes('application/json')) {
      // Handle JSON request
      const body = await request.json();
      content = body.content;
      imageUrl = body.imageUrl;
    } else {
      // Handle FormData request
      const formData = await request.formData();
      content = formData.get('content') as string;
      imageFile = formData.get('image') as File | null;
    }

    if (!content?.trim() && !imageFile && !imageUrl) {
      return NextResponse.json(
        { message: 'Content or image is required' },
        { status: 400 }
      );
    }

    // Handle image upload if there's a file
    if (imageFile) {
      const bytes = await imageFile.arrayBuffer();
      const buffer = Buffer.from(bytes);

      // Create uploads directory if it doesn't exist
      const uploadsDir = path.join(process.cwd(), 'public', 'uploads', 'posts');
      await mkdir(uploadsDir, { recursive: true });

      // Generate unique filename
      const timestamp = Date.now();
      const filename = `${timestamp}-${imageFile.name}`;
      const filepath = path.join(uploadsDir, filename);

      await writeFile(filepath, buffer);
      imageUrl = `/uploads/posts/${filename}`;
    }    // Create post using raw query
    const postId = await prisma.$queryRaw`
      INSERT INTO posts (user_id, content, image_urls)
      VALUES (${decoded.userId}::uuid, ${content?.trim() || ''}, ${imageUrl ? [imageUrl] : null})
      RETURNING id
    ` as { id: string }[];    // Fetch the created post with user data
    const post = await prisma.$queryRaw`
      SELECT 
        p.*,
        u.username,
        u.first_name,
        u.last_name,
        u.profile_image_url
      FROM posts p
      JOIN users u ON p.user_id = u.id
      WHERE p.id = ${postId[0].id}::uuid
    ` as any[];

    // Format the response to handle potential BigInt issues
    const formattedPost = {
      id: post[0].id,
      userId: post[0].user_id,
      content: post[0].content,
      imageUrl: post[0].image_urls?.[0] || null,
      createdAt: post[0].created_at?.toISOString() || new Date().toISOString(),
      user: {
        id: post[0].user_id,
        username: post[0].username,
        firstName: post[0].first_name,
        lastName: post[0].last_name,
        profilePicture: post[0].profile_image_url
      },
      likes: [],
      comments: [],
      _count: {
        likes: 0,
        comments: 0
      }
    };

    return NextResponse.json(formattedPost, { status: 201 });
  } catch (error) {
    console.error('Error creating post:', error);
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}
