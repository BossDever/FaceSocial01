import jwt from 'jsonwebtoken';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export interface AdminUser {
  id: string;
  username: string;
  email: string;
  isAdmin: boolean;
}

export async function verifyAdminAccess(authHeader: string | null): Promise<{ success: boolean; user?: AdminUser; error?: string }> {
  if (!authHeader?.startsWith('Bearer ')) {
    return { success: false, error: 'Unauthorized' };
  }

  const token = authHeader.split(' ')[1];

  // Allow mock token for testing
  if (token === 'admin_token') {
    return {
      success: true,
      user: {
        id: 'admin-mock',
        username: 'admin01',
        email: 'admin@email.com',
        isAdmin: true
      }
    };
  }
  // Verify JWT token
  try {
    const jwtSecret = process.env.JWT_SECRET || 'your-secret-key';
    const decoded = jwt.verify(token, jwtSecret) as any;
    
    // Get user info from database to check if admin
    const user = await prisma.user.findUnique({
      where: { id: decoded.userId },
      select: { 
        id: true,
        username: true, 
        email: true 
      }
    });

    if (!user) {
      return { success: false, error: 'User not found' };
    }

    // Check if user is admin
    if (user.username !== 'admin01') {
      return { success: false, error: 'Admin access required' };
    }

    return {
      success: true,
      user: {
        id: user.id,
        username: user.username,
        email: user.email,
        isAdmin: true
      }
    };
  } catch (error) {
    console.error('JWT verification error:', error);
    return { success: false, error: 'Invalid token' };
  }
}
