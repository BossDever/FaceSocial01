import { NextRequest, NextResponse } from 'next/server';
import { headers } from 'next/headers';
import { verifyAdminAccess } from '../utils/auth';

// Mock ฐานข้อมูลสำหรับการทดสอบ (ในการใช้งานจริงจะเชื่อมต่อกับ SQLite/MongoDB)
let mockUsers = [
  { id: '1', username: 'admin01', email: 'admin@example.com', createdAt: new Date('2024-01-01') },
  { id: '2', username: 'user123', email: 'user@example.com', createdAt: new Date('2024-01-15') },
  { id: '3', username: 'testuser', email: 'test@example.com', createdAt: new Date('2024-02-01') }
];

let mockChats = [
  { id: '1', from: 'admin01', to: 'user123', message: 'Hello', timestamp: new Date('2024-03-01') },
  { id: '2', from: 'user123', to: 'admin01', message: 'Hi there', timestamp: new Date('2024-03-01') },
  { id: '3', from: 'testuser', to: 'admin01', message: 'Test message', timestamp: new Date('2024-03-02') }
];

let mockEmbeddings = [
  { id: '1', userId: 'admin01', fileName: 'admin_face_1.jpg', size: 2048, createdAt: new Date('2024-01-01') },
  { id: '2', userId: 'user123', fileName: 'user_face_1.jpg', size: 2048, createdAt: new Date('2024-01-15') },
  { id: '3', userId: 'testuser', fileName: 'test_face_1.jpg', size: 2048, createdAt: new Date('2024-02-01') }
];

interface DatabaseStats {
  users: {
    total: number;
    active: number;
    inactive: number;
    verified: number;
    unverified: number;
  };
  chats: {
    total: number;
    today: number;
    thisWeek: number;
    thisMonth: number;
  };
  embeddings: {
    total: number;
    totalSize: number;
    averageSize: number;
    orphaned: number;
  };
  system: {
    dbSize: string;
    lastBackup: string;
    uptime: string;
    performance: {
      avgResponseTime: number;
      totalQueries: number;
      slowQueries: number;
    };
  };
}

// Helper functions
function calculateDbSize(): string {
  // จำลองขนาดฐานข้อมูล
  const totalRecords = mockUsers.length + mockChats.length + mockEmbeddings.length;
  const estimatedSize = totalRecords * 0.5; // MB
  return `${estimatedSize.toFixed(2)} MB`;
}

function getUptime(): string {
  // จำลองเวลาที่ระบบทำงาน
  const startTime = new Date('2024-01-01');
  const now = new Date();
  const diffMs = now.getTime() - startTime.getTime();
  const days = Math.floor(diffMs / (1000 * 60 * 60 * 24));
  const hours = Math.floor((diffMs % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
  return `${days} วัน ${hours} ชั่วโมง`;
}

function getLastBackup(): string {
  // จำลองการสำรองข้อมูลล่าสุด
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  return yesterday.toLocaleDateString('th-TH', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
}

// GET: ดึงสถิติฐานข้อมูล
export async function GET(request: NextRequest) {
  try {
    const headersList = await headers();
    const authorization = headersList.get('authorization');

    const authResult = await verifyAdminAccess(authorization);
    if (!authResult.success) {
      return NextResponse.json(
        { error: authResult.error },
        { status: authResult.error === 'Unauthorized' ? 401 : 403 }
      );
    }

    // คำนวณสถิติผู้ใช้
    const activeUsers = mockUsers.filter(u => u.username !== 'inactive_user').length;
    const verifiedUsers = mockUsers.filter(u => u.username !== 'unverified_user').length;

    // คำนวณสถิติแชท
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const thisWeek = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    const thisMonth = new Date(now.getFullYear(), now.getMonth(), 1);

    const chatsToday = mockChats.filter(c => c.timestamp >= today).length;
    const chatsThisWeek = mockChats.filter(c => c.timestamp >= thisWeek).length;
    const chatsThisMonth = mockChats.filter(c => c.timestamp >= thisMonth).length;

    // คำนวณสถิติ embeddings
    const totalEmbeddingSize = mockEmbeddings.reduce((sum, e) => sum + e.size, 0);
    const averageSize = mockEmbeddings.length > 0 ? totalEmbeddingSize / mockEmbeddings.length : 0;
    const orphanedEmbeddings = mockEmbeddings.filter(e => 
      !mockUsers.some(u => u.username === e.userId)
    ).length;

    const stats: DatabaseStats = {
      users: {
        total: mockUsers.length,
        active: activeUsers,
        inactive: mockUsers.length - activeUsers,
        verified: verifiedUsers,
        unverified: mockUsers.length - verifiedUsers
      },
      chats: {
        total: mockChats.length,
        today: chatsToday,
        thisWeek: chatsThisWeek,
        thisMonth: chatsThisMonth
      },
      embeddings: {
        total: mockEmbeddings.length,
        totalSize: totalEmbeddingSize,
        averageSize: Math.round(averageSize),
        orphaned: orphanedEmbeddings
      },
      system: {
        dbSize: calculateDbSize(),
        lastBackup: getLastBackup(),
        uptime: getUptime(),
        performance: {
          avgResponseTime: 45, // ms
          totalQueries: 15420,
          slowQueries: 23
        }
      }
    };

    return NextResponse.json(stats);

  } catch (error) {
    console.error('Database stats error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// POST: ดำเนินการจัดการฐานข้อมูล
export async function POST(request: NextRequest) {
  try {
    const headersList = await headers();
    const authorization = headersList.get('authorization');

    const authResult = await verifyAdminAccess(authorization);
    if (!authResult.success) {
      return NextResponse.json(
        { error: authResult.error },
        { status: authResult.error === 'Unauthorized' ? 401 : 403 }
      );
    }

    const body = await request.json();
    const { action, table, options } = body;

    switch (action) {
      case 'backup':
        // จำลองการสำรองข้อมูล
        return NextResponse.json({
          success: true,
          message: 'สำรองข้อมูลเรียบร้อยแล้ว',
          backupFile: `backup_${Date.now()}.sql`,
          timestamp: new Date().toISOString()
        });

      case 'cleanup':
        // จำลองการล้างข้อมูล
        let cleanedItems = 0;
        
        if (table === 'chats') {
          const oldChats = mockChats.filter(c => {
            const daysDiff = (Date.now() - c.timestamp.getTime()) / (1000 * 60 * 60 * 24);
            return daysDiff > (options?.olderThanDays || 30);
          });
          cleanedItems = oldChats.length;
          // ในการใช้งานจริงจะลบข้อมูลจริง
        }
        
        if (table === 'embeddings') {
          const orphaned = mockEmbeddings.filter(e => 
            !mockUsers.some(u => u.username === e.userId)
          );
          cleanedItems = orphaned.length;
          // ในการใช้งานจริงจะลบข้อมูลจริง
        }

        return NextResponse.json({
          success: true,
          message: `ล้างข้อมูลเรียบร้อยแล้ว`,
          cleanedItems,
          table
        });

      case 'optimize':
        // จำลองการปรับปรุงประสิทธิภาพ
        return NextResponse.json({
          success: true,
          message: 'ปรับปรุงประสิทธิภาพฐานข้อมูลเรียบร้อยแล้ว',
          optimization: {
            indexesRebuilt: 15,
            spaceSaved: '2.3 MB',
            performanceImprovement: '12%'
          }
        });

      case 'repair':
        // จำลองการซ่อมแซมฐานข้อมูล
        return NextResponse.json({
          success: true,
          message: 'ตรวจสอบและซ่อมแซมฐานข้อมูลเรียบร้อยแล้ว',
          repair: {
            corruptedTables: 0,
            repairedRecords: 0,
            status: 'healthy'
          }
        });

      default:
        return NextResponse.json(
          { error: 'Invalid action' },
          { status: 400 }
        );
    }

  } catch (error) {
    console.error('Database operation error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
