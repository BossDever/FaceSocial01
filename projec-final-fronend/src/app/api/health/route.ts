import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function GET() {
  try {
    console.log('🔗 Using Face API URL (server-side):', process.env.FACE_API_URL || 'http://host.docker.internal:8080');
    
    // Test database connection
    await prisma.$queryRaw`SELECT 1`;

    // Test Face Recognition API connection
    let faceApiStatus = 'unknown';
    try {
      const faceApiUrl = process.env.FACE_API_URL || 'http://host.docker.internal:8080';
      console.log('Testing Face API URL:', faceApiUrl);
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      
      const response = await fetch(`${faceApiUrl}/health`, {
        method: 'GET',
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      console.log('Face API response status:', response.status);
      faceApiStatus = response.ok ? 'connected' : 'error';
    } catch (error) {
      console.error('Face API health check error:', error);
      faceApiStatus = 'disconnected';
    }

    return NextResponse.json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      services: {
        database: 'connected',
        faceRecognitionApi: faceApiStatus
      }
    });

  } catch (error) {
    console.error('Health check failed:', error);
    return NextResponse.json(
      {
        status: 'unhealthy',
        timestamp: new Date().toISOString(),
        error: 'Database connection failed'
      },
      { status: 503 }
    );
  }
}
