import { NextResponse } from 'next/server';
import { getFaceApiUrl } from '@/lib/face-api';

export async function GET() {
  try {
    const apiUrl = getFaceApiUrl();
    console.log('🔍 Face API URL:', apiUrl);
    
    const response = await fetch(`${apiUrl}/health`);
    const data = await response.json();
    
    return NextResponse.json({
      success: true,
      apiUrl: apiUrl,
      response: data,
      status: response.status
    });  } catch (error) {
    console.error('❌ Face API test error:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : String(error),
      apiUrl: getFaceApiUrl()
    });
  }
}
