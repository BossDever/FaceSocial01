import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    
    // Forward the FormData to the actual Anti-Spoofing API
    const response = await fetch('http://localhost:8080/api/anti-spoofing/detect-upload', {
      method: 'POST',
      body: formData // Forward the FormData directly
    });

    if (!response.ok) {
      throw new Error(`Anti-spoofing API error: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error('Anti-spoofing error:', error);
    return NextResponse.json(
      { 
        success: false, 
        message: 'เกิดข้อผิดพลาดในการตรวจจับการปลอมแปลง',
        error: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}
