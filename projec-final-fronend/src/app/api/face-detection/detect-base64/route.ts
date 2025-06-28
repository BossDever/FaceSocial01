import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { image_base64 } = body;

    if (!image_base64) {
      return NextResponse.json(
        { success: false, message: 'ไม่พบข้อมูลรูปภาพ' },
        { status: 400 }
      );
    }

    // Forward request to the actual Face Recognition API
    const response = await fetch('http://localhost:8080/api/face-detection/detect-base64', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image_base64 })
    });

    if (!response.ok) {
      throw new Error(`Face detection API error: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error('Face detection error:', error);
    return NextResponse.json(
      { 
        success: false, 
        message: 'เกิดข้อผิดพลาดในการตรวจจับใบหน้า',
        error: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}
