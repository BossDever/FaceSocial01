import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    console.log('🧹 Starting gallery clear operation...');

    // Call the Face Recognition Server's clear gallery endpoint
    const clearResponse = await fetch('http://localhost:8080/api/face-recognition/gallery/clear', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({})
    });

    if (!clearResponse.ok) {
      const errorText = await clearResponse.text();
      console.error('❌ Failed to clear Face Recognition Server gallery:', errorText);
      return NextResponse.json(
        { success: false, message: 'Failed to clear Face Recognition Server gallery' },
        { status: 500 }
      );
    }

    const clearResult = await clearResponse.json();
    console.log('✅ Face Recognition Server gallery cleared:', clearResult);

    // Optional: Also clear any local database entries if needed
    // This would depend on your specific implementation
    
    return NextResponse.json({
      success: true,
      message: 'Gallery cleared successfully',
      data: clearResult
    });

  } catch (error) {
    console.error('❌ Error clearing gallery:', error);
    return NextResponse.json(
      { success: false, message: 'Failed to clear gallery' },
      { status: 500 }
    );
  }
}
