import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 1200));

    const formData = await request.formData();
    const file = formData.get('file') as File;

    if (!file) {
      return NextResponse.json({
        success: false,
        error: 'No image file provided',
        message: 'Image file is required'
      }, { status: 400 });
    }

    // Mock age and gender analysis
    const mockResponse = {
      success: true,
      message: "Analysis completed successfully",
      analyses: [
        {
          age: Math.floor(Math.random() * 60) + 18, // Random age between 18-77
          gender: Math.random() > 0.5 ? "Man" : "Woman",
          gender_confidence: 90 + Math.random() * 9.99, // Random confidence between 90-99.99%
          face_region: {
            x: Math.floor(Math.random() * 100),
            y: Math.floor(Math.random() * 100), 
            w: 300 + Math.floor(Math.random() * 200),
            h: 300 + Math.floor(Math.random() * 200)
          }
        }
      ],
      total_faces: 1,
      processing_time: 3 + Math.random() * 3 // Random processing time between 3-6 seconds
    };

    return NextResponse.json(mockResponse);

  } catch (error) {
    console.error('Age Gender API error:', error);
    return NextResponse.json({
      success: false,
      error: 'Internal server error',
      message: 'Failed to process age and gender analysis'
    }, { status: 500 });
  }
}
