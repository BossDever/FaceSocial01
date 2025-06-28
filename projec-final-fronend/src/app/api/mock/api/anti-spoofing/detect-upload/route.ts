import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 800));    const formData = await request.formData();
    const image = formData.get('image') as File;

    if (!image) {
      return NextResponse.json({
        success: false,
        error: 'No image file provided',
        message: 'Image file is required'
      }, { status: 400 });
    }

    // Mock detection results
    const mockResponse = {
      success: true,
      faces_detected: 1,
      faces_analysis: [
        {
          face_id: 1,
          is_real: Math.random() > 0.3, // 70% chance of being real
          confidence: 0.85 + Math.random() * 0.14, // Random confidence between 0.85-0.99
          spoofing_detected: Math.random() < 0.3, // 30% chance of spoofing detected
          region: {
            x: Math.floor(Math.random() * 100),
            y: Math.floor(Math.random() * 100),
            w: 300 + Math.floor(Math.random() * 200),
            h: 300 + Math.floor(Math.random() * 200),
            left_eye: {
              x: Math.floor(Math.random() * 50) + 150,
              y: Math.floor(Math.random() * 50) + 150
            },
            right_eye: {
              x: Math.floor(Math.random() * 50) + 250,
              y: Math.floor(Math.random() * 50) + 150
            }
          }
        }
      ],
      overall_result: {
        is_real: true,
        confidence: 0.92,
        spoofing_detected: false,
        real_faces: 1,
        fake_faces: 0
      },
      processing_time: 0.8947286605834961,
      model: "DeepFace Silent Face Anti-Spoofing",
      message: "Analysis completed successfully",
      error: null
    };

    // Update overall result based on faces analysis
    const realFaces = mockResponse.faces_analysis.filter(face => face.is_real).length;
    const fakeFaces = mockResponse.faces_analysis.length - realFaces;
    const hasSpoof = mockResponse.faces_analysis.some(face => face.spoofing_detected);
    
    mockResponse.overall_result = {
      is_real: realFaces > 0 && !hasSpoof,
      confidence: mockResponse.faces_analysis[0]?.confidence || 0.5,
      spoofing_detected: hasSpoof,
      real_faces: realFaces,
      fake_faces: fakeFaces
    };

    if (hasSpoof) {
      mockResponse.message = "Spoofing detected!";
    }

    return NextResponse.json(mockResponse);

  } catch (error) {
    console.error('Anti-spoofing API error:', error);
    return NextResponse.json({
      success: false,
      error: 'Internal server error',
      message: 'Failed to process anti-spoofing detection'
    }, { status: 500 });
  }
}
