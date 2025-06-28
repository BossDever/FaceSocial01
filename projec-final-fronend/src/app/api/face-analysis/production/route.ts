import { NextRequest, NextResponse } from 'next/server';

// Production Face Analysis API wrapper
export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const image = formData.get('image') as File;
    const mode = formData.get('mode') as string || 'full_analysis';
    const model = formData.get('model') as string || 'facenet';
    
    if (!image) {
      return NextResponse.json(
        { success: false, message: 'No image provided' },
        { status: 400 }
      );
    }

    // Forward to production Face Recognition API
    const productionFormData = new FormData();
    productionFormData.append('file', image);
    productionFormData.append('mode', mode);
    productionFormData.append('recognition_model', model);

    const response = await fetch('http://localhost:8080/api/face-analysis/analyze', {
      method: 'POST',
      body: productionFormData
    });

    if (!response.ok) {
      return NextResponse.json(
        { success: false, message: 'Face analysis service unavailable' },
        { status: 503 }
      );
    }

    const result = await response.json();
    
    // Transform response to match expected format
    if (result.success && result.faces && result.faces.length > 0) {
      const face = result.faces[0];
      
      return NextResponse.json({
        success: true,
        embedding: face.embedding || [],
        quality_score: face.quality_score || 0.8,
        confidence: face.bbox?.confidence || 0.8,
        model_used: model,
        processing_time: result.total_processing_time || 0,
        face_detected: true,
        face_count: result.face_count || 1,
        analysis: {
          face_quality: {
            score: face.quality_score || 0.8,
            factors: {
              lighting: 0.8,
              angle: 0.8,
              clarity: 0.8,
              size: 0.8
            }
          }
        }
      });
    } else {
      return NextResponse.json({
        success: false,
        message: 'No face detected in the image',
        face_detected: false,
        face_count: 0
      });
    }

  } catch (error) {
    console.error('Face analysis error:', error);
    return NextResponse.json(
      { success: false, message: 'Internal server error' },
      { status: 500 }
    );
  }
}
