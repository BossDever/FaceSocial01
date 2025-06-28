import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const image = formData.get('image') as File;
    const model = formData.get('model') as string || 'facenet';
    
    if (!image) {
      return NextResponse.json(
        { success: false, message: 'No image provided' },
        { status: 400 }
      );
    }

    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 300));

    // Generate mock embedding (512-dimensional for FaceNet)
    const embeddingSize = 512;
    const mockEmbedding = Array.from({ length: embeddingSize }, () => 
      (Math.random() - 0.5) * 2 // Random values between -1 and 1
    );

    // Normalize embedding (L2 normalization)
    const magnitude = Math.sqrt(mockEmbedding.reduce((sum, val) => sum + val * val, 0));
    const normalizedEmbedding = mockEmbedding.map(val => val / magnitude);

    // Mock quality score (higher for better quality)
    const qualityScore = Math.random() * 0.3 + 0.7; // Between 0.7-1.0

    const mockResult = {
      success: true,
      embedding: normalizedEmbedding,
      quality_score: qualityScore,
      confidence: qualityScore,
      model_used: model.toLowerCase(),
      processing_time: 0.038,
      face_detected: true,
      face_count: 1,
      analysis: {
        face_quality: {
          score: qualityScore,
          factors: {
            lighting: Math.random() * 0.3 + 0.7,
            angle: Math.random() * 0.3 + 0.7,
            clarity: Math.random() * 0.3 + 0.7,
            size: Math.random() * 0.3 + 0.7
          }
        },
        face_landmarks: {
          detected: true,
          confidence: qualityScore
        }
      }
    };

    // Simulate extraction failure sometimes (3% chance)
    if (Math.random() < 0.03) {
      return NextResponse.json({
        success: false,
        message: "Failed to analyze face",
        face_detected: false,
        processing_time: 0.025
      });
    }

    return NextResponse.json(mockResult);

  } catch (error) {
    console.error('Face analysis error:', error);
    return NextResponse.json(
      { success: false, message: 'Face analysis failed' },
      { status: 500 }
    );
  }
}
