import { NextRequest, NextResponse } from 'next/server';

export async function GET() {
  return NextResponse.json({
    service: "Anti-Spoofing",
    status: "running",
    model_info: {
      model_name: "Silent-Face-Anti-Spoofing",
      is_initialized: true,
      device: "cpu"
    },
    service_info: {
      performance_stats: {
        total_detections: 1543,
        successful_detections: 1487,
        average_processing_time: 0.125,
        model_usage_count: {
          "Silent-Face-Anti-Spoofing": 1543
        }
      }
    }
  });
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const image = formData.get('image') as File;
    
    if (!image) {
      return NextResponse.json(
        { success: false, message: 'No image provided' },
        { status: 400 }
      );
    }

    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 200));

    // Mock anti-spoofing result
    // In real implementation, this would analyze the actual image
    const isReal = Math.random() > 0.1; // 90% chance of real face
    const confidence = isReal ? Math.random() * 0.3 + 0.7 : Math.random() * 0.5 + 0.1;

    const mockResult = {
      success: true,
      overall_result: {
        is_real: isReal,
        confidence: confidence,
        spoofing_detected: !isReal
      },
      detailed_results: {
        liveness_score: confidence,
        texture_analysis: {
          score: confidence + Math.random() * 0.1 - 0.05,
          status: isReal ? "real" : "fake"
        },
        depth_analysis: {
          score: confidence + Math.random() * 0.1 - 0.05,
          status: isReal ? "real" : "fake"
        },
        motion_analysis: {
          score: confidence + Math.random() * 0.1 - 0.05,
          status: isReal ? "real" : "fake"
        }
      },
      processing_time: 0.125,
      model_used: "Silent-Face-Anti-Spoofing",
      device: "cpu"
    };

    return NextResponse.json(mockResult);

  } catch (error) {
    console.error('Anti-spoofing error:', error);
    return NextResponse.json(
      { success: false, message: 'Anti-spoofing detection failed' },
      { status: 500 }
    );
  }
}
