import { NextRequest, NextResponse } from 'next/server';

export async function GET() {
  return NextResponse.json({
    service: "Face Detection",
    status: "running",
    available_models: [
      {
        name: "YOLOv8n-face",
        loaded: true,
        device: "gpu",
        performance: {
          inference_count: 1234,
          average_inference_time: 0.025,
          throughput_fps: 40
        }
      },
      {
        name: "RetinaFace",
        loaded: true,
        device: "gpu", 
        performance: {
          inference_count: 856,
          average_inference_time: 0.035,
          throughput_fps: 28
        }
      }
    ],
    service_info: {
      performance_stats: {
        total_detections: 5678,
        successful_detections: 5432,
        average_processing_time: 0.030,
        model_usage_count: {
          "YOLOv8n-face": 3456,
          "RetinaFace": 2222
        }
      },
      vram_status: {
        total_vram: 8.0,
        allocated_vram: 2.1,
        usage_percentage: 26.25,
        gpu_models: 2,
        cpu_models: 0
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
    await new Promise(resolve => setTimeout(resolve, 300));

    // Mock face detection result
    // In real implementation, this would process the actual image
    const mockResult = {
      success: true,
      faces: [
        {
          bbox: [150, 100, 300, 250], // [x, y, width, height]
          confidence: Math.random() * 0.3 + 0.7, // Random confidence between 0.7-1.0
          landmarks: [
            [180, 140], [220, 140], // eyes
            [200, 160], // nose
            [185, 190], [215, 190]  // mouth corners
          ]
        }
      ],
      processing_time: 0.025,
      model_used: "YOLOv8n-face"
    };

    // Simulate no face detected sometimes (10% chance)
    if (Math.random() < 0.1) {
      return NextResponse.json({
        success: true,
        faces: [],
        message: "No faces detected",
        processing_time: 0.025,
        model_used: "YOLOv8n-face"
      });
    }

    return NextResponse.json(mockResult);

  } catch (error) {
    console.error('Face detection error:', error);
    return NextResponse.json(
      { success: false, message: 'Face detection failed' },
      { status: 500 }
    );
  }
}
