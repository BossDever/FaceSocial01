import { NextResponse } from 'next/server';

export async function GET() {
  return NextResponse.json({
    service: "Face Analysis",
    status: "running",
    service_info: {
      recognition_service: {
        info: {
          model_info: {
            current_model: "DeepFace VGG-Face",
            model_loaded: true,
            gpu_enabled: true
          }
        }
      },
      performance_stats: {
        total_detections: 987,
        successful_detections: 934,
        average_processing_time: 0.089,
        model_usage_count: {
          "DeepFace VGG-Face": 987
        }
      },
      vram_status: {
        total_vram: 8.0,
        allocated_vram: 1.8,
        usage_percentage: 22.5,
        gpu_models: 1,
        cpu_models: 0
      }
    }
  });
}
