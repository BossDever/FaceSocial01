import { NextResponse } from 'next/server';

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
