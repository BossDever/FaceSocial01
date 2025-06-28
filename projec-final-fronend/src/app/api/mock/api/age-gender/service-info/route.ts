import { NextResponse } from 'next/server';

export async function GET() {
  return NextResponse.json({
    service: "Age-Gender Detection",
    status: "running",
    backend: "opencv",
    detector: "opencv",
    initialized: true,
    service_info: {
      performance_stats: {
        total_detections: 2156,
        successful_detections: 2089,
        average_processing_time: 0.067,
        model_usage_count: {
          "age-gender-opencv": 2156
        }
      }
    }
  });
}
