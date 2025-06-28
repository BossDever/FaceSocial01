import { NextResponse } from 'next/server';

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
