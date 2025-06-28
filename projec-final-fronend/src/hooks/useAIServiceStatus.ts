import { useState, useEffect, useCallback } from 'react';

// Types for AI Service Status
export interface AIServiceStatus {
  status: 'online' | 'offline' | 'loading' | 'error';
  service: string;
  models?: ModelInfo[];
  performance?: PerformanceStats;
  vram?: VRAMStatus;
  lastUpdate?: number;
}

export interface ModelInfo {
  name: string;
  loaded: boolean;
  device: 'cpu' | 'gpu' | 'cuda';
  performance?: {
    inference_count?: number;
    average_inference_time?: number;
    throughput_fps?: number;
  };
}

export interface PerformanceStats {
  total_detections?: number;
  successful_detections?: number;
  average_processing_time?: number;
  model_usage_count?: Record<string, number>;
}

export interface VRAMStatus {
  total_vram?: number;
  allocated_vram?: number;
  usage_percentage?: number;
  gpu_models?: number;
  cpu_models?: number;
}

// Custom hook for AI service status
export const useAIServiceStatus = (
  serviceEndpoint: string,
  refreshInterval: number = 5000
) => {  const [status, setStatus] = useState<AIServiceStatus>({
    status: 'loading',
    service: serviceEndpoint
  });  const fetchStatus = useCallback(async () => {
    try {
      // ใช้ timeout ที่นานขึ้นสำหรับ AI services ที่รันบน CPU
      const timeoutDuration = serviceEndpoint.includes('age-gender') ? 10000 : 5000;
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeoutDuration);
      
      // Age-Gender service ใช้ URL pattern ที่แตกต่าง (ไม่มี /api prefix)
      const baseUrl = serviceEndpoint.includes('age-gender') 
        ? 'http://localhost:8080' 
        : 'http://localhost:8080';
      
      const fullUrl = serviceEndpoint.includes('age-gender')
        ? `${baseUrl}${serviceEndpoint.replace('/api', '')}`  // ลบ /api prefix
        : `${baseUrl}${serviceEndpoint}`;
        
      console.log(`Fetching AI service: ${fullUrl}`);
      
      const response = await fetch(fullUrl, {
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        console.warn(`AI Service ${serviceEndpoint} returned ${response.status}`);
        setStatus(prev => ({ ...prev, status: 'error' }));
        return;
      }
      
      const data = await response.json();
      console.log(`AI Service ${serviceEndpoint} response:`, data);
      
      // ถ้าเป็น health endpoint ให้ใช้ข้อมูลพื้นฐาน
      if (serviceEndpoint.includes('/health')) {
        setStatus({
          status: 'online',
          service: extractServiceName(serviceEndpoint),
          models: extractBasicModels(serviceEndpoint),
          performance: undefined,
          vram: undefined,
          lastUpdate: Date.now()
        });
        return;
      }
      
      // สำหรับ endpoint อื่นๆ ให้ใช้การแยกข้อมูลแบบเดิม
      const extractedModels = extractModels(data);
      const extractedPerformance = extractPerformance(data);
      const extractedVRAM = extractVRAM(data);
      
      const newStatus = {
        status: 'online' as const,
        service: data.service || data.service_name || serviceEndpoint,
        models: extractedModels,
        performance: extractedPerformance,
        vram: extractedVRAM,
        lastUpdate: Date.now()
      };
      
      setStatus(newStatus);
    } catch (error) {
      console.warn(`AI Service ${serviceEndpoint} failed:`, error);
      setStatus(prev => ({ ...prev, status: 'offline' }));
    }
  }, [serviceEndpoint]);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, refreshInterval);
    return () => clearInterval(interval);
  }, [serviceEndpoint, refreshInterval, fetchStatus]);

  return { status, refetch: fetchStatus };
};

// Helper functions to extract data
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const extractModels = (data: any): ModelInfo[] => {  // Face Detection API format - use model_info from service_info
  if (data.service_info?.model_info) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return Object.entries(data.service_info.model_info).map(([key, model]: [string, any]) => ({
      name: model.model_name || key,
      loaded: model.model_loaded || true,
      device: (model.device === 'cuda' ? 'cuda' : model.device === 'gpu' ? 'gpu' : 'cpu') as 'cpu' | 'gpu' | 'cuda',
      performance: {
        inference_count: model.inference_count || 0,
        average_inference_time: model.average_inference_time || 0,
        throughput_fps: model.throughput_fps || 0
      }
    }));
  }  
  // Face Recognition API format with available_models
  if (data.available_models) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return data.available_models.map((model: any) => ({
      name: model.name,
      loaded: model.loaded !== false,
      device: (model.device === 'cuda' ? 'cuda' : model.device === 'gpu' ? 'gpu' : 'cpu') as 'cpu' | 'gpu' | 'cuda',
      performance: {
        inference_count: 0,
        average_inference_time: 0,
        throughput_fps: 0
      }
    }));  }
  // Anti-Spoofing format - use model_info
  if (data.model_info && (data.model_info.model_name || data.model_info.technology)) {
    const models = [{
      name: data.model_info.model_name || data.model_info.technology || 'Anti-Spoofing',
      loaded: data.model_info.is_initialized || true,
      device: 'cpu' as const,
      performance: {}
    }];
    
    // Add backend models if available
    if (data.model_info.backend_models) {
      data.model_info.backend_models.forEach((modelName: string) => {
        models.push({
          name: modelName,
          loaded: true,
          device: 'cpu' as const,
          performance: {}
        });
      });
    }
    
    return models;
  }  // Age-Gender format
  if (data.backend && data.detector) {
    return [{
      name: `${data.backend} (${data.detector})`,
      loaded: data.initialized || true,
      device: 'cpu' as const,
      performance: {}
    }];
  }  // Face Analysis format - extract recognition service info
  if (data.service_info?.recognition_service?.info?.model_info) {
    const modelInfo = data.service_info.recognition_service.info.model_info;
    return [{
      name: modelInfo.current_model || 'Face Analysis',
      loaded: modelInfo.model_loaded || true,
      device: modelInfo.gpu_enabled ? 'gpu' : 'cpu',
      performance: {}
    }];
  }

  return [];
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const extractPerformance = (data: any): PerformanceStats | undefined => {
  // Check service_info first
  if (data.service_info?.performance_stats) {
    return data.service_info.performance_stats;
  }
  
  // Face Recognition - check recognition service
  if (data.service_info?.recognition_service?.info?.performance_stats) {
    return data.service_info.recognition_service.info.performance_stats;
  }
  
  // Direct performance_stats
  if (data.performance_stats) {
    return data.performance_stats;
  }
    return undefined;
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const extractVRAM = (data: any): VRAMStatus | undefined => {
  const vram = data.service_info?.vram_status || data.vram_status;
  if (vram) {
    return {
      total_vram: vram.total_vram ? vram.total_vram / (1024 * 1024 * 1024) : undefined, // Convert to GB
      allocated_vram: vram.allocated_vram ? vram.allocated_vram / (1024 * 1024 * 1024) : undefined, // Convert to GB
      usage_percentage: vram.usage_percentage,
      gpu_models: vram.gpu_models || vram.model_count,
      cpu_models: vram.cpu_models || 0
    };
  }
  return undefined;
};

// Helper function to extract service name from endpoint
const extractServiceName = (endpoint: string): string => {
  if (endpoint.includes('face-detection')) return 'Face Detection';
  if (endpoint.includes('face-recognition')) return 'Face Recognition';
  if (endpoint.includes('face-analysis')) return 'Face Analysis';
  if (endpoint.includes('anti-spoofing')) return 'Anti-Spoofing';
  if (endpoint.includes('age-gender')) return 'Age & Gender Detection';
  return 'AI Service';
};

// Helper function to create basic models for health endpoints
const extractBasicModels = (endpoint: string): ModelInfo[] => {
  if (endpoint.includes('face-detection')) {
    return [{
      name: 'Face Detection Model',
      loaded: true,
      device: 'cpu' as const,
      performance: {}
    }];
  }
  if (endpoint.includes('face-recognition')) {
    return [{
      name: 'Face Recognition Model',
      loaded: true,
      device: 'cpu' as const,
      performance: {}
    }];
  }
  if (endpoint.includes('face-analysis')) {
    return [{
      name: 'Face Analysis Model',
      loaded: true,
      device: 'cpu' as const,
      performance: {}
    }];
  }
  if (endpoint.includes('anti-spoofing')) {
    return [{
      name: 'Anti-Spoofing Model',
      loaded: true,
      device: 'cpu' as const,
      performance: {}
    }];
  }
  if (endpoint.includes('age-gender')) {
    return [{
      name: 'Age & Gender Model',
      loaded: true,
      device: 'cpu' as const,
      performance: {}
    }];
  }
  return [];
};
