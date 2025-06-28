import React from 'react';
import { AIServiceStatus } from '@/hooks/useAIServiceStatus';

interface AIStatusModalProps {
  status: AIServiceStatus;
  isOpen: boolean;
  onClose: () => void;
}

export const AIStatusModal: React.FC<AIStatusModalProps> = ({ 
  status, 
  isOpen, 
  onClose 
}) => {
  if (!isOpen) return null;

  const getStatusText = () => {
    switch (status.status) {
      case 'online': return 'ออนไลน์';
      case 'offline': return 'ออฟไลน์';
      case 'loading': return 'กำลังโหลด...';
      case 'error': return 'ข้อผิดพลาด';
      default: return 'ไม่ทราบสถานะ';
    }
  };

  const getStatusColor = () => {
    switch (status.status) {
      case 'online': return 'text-green-500';      case 'offline': return 'text-red-500';
      case 'loading': return 'text-yellow-500';
      case 'error': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const formatTime = (time?: number) => {
    if (!time) return 'N/A';
    return `${(time * 1000).toFixed(2)} ms`;
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl max-w-lg w-full max-h-[80vh] overflow-y-auto">
        {/* Header */}
        <div className="flex justify-between items-center p-6 border-b">
          <div>
            <h2 className="text-xl font-bold text-gray-900">{status.service}</h2>
            <p className={`text-sm font-medium ${getStatusColor()}`}>
              สถานะ: {getStatusText()}
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-2xl font-bold"
          >
            ×
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Service Info */}
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">ข้อมูลเซอร์วิส</h3>
            <div className="bg-gray-50 rounded-lg p-4 space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-600">สถานะ:</span>
                <span className={`font-medium ${getStatusColor()}`}>{getStatusText()}</span>
              </div>
              {status.lastUpdate && (
                <div className="flex justify-between">
                  <span className="text-gray-600">อัปเดตล่าสุด:</span>
                  <span className="text-gray-900">
                    {new Date(status.lastUpdate).toLocaleString('th-TH')}
                  </span>
                </div>
              )}
              <div className="flex justify-between">
                <span className="text-gray-600">Endpoint:</span>
                <span className="text-gray-900 text-sm font-mono">{status.service}</span>
              </div>
            </div>
          </div>

          {/* AI Models */}
          {status.models && status.models.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-3">โมเดล AI</h3>
              <div className="space-y-3">
                {status.models.map((model, index) => (
                  <div key={index} className="bg-gray-50 rounded-lg p-4">
                    <div className="flex justify-between items-center mb-2">
                      <h4 className="font-medium text-gray-900">{model.name}</h4>
                      <div className="flex items-center space-x-2">
                        <span className={`w-3 h-3 rounded-full ${model.loaded ? 'bg-green-500' : 'bg-red-500'}`}></span>
                        <span className="text-sm text-gray-600">{model.device.toUpperCase()}</span>
                      </div>
                    </div>
                    
                    {model.performance && (
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-gray-600">การใช้งาน:</span>
                          <span className="ml-2 font-medium">{model.performance.inference_count || 0} ครั้ง</span>
                        </div>
                        <div>
                          <span className="text-gray-600">เวลาเฉลี่ย:</span>
                          <span className="ml-2 font-medium">{formatTime(model.performance.average_inference_time)}</span>
                        </div>
                        {model.performance.throughput_fps && (
                          <>                            <div>
                              <span className="text-gray-600">FPS:</span>
                              <span className="ml-2 font-medium">{model.performance.throughput_fps.toFixed(2)}</span>
                            </div>
                          </>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* GPU Memory */}
          {status.vram && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-3">หน่วยความจำ GPU</h3>
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="mb-4">
                  <div className="flex justify-between mb-2">
                    <span className="text-gray-600">การใช้งาน</span>
                    <span className="font-medium">{status.vram.usage_percentage?.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div 
                      className="bg-blue-500 h-3 rounded-full transition-all duration-300" 
                      style={{ width: `${status.vram.usage_percentage || 0}%` }}
                    ></div>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600">GPU Models:</span>
                    <span className="ml-2 font-medium">{status.vram.gpu_models || 0}</span>
                  </div>                  <div>
                    <span className="text-gray-600">CPU Models:</span>
                    <span className="ml-2 font-medium">{status.vram.cpu_models || 0}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Performance Stats */}
          {status.performance && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-3">สถิติประสิทธิภาพ</h3>
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="grid grid-cols-2 gap-4 text-sm mb-4">
                  <div>
                    <span className="text-gray-600">การตรวจจับทั้งหมด:</span>
                    <span className="ml-2 font-medium">{status.performance.total_detections || 0}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">สำเร็จ:</span>
                    <span className="ml-2 font-medium text-green-600">{status.performance.successful_detections || 0}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">เวลาเฉลี่ย:</span>
                    <span className="ml-2 font-medium">{formatTime(status.performance.average_processing_time)}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">อัตราความสำเร็จ:</span>
                    <span className="ml-2 font-medium text-green-600">
                      {status.performance.total_detections ? 
                        ((status.performance.successful_detections || 0) / status.performance.total_detections * 100).toFixed(1) : 0}%
                    </span>
                  </div>
                </div>

                {status.performance.model_usage_count && (
                  <div>
                    <h4 className="text-sm font-medium text-gray-900 mb-2">การใช้งานโมเดล:</h4>
                    <div className="space-y-1">
                      {Object.entries(status.performance.model_usage_count).map(([model, count]) => (
                        <div key={model} className="flex justify-between text-sm">
                          <span className="text-gray-600">{model}:</span>
                          <span className="font-medium">{count} ครั้ง</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end p-6 border-t">
          <button
            onClick={onClose}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            ปิด
          </button>
        </div>
      </div>
    </div>
  );
};
