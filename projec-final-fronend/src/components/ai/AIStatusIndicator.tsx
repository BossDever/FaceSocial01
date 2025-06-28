import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { AIServiceStatus } from '@/hooks/useAIServiceStatus';

interface AIStatusIndicatorProps {
  service: AIServiceStatus;
  className?: string;
}

const AIStatusIndicator: React.FC<AIStatusIndicatorProps> = ({ service, className }) => {
  const [showTooltip, setShowTooltip] = useState(false);

  if (!service) {
    return (
      <div className="flex items-center space-x-2">
        <div className="w-3 h-3 rounded-full bg-gray-400 animate-pulse" />
        <span className="text-sm text-gray-500">กำลังโหลด...</span>
      </div>
    );
  }
  const statusColor = 
    service.status === 'online' ? 'bg-green-500' :
    service.status === 'offline' ? 'bg-red-500' :
    service.status === 'loading' ? 'bg-yellow-500 animate-pulse' :
    service.status === 'error' ? 'bg-orange-500' :
    'bg-gray-500';

  const statusText = 
    service.status === 'online' ? 'ออนไลน์' :
    service.status === 'offline' ? 'ออฟไลน์' :
    service.status === 'loading' ? 'กำลังเริ่มต้น...' :
    service.status === 'error' ? 'เริ่มต้นล้มเหลว' :
    'ไม่ทราบ';

  const getTooltipContent = () => {
    const models = service.models || [];
    const performance = service.performance;
    const vram = service.vram;
    
    return (
      <div className="p-3 max-w-sm">
        <div className="font-semibold text-white mb-2">{service.service}</div>
        <div className="text-sm text-gray-200 mb-2">สถานะ: {statusText}</div>
        
        {models.length > 0 && (
          <div className="mb-2">
            <div className="text-sm font-medium text-white mb-1">โมเดล AI:</div>
            {models.map((model, index) => (
              <div key={index} className="text-xs text-gray-200 mb-1">
                • {model.name} ({model.device?.toUpperCase()})
                {model.performance?.inference_count && (
                  <span className="ml-1">- {model.performance.inference_count} ครั้ง</span>
                )}
              </div>
            ))}
          </div>
        )}
        
        {performance && (
          <div className="mb-2">
            <div className="text-sm font-medium text-white mb-1">สถิติ:</div>
            <div className="text-xs text-gray-200">
              {performance.total_detections && (
                <div>การตรวจจับทั้งหมด: {performance.total_detections}</div>
              )}
              {performance.average_processing_time && (
                <div>เวลาเฉลี่ย: {(performance.average_processing_time * 1000).toFixed(2)} ms</div>
              )}
            </div>
          </div>
        )}
        
        {vram && (
          <div>
            <div className="text-sm font-medium text-white mb-1">VRAM:</div>
            <div className="text-xs text-gray-200">
              <div>GPU: {vram.gpu_models || 0} โมเดล</div>
              <div>CPU: {vram.cpu_models || 0} โมเดล</div>
              {vram.usage_percentage && (
                <div>การใช้งาน: {vram.usage_percentage.toFixed(1)}%</div>
              )}
            </div>
          </div>
        )}
        
        <div className="text-xs text-gray-300 mt-2">
          อัปเดตล่าสุด: {service.lastUpdate ? new Date(service.lastUpdate).toLocaleTimeString('th-TH') : 'ไม่ทราบ'}
        </div>
      </div>
    );
  };
  return (
    <div 
      className={`relative flex items-center space-x-2 cursor-pointer ${className || ''}`}
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      <motion.div
        className={`w-3 h-3 rounded-full ${statusColor}`}
        animate={service.status === 'online' ? 
          { scale: [1, 1.1, 1], opacity: [1, 0.8, 1] } : 
          { scale: 1, opacity: 1 }
        }
        transition={{ duration: 2, repeat: Infinity }}
      />
      <span className="text-sm text-gray-600">{statusText}</span>
      
      {showTooltip && (
        <div className="absolute bottom-full left-0 mb-2 z-50">
          <div className="bg-gray-800 text-white rounded-lg shadow-lg border border-gray-600 min-w-0">
            {getTooltipContent()}
            <div className="absolute top-full left-4 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-800"></div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AIStatusIndicator;
