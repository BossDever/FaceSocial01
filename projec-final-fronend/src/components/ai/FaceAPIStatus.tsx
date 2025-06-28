'use client';

import React, { useState, useEffect } from 'react';
import { Badge, Tooltip, Space } from 'antd';
import { 
  CheckCircleOutlined, 
  ExclamationCircleOutlined,
  ClockCircleOutlined,
  ApiOutlined
} from '@ant-design/icons';

interface FaceAPIStatusProps {
  showDetails?: boolean;
}

interface APIStatus {
  status: 'healthy' | 'unhealthy' | 'loading';
  services: {
    face_detection: boolean;
    face_recognition: boolean;
    face_analysis: boolean;
    vram_manager: boolean;
  };
  responseTime?: number;
  lastChecked?: Date;
}

const FaceAPIStatus: React.FC<FaceAPIStatusProps> = ({ showDetails = false }) => {
  const [apiStatus, setApiStatus] = useState<APIStatus>({
    status: 'loading',
    services: {
      face_detection: false,
      face_recognition: false,
      face_analysis: false,
      vram_manager: false
    }
  });

  const checkAPIStatus = async () => {
    try {
      const startTime = Date.now();
      const response = await fetch('http://localhost:8080/health', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      const responseTime = Date.now() - startTime;
      
      if (response.ok) {
        const data = await response.json();
        setApiStatus({
          status: 'healthy',
          services: data.services,
          responseTime,
          lastChecked: new Date()
        });
      } else {
        setApiStatus(prev => ({
          ...prev,
          status: 'unhealthy',
          responseTime,
          lastChecked: new Date()
        }));
      }
    } catch (error) {
      console.error('API Status Check Error:', error);
      setApiStatus(prev => ({
        ...prev,
        status: 'unhealthy',
        lastChecked: new Date()
      }));
    }
  };

  useEffect(() => {
    // Check status immediately
    checkAPIStatus();
    
    // Check every 30 seconds
    const interval = setInterval(checkAPIStatus, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const getStatusBadge = () => {
    switch (apiStatus.status) {
      case 'healthy':
        return <Badge status="success" text="Face API Online" />;
      case 'unhealthy':
        return <Badge status="error" text="Face API Offline" />;
      default:
        return <Badge status="processing" text="Checking API..." />;
    }
  };

  const getStatusIcon = () => {
    switch (apiStatus.status) {
      case 'healthy':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'unhealthy':
        return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
      default:
        return <ClockCircleOutlined style={{ color: '#1890ff' }} />;
    }
  };

  const getTooltipContent = () => {
    const { services, responseTime, lastChecked } = apiStatus;
    const allServicesOnline = Object.values(services).every(Boolean);
    
    return (
      <div>
        <div><strong>Face API Status</strong></div>
        <div>Face Detection: {services.face_detection ? '✅' : '❌'}</div>
        <div>Face Recognition: {services.face_recognition ? '✅' : '❌'}</div>
        <div>Face Analysis: {services.face_analysis ? '✅' : '❌'}</div>
        <div>GPU Manager: {services.vram_manager ? '✅' : '❌'}</div>
        {responseTime && <div>Response Time: {responseTime}ms</div>}
        {lastChecked && <div>Last Checked: {lastChecked.toLocaleTimeString()}</div>}
        <div style={{ marginTop: 8, fontSize: '12px', opacity: 0.8 }}>
          {allServicesOnline ? 
            'All services are operational' : 
            'Some services may be unavailable'
          }
        </div>
      </div>
    );
  };

  if (!showDetails) {
    return (
      <Tooltip title={getTooltipContent()}>
        <Space size="small">
          <ApiOutlined />
          {getStatusBadge()}
        </Space>
      </Tooltip>
    );
  }

  return (
    <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 mb-4">
      <Space direction="vertical" size="small" style={{ width: '100%' }}>
        <Space>
          {getStatusIcon()}
          <strong>Face Recognition API Status</strong>
        </Space>
        
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div>
            Face Detection: {apiStatus.services.face_detection ? 
              <Badge status="success" text="Online" /> : 
              <Badge status="error" text="Offline" />
            }
          </div>
          <div>
            Face Recognition: {apiStatus.services.face_recognition ? 
              <Badge status="success" text="Online" /> : 
              <Badge status="error" text="Offline" />
            }
          </div>
          <div>
            Face Analysis: {apiStatus.services.face_analysis ? 
              <Badge status="success" text="Online" /> : 
              <Badge status="error" text="Offline" />
            }
          </div>
          <div>
            GPU Manager: {apiStatus.services.vram_manager ? 
              <Badge status="success" text="Online" /> : 
              <Badge status="error" text="Offline" />
            }
          </div>
        </div>
        
        {apiStatus.responseTime && (
          <div className="text-xs text-gray-500">
            Response Time: {apiStatus.responseTime}ms | 
            Last Checked: {apiStatus.lastChecked?.toLocaleTimeString()}
          </div>
        )}
      </Space>
    </div>
  );
};

export default FaceAPIStatus;
