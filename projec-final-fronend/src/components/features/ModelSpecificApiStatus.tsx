'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Badge, Typography, Space, Spin } from 'antd';
import { CheckCircleOutlined, ExclamationCircleOutlined, CloseCircleOutlined } from '@ant-design/icons';

const { Text } = Typography;

interface ModelSpecificApiStatusProps {
  type: 'face-detection' | 'anti-spoofing' | 'age-gender' | 'face-analysis';
  checkInterval?: number;
  size?: 'small' | 'default' | 'large';
  showText?: boolean;
}

const ModelSpecificApiStatus: React.FC<ModelSpecificApiStatusProps> = ({
  type,
  checkInterval = 10000, // Reduced frequency to 10 seconds
  size = 'default',
  showText = false
}) => {
  const [status, setStatus] = useState<'checking' | 'online' | 'offline' | 'error'>('checking');
  const [lastCheck, setLastCheck] = useState<Date | null>(null);  const getApiEndpoint = (apiType: string) => {
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
    switch (apiType) {
      case 'face-detection':
        return `${baseUrl}/api/face-detection/health`;
      case 'anti-spoofing':
        return `${baseUrl}/api/anti-spoofing/health`;
      case 'age-gender':
        return `${baseUrl}/api/age-gender/health`;
      case 'face-analysis':
        return `${baseUrl}/api/face-analysis/health`;
      default:        return `${baseUrl}/api/health`;
    }
  };
  
  const checkApiStatus = useCallback(async () => {
    try {
      setStatus('checking');
      const endpoint = getApiEndpoint(type);      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000); // Reduced timeout to 3 seconds
        const response = await fetch(endpoint, {
        method: 'GET',
        signal: controller.signal,
        mode: 'cors',
        headers: {
          'Accept': 'application/json'
        }
      });
      
      clearTimeout(timeoutId);
      
      if (response.ok) {
        setStatus('online');
      } else {
        setStatus('error');
      }
      setLastCheck(new Date());    } catch (error) {
      // Only log detailed errors in development
      if (process.env.NODE_ENV === 'development') {
        console.warn(`API Status check failed for ${type}:`, error);
      }
      
      // Check if it's a network error or timeout
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          setStatus('offline'); // Timeout
        } else if (error.message.includes('Failed to fetch')) {          setStatus('offline'); // Network error
        } else {
          setStatus('error'); // Other errors
        }
      } else {
        setStatus('offline');
      }
      
      setLastCheck(new Date());
    }
  }, [type]);

  useEffect(() => {
    // Initial check
    checkApiStatus();
    
    // Set up interval
    const interval = setInterval(checkApiStatus, checkInterval);
    
    return () => clearInterval(interval);
  }, [type, checkInterval, checkApiStatus]);

  const getStatusIcon = () => {
    switch (status) {
      case 'checking':
        return <Spin size="small" />;
      case 'online':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'error':
        return <ExclamationCircleOutlined style={{ color: '#faad14' }} />;
      case 'offline':
      default:
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'online':
        return 'success';
      case 'error':
        return 'warning';
      case 'offline':
        return 'error';
      case 'checking':
      default:
        return 'processing';
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'checking':
        return 'กำลังตรวจสอบ...';
      case 'online':
        return 'ออนไลน์';
      case 'error':
        return 'มีปัญหา';
      case 'offline':
      default:
        return 'ออฟไลน์';
    }
  };

  const getModelName = () => {
    switch (type) {
      case 'face-detection':
        return 'Face Detection API';
      case 'anti-spoofing':
        return 'Anti-Spoofing API';
      case 'age-gender':
        return 'Age & Gender API';
      case 'face-analysis':
        return 'Face Analysis API';
      default:
        return 'API';
    }
  };

  if (showText) {
    return (
      <Space size="small">
        {getStatusIcon()}
        <Text style={{ fontSize: size === 'small' ? '12px' : '14px' }}>
          {getModelName()}
        </Text>        <Badge 
          status={getStatusColor() as "success" | "processing" | "error" | "default" | "warning"} 
          text={getStatusText()} 
        />
        {lastCheck && (
          <Text type="secondary" style={{ fontSize: '11px' }}>
            ({lastCheck.toLocaleTimeString()})
          </Text>
        )}
      </Space>
    );
  }

  return (
    <Space size="small">
      {getStatusIcon()}
      <Badge 
        status={getStatusColor() as "success" | "processing" | "error" | "default" | "warning"} 
        text={getStatusText()} 
      />
    </Space>
  );
};

export default ModelSpecificApiStatus;
