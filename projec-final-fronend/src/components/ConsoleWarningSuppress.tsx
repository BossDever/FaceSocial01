'use client';

import { useEffect } from 'react';

export default function ConsoleWarningSuppress() {
  useEffect(() => {
    // Suppress various console warnings and errors that are not critical
    const originalError = console.error;
    const originalWarn = console.warn;    console.error = (...args) => {
      const message = args[0];
      if (typeof message === 'string') {
        // Filter out Ant Design compatibility warnings
        if (message.includes('antd v5 support React is 16 ~ 18')) return;
        if (message.includes('antd: compatible')) return;
        if (message.includes('antd: message')) return;
        if (message.includes('Static function can not consume context')) return;
        if (message.includes('compatible') && message.includes('antd')) return;
        if (message.includes('u.ant.design/v5-for-19')) return;
        // Filter out specific face detection errors that are handled
        if (message.includes('Frame processing error')) return;
        if (message.includes('Face detection error')) return;
        // Don't filter out registration errors - these are important!
        // if (message.includes('API Error: {}')) return;
        // if (message.includes('Request failed with status code')) return;
      }
      originalError(...args);
    };

    console.warn = (...args) => {
      const message = args[0];
      if (typeof message === 'string') {
        // Filter out Ant Design warnings
        if (message.includes('antd')) return;
        if (message.includes('compatible')) return;
      }
      originalWarn(...args);
    };

    return () => {
      // Restore original console methods on cleanup
      console.error = originalError;
      console.warn = originalWarn;
    };
  }, []);

  return null;
}
