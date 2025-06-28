// Suppress specific console warnings
export const suppressAntdCompatibilityWarning = () => {  if (typeof window !== 'undefined') {
    const originalError = console.error;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    console.error = (...args: any[]) => {
      const message = args[0];
      
      // Suppress Ant Design compatibility warning
      if (typeof message === 'string' && 
          message.includes('antd v5 support React is 16 ~ 18')) {
        return;
      }
      
      // Call original console.error for other messages
      originalError.apply(console, args);
    };
  }
};
