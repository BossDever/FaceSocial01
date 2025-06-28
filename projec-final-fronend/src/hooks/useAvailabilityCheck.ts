import { useState, useEffect, useCallback } from 'react';

// Simple debounce implementation
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function debounce<T extends (...args: any[]) => void>(func: T, delay: number) {
  let timeoutId: NodeJS.Timeout;
  const debounced = (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func(...args), delay);
  };
  
  debounced.cancel = () => {
    clearTimeout(timeoutId);
  };
  
  return debounced;
}

interface AvailabilityResult {
  available: boolean;
  message: string;
  loading: boolean;
  error: string | null;
}

interface AvailabilityCheckResponse {
  success: boolean;
  available: boolean;
  message: string;
  field: string;
  value: string;
}

export const useAvailabilityCheck = (field: string, initialValue: string = '') => {
  const [result, setResult] = useState<AvailabilityResult>({
    available: true,
    message: '',
    loading: false,
    error: null
  });

  const checkAvailability = useCallback(async (value: string) => {
    if (!value || value.trim().length === 0) {
      setResult({
        available: true,
        message: '',
        loading: false,
        error: null
      });
      return;
    }

    setResult(prev => ({
      ...prev,
      loading: true,
      error: null
    }));

    try {
      const response = await fetch('/api/auth/check-availability', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          field,
          value: value.trim()
        })
      });

      const data: AvailabilityCheckResponse = await response.json();

      if (data.success) {
        setResult({
          available: data.available,
          message: data.message,
          loading: false,
          error: null
        });
      } else {
        setResult({
          available: false,
          message: '',
          loading: false,
          error: data.message || 'เกิดข้อผิดพลาดในการตรวจสอบ'        });
      }
    } catch {
      setResult({
        available: false,
        message: '',
        loading: false,
        error: 'เกิดข้อผิดพลาดในการเชื่อมต่อ'
      });
    }
  }, [field]);

  // Debounced version to avoid too many API calls
  const debouncedCheck = useCallback(
    debounce(checkAvailability, 800),
    [checkAvailability]
  );

  const check = useCallback((value: string) => {
    debouncedCheck(value);
  }, [debouncedCheck]);

  // Cleanup debounce on unmount
  useEffect(() => {
    return () => {
      debouncedCheck.cancel();
    };
  }, [debouncedCheck]);

  // Initial check if there's an initial value
  useEffect(() => {
    if (initialValue) {
      check(initialValue);
    }
  }, [initialValue, check]);

  return {
    ...result,
    check
  };
};

// Hook for checking full name (firstName + lastName combination)
export const useFullNameAvailabilityCheck = () => {
  const [result, setResult] = useState<AvailabilityResult>({
    available: true,
    message: '',
    loading: false,
    error: null
  });

  const checkFullNameAvailability = useCallback(async (firstName: string, lastName: string) => {
    if (!firstName || !lastName || firstName.trim().length === 0 || lastName.trim().length === 0) {
      setResult({
        available: true,
        message: '',
        loading: false,
        error: null
      });
      return;
    }

    setResult(prev => ({
      ...prev,
      loading: true,
      error: null
    }));

    try {
      const response = await fetch('/api/auth/check-availability', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          field: 'fullName',
          value: `${firstName.trim()}|${lastName.trim()}`
        })
      });

      const data: AvailabilityCheckResponse = await response.json();

      if (data.success) {
        setResult({
          available: data.available,
          message: data.message,
          loading: false,
          error: null
        });
      } else {
        setResult({
          available: false,
          message: '',
          loading: false,
          error: data.message || 'เกิดข้อผิดพลาดในการตรวจสอบ'        });
      }
    } catch {
      setResult({
        available: false,
        message: '',
        loading: false,
        error: 'เกิดข้อผิดพลาดในการเชื่อมต่อ'
      });
    }
  }, []);

  // Debounced version
  const debouncedCheck = useCallback(
    debounce(checkFullNameAvailability, 800),
    [checkFullNameAvailability]
  );

  const check = useCallback((firstName: string, lastName: string) => {
    debouncedCheck(firstName, lastName);
  }, [debouncedCheck]);

  // Cleanup
  useEffect(() => {
    return () => {
      debouncedCheck.cancel();
    };
  }, [debouncedCheck]);

  return {
    ...result,
    check
  };
};
