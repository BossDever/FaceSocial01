"""
Common utility functions and imports for AI services
"""

# Check if ONNX Runtime is available
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# Check if OpenCV is available
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None
    OPENCV_AVAILABLE = False

# Check if NumPy is available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

# Common utility functions
def check_dependencies() -> dict:
    """
    Check availability of common dependencies
    
    Returns:
        Dictionary with dependency status
    """
    return {
        "onnxruntime": ONNX_AVAILABLE,
        "torch": TORCH_AVAILABLE,
        "opencv": OPENCV_AVAILABLE,
        "numpy": NUMPY_AVAILABLE,
    }

def get_device_info() -> dict:
    """
    Get device information (CPU/GPU)
    
    Returns:
        Dictionary with device information
    """
    device_info = {
        "cpu_available": True,
        "gpu_available": False,
        "cuda_available": False,
        "device_count": 0,
        "current_device": None,
        "device_name": None,
    }
    
    if TORCH_AVAILABLE and torch:
        try:
            device_info["gpu_available"] = torch.cuda.is_available()
            device_info["cuda_available"] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                device_info["device_count"] = torch.cuda.device_count()
                device_info["current_device"] = torch.cuda.current_device()
                device_info["device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    
    return device_info

def ensure_numpy_array(data, dtype=None):
    """
    Ensure data is a numpy array
    
    Args:
        data: Input data
        dtype: Optional data type
        
    Returns:
        numpy array or None if numpy not available
    """
    if not NUMPY_AVAILABLE or np is None:
        return None
        
    if not isinstance(data, np.ndarray):
        try:
            data = np.array(data, dtype=dtype)
        except Exception:
            return None
    
    if dtype is not None and data.dtype != dtype:
        try:
            data = data.astype(dtype)
        except Exception:
            return None
            
    return data

def safe_import(module_name: str, package=None):
    """
    Safely import a module
    
    Args:
        module_name: Name of module to import
        package: Package name for relative imports
        
    Returns:
        Module if successful, None otherwise
    """
    try:
        if package:
            return __import__(module_name, fromlist=[package])
        else:
            return __import__(module_name)
    except ImportError:
        return None

# Export what's needed
__all__ = [
    'ONNX_AVAILABLE', 
    'TORCH_AVAILABLE', 
    'OPENCV_AVAILABLE',
    'NUMPY_AVAILABLE',
    'ort', 
    'torch', 
    'cv2', 
    'np',
    'check_dependencies',
    'get_device_info',
    'ensure_numpy_array',
    'safe_import'
]