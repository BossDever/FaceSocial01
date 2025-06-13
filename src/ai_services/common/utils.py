# filepath: d:\projec-finals\src\ai_services\common\utils.py
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

# Export what's needed
__all__ = ['ONNX_AVAILABLE', 'TORCH_AVAILABLE', 'ort', 'torch']