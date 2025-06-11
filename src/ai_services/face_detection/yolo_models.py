# cSpell:disable
# mypy: ignore-errors
"""
ระบบโมเดล YOLO สำหรับการตรวจจับใบหน้า
รองรับ YOLOv9c, YOLOv9e และ YOLOv11m
Enhanced with better error handling and GPU optimization
"""
import time
import os
import logging
import numpy as np
import cv2
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import onnxruntime as ort

logger = logging.getLogger(__name__)

class FaceDetector(ABC):
    """คลาสพื้นฐานสำหรับการตรวจจับใบหน้า - Enhanced"""
    
    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        self.model_loaded = False
        self.device = "cpu"
        
        # Performance tracking
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.last_inference_time = 0.0
        
    @abstractmethod
    def load_model(self, device: str = "cuda") -> bool:
        """โหลดโมเดล"""
        pass
    
    @abstractmethod
    def detect(self, image, conf_threshold: float = 0.5, iou_threshold: float = 0.4) -> List[np.ndarray]:
        """ตรวจจับใบหน้าในรูปภาพ"""
        pass
    
    @abstractmethod
    def get_input_size(self) -> Tuple[int, int]:
        """ขนาด input ของโมเดล"""
        pass
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """ดึงสถิติประสิทธิภาพ"""
        avg_time = self.total_inference_time / max(self.inference_count, 1)
        return {
            "model_name": self.model_name,
            "device": self.device,
            "model_loaded": self.model_loaded,
            "inference_count": self.inference_count,
            "total_inference_time": self.total_inference_time,
            "average_inference_time": avg_time,
            "last_inference_time": self.last_inference_time,
            "throughput_fps": 1.0 / max(avg_time, 0.001)
        }
    
    def cleanup(self):
        """ทำความสะอาดทรัพยากร"""
        self.model_loaded = False
        logger.info(f"Cleaned up {self.model_name}")
    
    def preprocess_image(self, image_input) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        แปลงรูปภาพให้เหมาะสมกับการใช้งานกับโมเดล - Enhanced
        """
        try:
            # รองรับทั้งชื่อไฟล์และ numpy array
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                image = cv2.imread(image_input)
                if image is None:
                    raise ValueError(f"Cannot read image file: {image_input}")
            else:
                image = image_input
                
            if image is None or image.size == 0:
                raise ValueError("Invalid image data")
                
            # เก็บรูปร่างต้นฉบับ
            original_height, original_width = image.shape[:2]
            
            # ปรับขนาดรูปภาพตามขนาด input ของโมเดล
            target_height, target_width = self.get_input_size()
            scale = min(target_width / original_width, target_height / original_height)
            
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # Resize with high quality interpolation
            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            # เพิ่ม padding ให้ได้ขนาดตามที่ต้องการ
            padded_image = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
            
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            
            padded_image[y_offset:y_offset + new_height, 
                        x_offset:x_offset + new_width] = resized_image
            
            # แปลงเป็นรูปแบบที่เหมาะสม
            input_tensor = padded_image.astype(np.float32) / 255.0
            input_tensor = np.transpose(input_tensor, (2, 0, 1))
            input_tensor = np.expand_dims(input_tensor, axis=0)
            
            scale_factors = {
                'scale': scale,
                'x_offset': x_offset,
                'y_offset': y_offset,
                'original_width': original_width,
                'original_height': original_height
            }
            
            return input_tensor, scale_factors
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise

class YOLOv9ONNXDetector(FaceDetector):
    """
    คลาสสำหรับโมเดล YOLO v9 แบบ ONNX - Enhanced
    รองรับทั้ง YOLOv9c และ YOLOv9e
    """
    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)
        self.session = None
        self.input_size = (640, 640)
        self.input_name = None
        self.output_names = None
        
    def load_model(self, device: str = "cuda") -> bool:
        """โหลดโมเดล ONNX - Enhanced with better GPU management"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
                
            logger.info(f"Loading {self.model_name} model from: {self.model_path}")
            
            # Create session options
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.enable_mem_pattern = False
            session_options.enable_cpu_mem_arena = False
            
            # Configure providers
            providers = []
            
            if device == "cuda" and torch.cuda.is_available():
                try:
                    # Get GPU memory info
                    gpu_properties = torch.cuda.get_device_properties(0)
                    available_memory = gpu_properties.total_memory
                    allocated_memory = torch.cuda.memory_allocated(0)
                    free_memory = available_memory - allocated_memory
                    
                    # Configure memory limit based on model
                    if "yolov9e" in self.model_name.lower():
                        memory_limit = min(2048 * 1024 * 1024, int(free_memory * 0.6))  # 2GB or 60%
                    else:
                        memory_limit = min(1024 * 1024 * 1024, int(free_memory * 0.4))  # 1GB or 40%
                    
                    cuda_options = {
                        'device_id': 0,
                        'arena_extend_strategy': 'kSameAsRequested',
                        'gpu_mem_limit': memory_limit,
                        'cudnn_conv_algo_search': 'HEURISTIC',
                        'do_copy_in_default_stream': True,
                        'enable_cuda_graph': False,
                    }
                    
                    providers.append(('CUDAExecutionProvider', cuda_options))
                    logger.info(f"Configured {self.model_name} with {memory_limit/1024/1024:.1f}MB GPU memory")
                    
                except Exception as cuda_error:
                    logger.warning(f"CUDA setup failed for {self.model_name}: {cuda_error}")
                    device = "cpu"
            
            providers.append('CPUExecutionProvider')
            
            # Create inference session
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers
            )
            
            # Store model info
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            self.model_loaded = True
            self.device = device
            
            # Log success
            actual_providers = self.session.get_providers()
            device_used = "GPU" if 'CUDAExecutionProvider' in actual_providers else "CPU"
            logger.info(f"✅ {self.model_name} loaded successfully on {device_used}")
            logger.info(f"   Input: {self.input_name}")
            logger.info(f"   Outputs: {len(self.output_names)}")
            logger.info(f"   Providers: {actual_providers}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")
            self.model_loaded = False
            return False
    
    def get_input_size(self) -> Tuple[int, int]:
        """ขนาด input ของโมเดล"""
        return self.input_size
        
    def detect(self, image, conf_threshold: float = 0.5, iou_threshold: float = 0.4) -> List[np.ndarray]:
        """ตรวจจับใบหน้าในรูปภาพด้วย YOLO v9 - Enhanced"""
        if not self.model_loaded:
            raise RuntimeError(f"Model {self.model_name} not loaded")
        
        start_time = time.time()
        
        try:
            # Preprocess image
            input_tensor, scale_factors = self.preprocess_image(image)
            
            # Clear GPU cache if using CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # Run inference
            inference_start = time.time()
            try:
                outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            except Exception as inference_error:
                # Handle memory issues
                if "memory" in str(inference_error).lower() or "allocation" in str(inference_error).lower():
                    logger.warning(f"Memory issue in {self.model_name}, clearing cache and retrying...")
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Retry inference
                    outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
                else:
                    raise inference_error
                    
            inference_time = time.time() - inference_start
            
            # Post-process outputs
            detections = self._postprocess_outputs(
                outputs, scale_factors, conf_threshold, iou_threshold
            )
            
            # Update performance stats
            total_time = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += total_time
            self.last_inference_time = total_time
            
            # Clean up GPU memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            logger.debug(f"{self.model_name}: {len(detections)} faces, {total_time:.3f}s")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed in {self.model_name}: {e}")
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return []
    
    def detect_faces_raw(self, image, conf_threshold: float = 0.5, iou_threshold: float = 0.4) -> List[np.ndarray]:
        """Wrapper for detect method"""
        return self.detect(image, conf_threshold, iou_threshold)

    def _postprocess_outputs(self, 
                           outputs: List[np.ndarray], 
                           scale_factors: Dict[str, Any],
                           conf_threshold: float,
                           iou_threshold: float) -> List[np.ndarray]:
        """แปลงผลลัพธ์จากโมเดล YOLO v9 - Enhanced"""
        try:
            if not outputs or len(outputs) == 0:
                return []
                
            predictions = outputs[0]
            detections = []

            # Handle different output shapes
            if len(predictions.shape) == 3:
                batch_predictions = predictions[0]
            elif len(predictions.shape) == 2:
                batch_predictions = predictions
            else:
                logger.warning(f"Unexpected predictions shape: {predictions.shape}")
                return []

            for pred in batch_predictions:
                try:
                    if len(pred) < 5:
                        continue

                    # Extract coordinates and confidence
                    x_center = float(pred[0])
                    y_center = float(pred[1])
                    width = float(pred[2])
                    height = float(pred[3])
                    confidence = float(pred[4])

                    if confidence < conf_threshold:
                        continue

                    # Convert from center format to corner format
                    x1 = x_center - (width / 2)
                    y1 = y_center - (height / 2)
                    x2 = x_center + (width / 2)
                    y2 = y_center + (height / 2)

                    # Scale to input size
                    input_size = self.get_input_size()
                    x1 *= input_size[1]
                    y1 *= input_size[0]
                    x2 *= input_size[1]
                    y2 *= input_size[0]

                    # Convert back to original coordinates
                    scale = scale_factors['scale']
                    x_offset = scale_factors['x_offset']
                    y_offset = scale_factors['y_offset']

                    x1 = (x1 - x_offset) / scale
                    y1 = (y1 - y_offset) / scale
                    x2 = (x2 - x_offset) / scale
                    y2 = (y2 - y_offset) / scale

                    # Clip to image bounds
                    x1 = max(0, min(x1, scale_factors['original_width']))
                    y1 = max(0, min(y1, scale_factors['original_height']))
                    x2 = max(0, min(x2, scale_factors['original_width']))
                    y2 = max(0, min(y2, scale_factors['original_height']))

                    # Create detection array
                    detection = np.array([x1, y1, x2, y2, confidence], dtype=np.float32)
                    detections.append(detection)

                except (IndexError, ValueError, TypeError) as e:
                    logger.debug(f"Error processing prediction: {e}")
                    continue

            # Apply NMS
            if detections:
                detections_array = np.array(detections)
                final_detections = self._nms(detections_array, iou_threshold)
                return final_detections
            else:
                return []

        except Exception as e:
            logger.error(f"Error in postprocessing: {e}")
            return []

    def _nms(self, detections: np.ndarray, iou_threshold: float) -> List[np.ndarray]:
        """Non-Maximum Suppression - Enhanced"""
        if detections.shape[0] == 0:
            return []

        try:
            # Extract coordinates and scores
            x1 = detections[:, 0]
            y1 = detections[:, 1]
            x2 = detections[:, 2]
            y2 = detections[:, 3]
            scores = detections[:, 4]

            # Calculate areas
            areas = (x2 - x1) * (y2 - y1)
            
            # Sort by confidence
            order = scores.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(detections[i])
                
                if order.size == 1:
                    break

                # Calculate IoU with remaining boxes
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h
                
                iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

                # Keep boxes with IoU less than threshold
                inds = np.where(iou <= iou_threshold)[0]
                order = order[inds + 1]

            return keep
            
        except Exception as e:
            logger.error(f"NMS failed: {e}")
            return detections.tolist()

class YOLOv11Detector(FaceDetector):
    """
    คลาสสำหรับโมเดล YOLO v11 (Ultralytics) - Enhanced
    """
    
    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_path, model_name)
        self.model = None
        self.input_size = (640, 640)
        
    def load_model(self, device: str = "cuda") -> bool:
        """โหลดโมเดล YOLOv11 - Enhanced"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
                
            # Check CUDA availability
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, using CPU")
                device = "cpu"
            
            logger.info(f"Loading {self.model_name} model from: {self.model_path}")
            
            # Load model with Ultralytics
            from ultralytics import YOLO
            
            self.model = YOLO(self.model_path)
            self.device = device
            self.model_loaded = True
            
            logger.info(f"✅ {self.model_name} loaded successfully on {device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")
            self.model_loaded = False
            return False
    
    def get_input_size(self) -> Tuple[int, int]:
        """ขนาด input ของโมเดล"""
        return self.input_size
    
    def detect(self, image, conf_threshold: float = 0.5, iou_threshold: float = 0.4) -> List[np.ndarray]:
        """ตรวจจับใบหน้าในรูปภาพด้วย YOLO v11 - Enhanced"""
        if not self.model_loaded:
            raise RuntimeError(f"Model {self.model_name} not loaded")
        
        start_time = time.time()
        
        try:
            # Handle input image
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image file not found: {image}")
                img_input = image
            else:
                # Save numpy array as temporary file for YOLO
                if not isinstance(image, np.ndarray):
                    raise ValueError("Image must be numpy array or file path")
                
                temp_img_path = "temp_yolov11_input.jpg"
                cv2.imwrite(temp_img_path, image)
                img_input = temp_img_path
            
            # Run model inference
            inference_start = time.time()
            results = self.model(
                img_input, 
                conf=conf_threshold, 
                iou=iou_threshold, 
                device=self.device,
                verbose=False  # Suppress output
            )
            inference_time = time.time() - inference_start
            
            # Clean up temporary file
            if isinstance(image, np.ndarray) and os.path.exists("temp_yolov11_input.jpg"):
                os.remove("temp_yolov11_input.jpg")
            
            # Convert results
            detections = self._convert_results(results)
            
            # Update performance stats
            total_time = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += total_time
            self.last_inference_time = total_time
            
            logger.debug(f"{self.model_name}: {len(detections)} faces, {total_time:.3f}s")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed in {self.model_name}: {e}")
            # Clean up temporary file on error
            if isinstance(image, np.ndarray) and os.path.exists("temp_yolov11_input.jpg"):
                try:
                    os.remove("temp_yolov11_input.jpg")
                except:
                    pass
            return []
    
    def detect_faces_raw(self, image, conf_threshold: float = 0.5, iou_threshold: float = 0.4) -> List[np.ndarray]:
        """Wrapper for detect method"""
        return self.detect(image, conf_threshold, iou_threshold)
    
    def _convert_results(self, results) -> List[np.ndarray]:
        """แปลงผลลัพธ์จาก YOLO v11 - Enhanced"""
        detections = []
        
        try:
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        try:
                            # Get box coordinates and confidence
                            box = boxes[i]
                            xyxy = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            
                            x1, y1, x2, y2 = xyxy
                            confidence = float(conf)
                            
                            # Create detection array
                            detection = np.array([x1, y1, x2, y2, confidence], dtype=np.float32)
                            detections.append(detection)
                            
                        except Exception as e:
                            logger.debug(f"Error processing box {i}: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error converting YOLOv11 results: {e}")
        
        return detections
    
    def cleanup(self):
        """ทำความสะอาดทรัพยากร"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            super().cleanup()
            
        except Exception as e:
            logger.error(f"Error cleaning up {self.model_name}: {e}")

# Fallback OpenCV detection function
def fallback_opencv_detection(image: np.ndarray,
                              scale_factor: float = 1.1,
                              min_neighbors: int = 5,
                              min_size: Tuple[int, int] = (30, 30)) -> List[np.ndarray]:
    """OpenCV Haar Cascade fallback detection - Enhanced"""
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        
        # Load Haar Cascade
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if not os.path.exists(cascade_path):
            logger.error(f"Haar Cascade file not found: {cascade_path}")
            return []
        
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            logger.error("Failed to load Haar Cascade classifier")
            return []
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray_image, 
            scaleFactor=scale_factor, 
            minNeighbors=min_neighbors, 
            minSize=min_size
        )
        
        # Convert to detection format
        detections = []
        for (x, y, w, h) in faces:
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            confidence = 0.5  # Default confidence for Haar
            
            detection = np.array([x1, y1, x2, y2, confidence], dtype=np.float32)
            detections.append(detection)
        
        logger.debug(f"OpenCV Haar detected {len(detections)} faces")
        return detections
        
    except Exception as e:
        logger.error(f"OpenCV fallback detection failed: {e}")
        return []