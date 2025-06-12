# cSpell:disable
# mypy: ignore-errors
"""
ระบบโมเดล YOLO สำหรับการตรวจจับใบหน้า
รองรับ YOLOv9c, YOLOv9e และ YOLOv11m
Enhanced with better error handling and GPU optimization
FIXED: No more temp file creation that causes reload loops
"""

import time
import os
import logging
import numpy as np
import cv2
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Sequence # Added Sequence
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
    def detect(
        self, image, conf_threshold: float = 0.5, iou_threshold: float = 0.4
    ) -> List[np.ndarray]:
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
            "throughput_fps": 1.0 / max(avg_time, 0.001),
        }

    def cleanup(self):
        """ทำความสะอาดทรัพยากร"""
        self.model_loaded = False
        logger.info(f"Cleaned up {self.model_name}")

    def preprocess_image(self, image_input) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        แปลงรูปภาพให้เหมาะสมกับการใช้งานกับโมเดล - Enhanced
        Returns:
            input_tensor: The preprocessed image tensor for the model.
            scale_factors: A dictionary containing:
                "scale": The scale factor used to resize the image.
                "x_offset": The horizontal padding added.
                "y_offset": The vertical padding added.
                "original_width": The original width of the image.
                "original_height": The original height of the image.
                "new_width": Width of the image after scaling, before padding.
                "new_height": Height of the image after scaling, before padding.
        """
        try:
            # รองรับทั้งชื่อไฟล์และ numpy array
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                image = cv2.imread(image_input)
                if image is None:
                    raise ValueError(f"Cannot read image file: {image_input}")
            elif isinstance(image_input, np.ndarray):
                image = image_input.copy() # Work on a copy
            else:
                raise TypeError(
                    "Image input must be a file path (str) or a NumPy array."
                )

            if image is None or image.size == 0:
                raise ValueError("Invalid image data")

            original_height, original_width = image.shape[:2]
            target_height, target_width = self.get_input_size()

            # Calculate scale ratio and new dimensions
            scale = min(target_width / original_width, target_height / original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            resized_image = cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
            )

            # Create padded image
            padded_image = np.full(
                (target_height, target_width, 3), 114, dtype=np.uint8
            )

            # Calculate padding offsets
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2

            padded_image[
                y_offset : y_offset + new_height, x_offset : x_offset + new_width
            ] = resized_image

            input_tensor = padded_image.astype(np.float32) / 255.0
            input_tensor = np.transpose(input_tensor, (2, 0, 1))
            input_tensor = np.expand_dims(input_tensor, axis=0)

            scale_factors = {
                "scale": scale,
                "x_offset": x_offset,
                "y_offset": y_offset,
                "original_width": original_width,
                "original_height": original_height,
                "new_width": new_width, # Added for clarity
                "new_height": new_height, # Added for clarity
            }
            return input_tensor, scale_factors
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}", exc_info=True)
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
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
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
                        memory_limit = min(
                            2048 * 1024 * 1024, int(free_memory * 0.6)
                        )  # 2GB or 60%
                    else:
                        memory_limit = min(
                            1024 * 1024 * 1024, int(free_memory * 0.4)
                        )  # 1GB or 40%

                    cuda_options = {
                        "device_id": 0,
                        "arena_extend_strategy": "kSameAsRequested",
                        "gpu_mem_limit": memory_limit,
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "do_copy_in_default_stream": True,
                        "enable_cuda_graph": False,
                    }

                    providers.append(("CUDAExecutionProvider", cuda_options))
                    logger.info(
                        f"Configured {self.model_name} with "
                        f"{memory_limit / 1024 / 1024:.1f}MB GPU memory"
                    )

                except Exception as cuda_error:
                    logger.warning(
                        f"CUDA setup failed for {self.model_name}: "
                        f"{cuda_error}"
                    )
                    device = "cpu"

            providers.append("CPUExecutionProvider")

            # Create inference session
            self.session = ort.InferenceSession(
                self.model_path, sess_options=session_options, providers=providers
            )

            # Store model info
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]

            self.model_loaded = True
            self.device = device

            # Log success
            actual_providers = self.session.get_providers()
            device_used = (
                "GPU" if "CUDAExecutionProvider" in actual_providers else "CPU"
            )
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

    def detect(
        self, image, conf_threshold: float = 0.5, iou_threshold: float = 0.4
    ) -> List[np.ndarray]:
        """ตรวจจับใบหน้าในรูปภาพด้วย YOLO v9 - Enhanced"""
        if not self.model_loaded:
            # This check should ideally be more specific, e.g., self.session is None
            logger.error(
                f"Model {self.model_name} not loaded or session not initialized."
            )
            raise RuntimeError(f"Model {self.model_name} not loaded")

        start_time = time.time()

        try:
            input_tensor, scale_factors = self.preprocess_image(image)

            if self.device == "cuda":
                torch.cuda.empty_cache()

            inference_start_time = time.time()
            outputs = self.session.run(
                self.output_names, {self.input_name: input_tensor}
            )
            # Removed retry logic for brevity here, can be added back if needed

            inference_duration = time.time() - inference_start_time
            logger.debug(f"{self.model_name} inference took {inference_duration:.4f}s")

            detections = self._postprocess_outputs(
                outputs, scale_factors, conf_threshold, iou_threshold
            )

            total_time = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += total_time
            self.last_inference_time = total_time

            if self.device == "cuda":
                torch.cuda.empty_cache()

            logger.debug(
                f"{self.model_name} detected {len(detections)} faces "
                f"in {total_time:.3f}s"
            )
            return detections

        except Exception as e:
            logger.error(
                f"Detection failed in {self.model_name}: {e}", exc_info=True
            )
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return []

    def _postprocess_outputs(
        self,
        outputs: List[np.ndarray],
        scale_factors: Dict[str, Any],
        conf_threshold: float,
        iou_threshold: float,
    ) -> List[np.ndarray]:
        """
        แปลงผลลัพธ์จากโมเดล YOLO v9 ONNX.
        Handles output shape (1, 5, 8400) where 5 = xc, yc, w, h, conf.
        Coordinates are assumed to be pixel values on the 640x640 model input.
        """
        try:
            if not outputs or len(outputs) == 0 or outputs[0] is None:
                logger.warning(f"{self.model_name}: No outputs from model.")
                return []

            # outputs[0] has shape (1, 5, 8400)
            raw_predictions_tensor = outputs[0]
            if (
                raw_predictions_tensor.shape[0] != 1
                or raw_predictions_tensor.shape[1] != 5
            ):
                logger.warning(
                    f"{self.model_name}: Unexpected output tensor shape: "
                    f"{raw_predictions_tensor.shape}. Expected (1, 5, N)."
                )
                return []

            # Transpose from (5, num_detections) to (num_detections, 5)
            # after removing the batch dimension.
            predictions_data = raw_predictions_tensor[0]  # Shape (5, num_detections)
            transposed_predictions = np.transpose(
                predictions_data, (1, 0)
            )  # Shape (num_detections, 5)

            detections = []

            scale = scale_factors["scale"]
            x_offset = scale_factors["x_offset"] # dw: padding on left
            y_offset = scale_factors["y_offset"] # dh: padding on top
            original_width = scale_factors["original_width"]
            original_height = scale_factors["original_height"]

            for pred in transposed_predictions:
                # pred is [xc, yc, w, h, conf]
                # These are pixel coordinates on the 640x640 letterboxed/padded image
                x_center, y_center, width, height, confidence = pred

                if confidence < conf_threshold:
                    continue

                # Convert from center format to corner format (on 640x640 padded image)
                x1_padded = x_center - (width / 2)
                y1_padded = y_center - (height / 2)
                x2_padded = x_center + (width / 2)
                y2_padded = y_center + (height / 2)

                # Convert back to original image coordinates
                # Step 1: Remove padding to get coordinates relative to the scaled image
                # (the image content within the 640x640 padded input)
                x1_scaled_img = x1_padded - x_offset
                y1_scaled_img = y1_padded - y_offset
                x2_scaled_img = x2_padded - x_offset
                y2_scaled_img = y2_padded - y_offset

                # Step 2: Scale back to original image dimensions
                # The 'scale' factor is min(target_dim / orig_dim).
                # So, orig_coord = scaled_coord / scale.
                x1_orig = x1_scaled_img / scale
                y1_orig = y1_scaled_img / scale
                x2_orig = x2_scaled_img / scale
                y2_orig = y2_scaled_img / scale

                # Clip to original image bounds
                x1 = max(0, min(int(x1_orig), original_width -1))
                y1 = max(0, min(int(y1_orig), original_height -1))
                x2 = max(0, min(int(x2_orig), original_width -1))
                y2 = max(0, min(int(y2_orig), original_height -1))

                # Ensure valid box
                if x1 < x2 and y1 < y2:
                    detections.append(
                        np.array([x1, y1, x2, y2, confidence], dtype=np.float32)
                    )

            if not detections:
                return []

            # Apply NMS using the existing _nms method
            detections_array = np.array(detections)
            final_detections = self._nms(detections_array, iou_threshold)
            return final_detections

        except Exception as e:
            logger.error(
                f"Error in {self.model_name} _postprocess_outputs: {e}",
                exc_info=True
            )
            return []

    def detect_faces_raw(
        self, image, conf_threshold: float = 0.5, iou_threshold: float = 0.4
    ) -> List[np.ndarray]:
        """Wrapper for detect method"""
        return self.detect(image, conf_threshold, iou_threshold)

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
    FIXED: No more temp file creation
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
        except Exception as e:  # Corrected indentation for except block
            logger.error(f"Failed to load {self.model_name}: {e}")
            self.model_loaded = False
            return False

    def get_input_size(self) -> Tuple[int, int]:
        """ขนาด input ของโมเดล"""
        return self.input_size

    def _process_yolov11_results(
        self, results: Any, original_image_shape: Sequence[int]  # Changed type hint
    ) -> List[np.ndarray]:
        """Helper function to process detection results from YOLOv11 model."""
        detections = []
        if not results or not hasattr(results[0], 'boxes') or results[0].boxes is None:
            return detections

        h, w = original_image_shape[:2]
        for box_data in results[0].boxes:
            if box_data.xyxyn is not None and len(box_data.xyxyn) > 0:
                x1_norm, y1_norm, x2_norm, y2_norm = box_data.xyxyn[0].tolist()

                x1, y1, x2, y2 = (
                    int(x1_norm * w),
                    int(y1_norm * h),
                    int(x2_norm * w),
                    int(y2_norm * h),
                )

                conf = 0.0
                if box_data.conf is not None and len(box_data.conf) > 0:
                    conf = float(box_data.conf[0])

                if x1 < x2 and y1 < y2:  # Ensure coordinates are valid
                    detections.append(np.array([x1, y1, x2, y2, conf]))
                else:
                    logger.warning(
                        "Invalid box coordinates. Norm: (%.2f, %.2f, %.2f, %.2f). "
                        "Abs: (%d, %d, %d, %d). Image HxW: %dx%d.",
                        x1_norm, y1_norm, x2_norm, y2_norm,
                        x1, y1, x2, y2,
                        h, w
                    )
            else:
                logger.warning(
                    "box_data.xyxyn is None or empty, skipping this box."
                )
        return detections

    def detect(
        self, image: Any, conf_threshold: float = 0.5, iou_threshold: float = 0.4
    ) -> List[np.ndarray]:
        """ตรวจจับใบหน้าในรูปภาพด้วย YOLO v11 - COMPLETELY NO TEMP FILES"""
        if not self.model_loaded:
            raise RuntimeError(f"Model {self.model_name} not loaded")

        start_time = time.time()
        img_input_shape_for_error_log: Any = "unknown (image processing failed)"
        img_input: np.ndarray

        try:
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image file not found: {image}")
                img_input = cv2.imread(image)
                if img_input is None:
                    raise ValueError(f"Cannot read image file: {image}")
            elif isinstance(image, np.ndarray):
                img_input = image.copy()  # Use a copy to avoid modifying the original
            else:
                raise TypeError(
                    "Unsupported image type. Must be file path (str) or NumPy array."
                )

            img_input_shape_for_error_log = img_input.shape

            if img_input.size == 0:
                logger.warning(
                    f"Empty image provided to {self.model_name} "
                    f"(shape: {img_input.shape}). Skipping detection."
                )
                return []

            results = self.model.predict(
                source=img_input,
                conf=conf_threshold,
                iou=iou_threshold,
                device=self.device,
                verbose=False,
            )

            detections = self._process_yolov11_results(results, img_input.shape)

            self.last_inference_time = time.time() - start_time
            self.total_inference_time += self.last_inference_time
            self.inference_count += 1

            logger.debug(
                f"{self.model_name} detected {len(detections)} faces "
                f"in {self.last_inference_time:.3f}s"
            )

            return detections

        except Exception as e:
            logger.error(
                "Detection failed for %s on image of shape %s: %s",
                self.model_name,
                str(img_input_shape_for_error_log), # Ensure it\'s a string
                e,
                exc_info=True
            )
            return []

def fallback_opencv_detection(
    image: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: Tuple[int, int] = (30, 30),
) -> List[np.ndarray]:
    """
    Fallback face detection using OpenCV's Haar Cascade classifier.
    Returns a list of bounding boxes in [x1, y1, x2, y2, confidence] format.
    Confidence is set to a fixed value (e.g., 0.5) as Haar cascades don't
    provide it.
    """
    try:
        # Path to Haar cascade file from cv2.data
        cascade_path = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        if not os.path.exists(cascade_path):
            logger.error(f"Haar cascade file not found at {cascade_path}")
            # Attempt an alternative common path (less ideal)
            alt_cascade_path = "haarcascade_frontalface_default.xml"
            if os.path.exists(alt_cascade_path):
                cascade_path = alt_cascade_path
            else:
                logger.error(
                    f"Alternative Haar cascade not found: {alt_cascade_path}"
                )
                return []

        face_cascade = cv2.CascadeClassifier(cascade_path)

        if image is None or image.size == 0:
            logger.warning("Fallback OpenCV: Empty image received.")
            return []

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )

        detections = []
        for x, y, w, h in faces:
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            # Haar cascades don't provide confidence; use a fixed value.
            confidence = 0.5
            detections.append(np.array([x1, y1, x2, y2, confidence], dtype=np.float32))

        if not detections:
            logger.info("Fallback OpenCV: No faces detected.")
        else:
            logger.info(f"Fallback OpenCV: Detected {len(detections)} faces.")

        return detections
    except Exception as e:
        logger.error(f"Error in fallback_opencv_detection: {e}", exc_info=True)
        return []

# Ensure this function is exported if __all__ is defined in an __init__.py
