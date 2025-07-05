"""
Base Model Class for YOLO Model Comparison System
Provides a common interface for all YOLO implementations.
"""

import time
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any
import cv2
from PIL import Image


class BaseYOLOModel(ABC):
    """Abstract base class for all YOLO models."""
    
    def __init__(self, model_name: str, variant: str, weights_path: str, 
                 input_size: Tuple[int, int], conf_threshold: float = 0.25, 
                 iou_threshold: float = 0.45, device: str = 'cuda'):
        """
        Initialize the base YOLO model.
        
        Args:
            model_name: Name of the YOLO model (e.g., 'YOLOv6', 'YOLOX')
            variant: Model variant (e.g., 'yolov6s', 'yolox_m')
            weights_path: Path to the model weights
            input_size: Input image size (height, width)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.variant = variant
        self.weights_path = weights_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        self.model = None
        self.is_loaded = False
        self.inference_times = []
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the model weights and prepare for inference."""
        pass
    
    @abstractmethod
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model inference.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Preprocessed tensor ready for model inference
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Preprocessed input tensor
            
        Returns:
            Raw model predictions
        """
        pass
    
    @abstractmethod
    def postprocess_predictions(self, predictions: torch.Tensor, 
                               original_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Postprocess model predictions to extract detections.
        
        Args:
            predictions: Raw model predictions
            original_shape: Original image shape (height, width)
            
        Returns:
            List of detection dictionaries with keys:
            - 'bbox': [x1, y1, x2, y2] in original image coordinates
            - 'confidence': confidence score
            - 'class_id': class index
            - 'class_name': class name
        """
        pass
    
    def predict(self, image: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """
        Run complete inference pipeline on an image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Tuple of (detections, inference_time)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Preprocess
        preprocessed = self.preprocess_image(image)
        
        # Forward pass
        with torch.no_grad():
            predictions = self.forward(preprocessed)
        
        # Postprocess
        detections = self.postprocess_predictions(predictions, image.shape[:2])
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return detections, inference_time
    
    def predict_batch(self, images: List[np.ndarray]) -> Tuple[List[List[Dict[str, Any]]], List[float]]:
        """
        Run inference on a batch of images.
        
        Args:
            images: List of input images as numpy arrays
            
        Returns:
            Tuple of (batch_detections, batch_inference_times)
        """
        batch_detections = []
        batch_times = []
        
        for image in images:
            detections, inference_time = self.predict(image)
            batch_detections.append(detections)
            batch_times.append(inference_time)
        
        return batch_detections, batch_times
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information including parameters and FLOPs.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'model_name': self.model_name,
            'variant': self.variant,
            'input_size': self.input_size,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'device': self.device,
            'is_loaded': self.is_loaded
        }
        
        if self.model is not None:
            # Calculate model parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            info.update({
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
            })
        
        return info
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics based on inference times.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        return {
            'avg_inference_time': float(np.mean(times)),
            'min_inference_time': float(np.min(times)),
            'max_inference_time': float(np.max(times)),
            'std_inference_time': float(np.std(times)),
            'avg_fps': float(1.0 / np.mean(times)),
            'total_inferences': len(times)
        }
    
    def reset_performance_stats(self) -> None:
        """Reset inference time statistics."""
        self.inference_times = []
    
    def __str__(self) -> str:
        return f"{self.model_name}-{self.variant}"
    
    def __repr__(self) -> str:
        return f"BaseYOLOModel(model_name='{self.model_name}', variant='{self.variant}')"


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""
    pass


class InferenceError(Exception):
    """Exception raised during inference."""
    pass