"""
Mock YOLO model implementation for demonstration and testing purposes.
This is used when actual model implementations are not available.
"""

import time
import random
import torch
import numpy as np
from typing import List, Dict, Tuple, Any

from utils.base_model import BaseYOLOModel


class MockYOLOModel(BaseYOLOModel):
    """Mock YOLO model for demonstration purposes."""
    
    def __init__(self, model_name: str, variant: str, weights_path: str, 
                 input_size: Tuple[int, int], conf_threshold: float = 0.5, 
                 iou_threshold: float = 0.45, device: str = 'cuda'):
        """Initialize mock model."""
        super().__init__(model_name, variant, weights_path, input_size, 
                        conf_threshold, iou_threshold, device)
        
        # Mock model parameters based on variant
        self.mock_params = self._get_mock_params(variant)
        
    def _get_mock_params(self, variant: str) -> Dict[str, Any]:
        """Get mock parameters based on variant."""
        params_map = {
            # YOLOv6 variants
            'yolov6n': {'params': 4_200_000, 'flops': 4.7e9, 'speed_factor': 1.2},
            'yolov6s': {'params': 18_500_000, 'flops': 45.3e9, 'speed_factor': 1.0},
            'yolov6m': {'params': 34_300_000, 'flops': 85.8e9, 'speed_factor': 0.8},
            'yolov6l': {'params': 58_500_000, 'flops': 150.7e9, 'speed_factor': 0.6},
            
            # YOLOX variants
            'yolox_nano': {'params': 900_000, 'flops': 1.08e9, 'speed_factor': 1.5},
            'yolox_tiny': {'params': 5_100_000, 'flops': 6.45e9, 'speed_factor': 1.3},
            'yolox_s': {'params': 9_000_000, 'flops': 26.8e9, 'speed_factor': 1.1},
            'yolox_m': {'params': 25_300_000, 'flops': 73.8e9, 'speed_factor': 0.9},
            'yolox_l': {'params': 54_200_000, 'flops': 155.6e9, 'speed_factor': 0.7},
            'yolox_x': {'params': 99_100_000, 'flops': 281.9e9, 'speed_factor': 0.5},
            
            # YOLOv3 variants
            'yolov3': {'params': 61_900_000, 'flops': 156.4e9, 'speed_factor': 0.6},
            
            # PP-YOLOE variants
            'ppyoloe_s': {'params': 7_900_000, 'flops': 17.4e9, 'speed_factor': 1.0},
            'ppyoloe_m': {'params': 23_400_000, 'flops': 49.9e9, 'speed_factor': 0.8},
            'ppyoloe_l': {'params': 52_200_000, 'flops': 110.1e9, 'speed_factor': 0.6},
            
            # YOLOv11 variants
            'yolov11n': {'params': 2_600_000, 'flops': 6.5e9, 'speed_factor': 1.3},
            'yolov11s': {'params': 9_400_000, 'flops': 21.5e9, 'speed_factor': 1.1},
            'yolov11m': {'params': 20_100_000, 'flops': 68.2e9, 'speed_factor': 0.9},
            'yolov11l': {'params': 25_300_000, 'flops': 86.9e9, 'speed_factor': 0.7},
        }
        
        return params_map.get(variant, {'params': 10_000_000, 'flops': 50e9, 'speed_factor': 1.0})
    
    def load_model(self) -> None:
        """Load the mock model."""
        # Simulate model loading
        time.sleep(0.1)
        
        # Create a simple mock model
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 85 * 3)  # 85 classes * 3 anchors (simplified)
        )
        
        # Move to device
        if self.device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.model.eval()
        self.is_loaded = True
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for mock model."""
        # Resize image
        height, width = self.input_size
        resized = cv2.resize(image, (width, height))
        
        # Convert to tensor and normalize
        tensor = torch.from_numpy(resized).float()
        tensor = tensor.permute(2, 0, 1)  # HWC to CHW
        tensor = tensor / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        # Move to device
        if self.device == 'cuda' and torch.cuda.is_available():
            tensor = tensor.cuda()
        
        return tensor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through mock model."""
        # Simulate inference time based on model complexity
        base_time = 0.01  # Base inference time
        complexity_factor = self.mock_params['params'] / 10_000_000  # Normalize by 10M params
        inference_time = base_time * complexity_factor / self.mock_params['speed_factor']
        time.sleep(max(0.005, min(0.1, inference_time)))  # Clamp between 5ms and 100ms
        
        return self.model(x)
    
    def postprocess_predictions(self, predictions: torch.Tensor, 
                               original_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Postprocess mock predictions."""
        # Generate mock detections
        detections = []
        
        # Number of detections varies by model complexity
        max_detections = max(1, min(15, int(self.mock_params['params'] / 5_000_000)))
        num_detections = random.randint(1, max_detections)
        
        original_height, original_width = original_shape
        
        # COCO class names
        class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                      'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                      'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                      'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                      'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                      'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                      'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                      'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                      'hair drier', 'toothbrush']
        
        # Hot dog is at index 52 in COCO dataset
        hot_dog_class_id = 52
        
        for i in range(num_detections):
            # Generate bounding box in percentage coordinates (0.0 to 1.0)
            # Center coordinates
            center_x = random.uniform(0.1, 0.9)
            center_y = random.uniform(0.1, 0.9)
            
            # Width and height as percentages
            bbox_width = random.uniform(0.05, 0.4)  # 5% to 40% of image width
            bbox_height = random.uniform(0.05, 0.4)  # 5% to 40% of image height
            
            # Convert center format to top-left format (x, y, width, height)
            x = max(0.0, center_x - bbox_width / 2)
            y = max(0.0, center_y - bbox_height / 2)
            
            # Ensure bbox doesn't go outside image bounds
            if x + bbox_width > 1.0:
                bbox_width = 1.0 - x
            if y + bbox_height > 1.0:
                bbox_height = 1.0 - y
            
            # Bias toward detecting hot dogs (higher probability)
            if random.random() < 0.6:  # 60% chance of hot dog
                class_id = hot_dog_class_id
                # Hot dogs get higher confidence
                confidence = random.uniform(max(self.conf_threshold, 0.6), 0.95)
            else:
                # Other common food items and objects
                food_classes = [47, 48, 49, 50, 51, 53, 54, 55]  # banana, apple, sandwich, orange, broccoli, pizza, donut, cake
                common_classes = [0, 2, 15, 16, 56, 57, 59, 60, 61, 62]  # person, car, cat, dog, chair, couch, dining table, toilet, tv, laptop
                
                if random.random() < 0.4:  # 40% chance of food items
                    class_id = random.choice(food_classes)
                else:
                    class_id = random.choice(common_classes)
                
                confidence = random.uniform(self.conf_threshold, 0.85)
            
            detection = {
                'bbox': [x, y, bbox_width, bbox_height],  # [x, y, width, height] in percentage coordinates
                'confidence': confidence,
                'class_id': class_id,
                'class_name': class_names[class_id]
            }
            
            detections.append(detection)
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model information."""
        info = super().get_model_info()
        
        # Add mock-specific information
        info.update({
            'total_parameters': self.mock_params['params'],
            'trainable_parameters': self.mock_params['params'],
            'model_size_mb': self.mock_params['params'] * 4 / (1024 * 1024),
            'flops': self.mock_params['flops'],
            'is_mock': True
        })
        
        return info


# Import cv2 here to avoid circular imports
import cv2