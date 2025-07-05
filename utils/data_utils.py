"""
Data utilities for YOLO model comparison system.
Handles image loading, COCO format, and test data management.
"""

import os
import json
import random
import requests
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import cv2
from PIL import Image
import yaml
from pathlib import Path


class COCODataset:
    """Handler for COCO dataset format."""
    
    def __init__(self, images_dir: str, annotations_file: str = None):
        """
        Initialize COCO dataset handler.
        
        Args:
            images_dir: Directory containing images
            annotations_file: Path to COCO annotations JSON file
        """
        self.images_dir = Path(images_dir)
        self.annotations_file = annotations_file
        self.annotations = {}
        self.images = []
        self.categories = []
        
        if annotations_file and os.path.exists(annotations_file):
            self.load_annotations()
    
    def load_annotations(self):
        """Load COCO annotations from JSON file."""
        with open(self.annotations_file, 'r') as f:
            data = json.load(f)
        
        self.annotations = {ann['image_id']: [] for ann in data['annotations']}
        for ann in data['annotations']:
            self.annotations[ann['image_id']].append(ann)
        
        self.images = {img['id']: img for img in data['images']}
        self.categories = {cat['id']: cat for cat in data['categories']}
    
    def get_image_annotations(self, image_id: int) -> List[Dict[str, Any]]:
        """
        Get annotations for a specific image.
        
        Args:
            image_id: Image ID
            
        Returns:
            List of annotation dictionaries
        """
        return self.annotations.get(image_id, [])
    
    def convert_bbox_format(self, bbox: List[float], from_format: str = 'xywh', 
                           to_format: str = 'xyxy') -> List[float]:
        """
        Convert bounding box format.
        
        Args:
            bbox: Bounding box coordinates
            from_format: Source format ('xywh' or 'xyxy')
            to_format: Target format ('xywh' or 'xyxy')
            
        Returns:
            Converted bounding box
        """
        if from_format == to_format:
            return bbox.copy()
        
        if from_format == 'xywh' and to_format == 'xyxy':
            return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        elif from_format == 'xyxy' and to_format == 'xywh':
            return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        else:
            raise ValueError(f"Unsupported conversion: {from_format} -> {to_format}")


class ImageDataLoader:
    """Loader for image data with various formats and sources."""
    
    def __init__(self, supported_formats: List[str] = None):
        """
        Initialize image data loader.
        
        Args:
            supported_formats: List of supported image formats
        """
        self.supported_formats = supported_formats or ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array (H, W, C) in BGR format
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Try OpenCV first
        try:
            image = cv2.imread(image_path)
            if image is not None:
                return image
        except Exception:
            pass
        
        # Try PIL as fallback
        try:
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            image = np.array(pil_image)
            # Convert RGB to BGR for OpenCV compatibility
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {str(e)}")
    
    def load_images_from_directory(self, directory: str, limit: Optional[int] = None) -> List[Tuple[str, np.ndarray]]:
        """
        Load images from a directory.
        
        Args:
            directory: Directory containing images
            limit: Maximum number of images to load
            
        Returns:
            List of (image_path, image_array) tuples
        """
        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(directory_path.glob(f"*{ext}"))
            image_files.extend(directory_path.glob(f"*{ext.upper()}"))
        
        if limit:
            image_files = image_files[:limit]
        
        images = []
        for image_path in image_files:
            try:
                image = self.load_image(str(image_path))
                images.append((str(image_path), image))
            except Exception as e:
                print(f"Warning: Failed to load {image_path}: {str(e)}")
        
        return images
    
    def download_sample_images(self, output_dir: str, num_images: int = 100) -> List[str]:
        """
        Download sample images for testing.
        
        Args:
            output_dir: Directory to save downloaded images
            num_images: Number of images to download
            
        Returns:
            List of downloaded image paths
        """
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Sample image URLs (placeholder - in real implementation, use actual dataset)
        sample_urls = [
            "https://via.placeholder.com/640x480/FF0000/FFFFFF?text=Sample+Image+{i}"
            for i in range(1, num_images + 1)
        ]
        
        downloaded_paths = []
        for i, url in enumerate(sample_urls[:num_images]):
            try:
                response = requests.get(url.format(i=i+1), timeout=10)
                response.raise_for_status()
                
                image_path = output_dir_path / f"sample_image_{i+1:03d}.jpg"
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                
                downloaded_paths.append(str(image_path))
                
            except Exception as e:
                print(f"Warning: Failed to download image {i+1}: {str(e)}")
        
        return downloaded_paths


class TestDataGenerator:
    """Generator for test data and synthetic annotations."""
    
    def __init__(self, class_names: List[str]):
        """
        Initialize test data generator.
        
        Args:
            class_names: List of class names for annotations
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def generate_synthetic_annotations(self, image_shape: Tuple[int, int], 
                                     num_objects: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate synthetic annotations for testing.
        
        Args:
            image_shape: Image shape (height, width)
            num_objects: Number of objects to generate
            
        Returns:
            List of annotation dictionaries
        """
        if num_objects is None:
            num_objects = random.randint(1, 5)
        
        annotations = []
        height, width = image_shape
        
        for _ in range(num_objects):
            # Generate random bounding box
            x1 = random.randint(0, width - 50)
            y1 = random.randint(0, height - 50)
            x2 = random.randint(x1 + 20, min(x1 + 200, width))
            y2 = random.randint(y1 + 20, min(y1 + 200, height))
            
            # Generate random class
            class_id = random.randint(0, self.num_classes - 1)
            
            annotation = {
                'bbox': [x1, y1, x2, y2],
                'class_id': class_id,
                'class_name': self.class_names[class_id],
                'area': (x2 - x1) * (y2 - y1),
                'iscrowd': 0
            }
            annotations.append(annotation)
        
        return annotations
    
    def create_test_dataset(self, images_dir: str, annotations_output: str, 
                          num_images: int = 100) -> str:
        """
        Create a test dataset with synthetic annotations.
        
        Args:
            images_dir: Directory containing images
            annotations_output: Output path for annotations JSON
            num_images: Number of images to process
            
        Returns:
            Path to created annotations file
        """
        loader = ImageDataLoader()
        images_data = loader.load_images_from_directory(images_dir, limit=num_images)
        
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Add categories
        for i, class_name in enumerate(self.class_names):
            coco_data['categories'].append({
                'id': i,
                'name': class_name,
                'supercategory': 'object'
            })
        
        annotation_id = 1
        
        for image_id, (image_path, image) in enumerate(images_data, 1):
            # Add image info
            height, width = image.shape[:2]
            coco_data['images'].append({
                'id': image_id,
                'file_name': os.path.basename(image_path),
                'width': width,
                'height': height
            })
            
            # Generate synthetic annotations
            annotations = self.generate_synthetic_annotations((height, width))
            
            for ann in annotations:
                coco_data['annotations'].append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': ann['class_id'],
                    'bbox': ann['bbox'],
                    'area': ann['area'],
                    'iscrowd': ann['iscrowd']
                })
                annotation_id += 1
        
        # Save annotations
        with open(annotations_output, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        return annotations_output


class ModelWeightDownloader:
    """Downloader for model weights."""
    
    def __init__(self, weights_dir: str):
        """
        Initialize weight downloader.
        
        Args:
            weights_dir: Directory to store downloaded weights
        """
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
    
    def download_weights(self, url: str, model_name: str, force_download: bool = False) -> str:
        """
        Download model weights from URL.
        
        Args:
            url: URL to download weights from
            model_name: Name for the downloaded file
            force_download: Force re-download even if file exists
            
        Returns:
            Path to downloaded weights file
        """
        # Extract filename from URL or use model name
        filename = url.split('/')[-1]
        if not filename or '.' not in filename:
            filename = f"{model_name}.pth"
        
        output_path = self.weights_dir / filename
        
        if output_path.exists() and not force_download:
            print(f"Weights already exist: {output_path}")
            return str(output_path)
        
        try:
            print(f"Downloading weights: {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rDownloading: {progress:.1f}%", end='', flush=True)
            
            print(f"\nDownload completed: {output_path}")
            return str(output_path)
            
        except Exception as e:
            if output_path.exists():
                output_path.unlink()
            raise RuntimeError(f"Failed to download weights from {url}: {str(e)}")
    
    def download_all_weights(self, config: Dict[str, Any]) -> Dict[str, str]:
        """
        Download all model weights from configuration.
        
        Args:
            config: Configuration dictionary with model weight URLs
            
        Returns:
            Dictionary mapping model names to downloaded weight paths
        """
        weight_paths = {}
        
        for model_type, model_config in config.get('models', {}).items():
            weight_urls = model_config.get('weight_urls', {})
            
            for variant, url in weight_urls.items():
                model_name = f"{model_type}_{variant}"
                try:
                    weight_path = self.download_weights(url, model_name)
                    weight_paths[model_name] = weight_path
                except Exception as e:
                    print(f"Warning: Failed to download {model_name}: {str(e)}")
        
        return weight_paths


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_directories(base_dir: str) -> Dict[str, str]:
    """
    Set up directory structure for the comparison system.
    
    Args:
        base_dir: Base directory for the project
        
    Returns:
        Dictionary of directory paths
    """
    base_path = Path(base_dir)
    
    directories = {
        'images': base_path / 'data' / 'images',
        'annotations': base_path / 'data' / 'annotations',
        'weights': base_path / 'weights',
        'results': base_path / 'data' / 'results',
        'logs': base_path / 'logs',
        'visualizations': base_path / 'data' / 'visualizations'
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return {key: str(path) for key, path in directories.items()}