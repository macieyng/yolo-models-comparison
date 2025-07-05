#!/usr/bin/env python3
"""
Inference Comparator for YOLO Models
Compares inference results across different models without ground truth evaluation.
"""

import os
import json
import csv
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
from pathlib import Path

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class InferenceComparator:
    """Compare inference results across different YOLO models."""
    
    def __init__(self, class_names: List[str], confidence_threshold: float = 0.5):
        """
        Initialize the inference comparator.
        
        Args:
            class_names: List of class names for object detection
            confidence_threshold: Minimum confidence threshold for detections
        """
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.inference_results = {}  # {model_name: {image_path: detections}}
        self.image_paths = []
        self.model_names = []
        
    def add_model_results(self, model_name: str, image_path: str, detections: List[Dict[str, Any]]):
        """
        Add inference results for a specific model and image.
        
        Args:
            model_name: Name of the model
            image_path: Path to the image
            detections: List of detection dictionaries with keys:
                       - bbox: [x, y, width, height]
                       - class_id: integer class ID
                       - confidence: detection confidence
                       - class_name: string class name
        """
        if model_name not in self.inference_results:
            self.inference_results[model_name] = {}
            self.model_names.append(model_name)
            
        # Filter detections by confidence threshold
        filtered_detections = [
            det for det in detections 
            if det.get('confidence', 0) >= self.confidence_threshold
        ]
        
        self.inference_results[model_name][image_path] = filtered_detections
        
        if image_path not in self.image_paths:
            self.image_paths.append(image_path)
    
    def get_detection_counts(self, model_name: str, image_path: str) -> Dict[str, int]:
        """
        Get class detection counts for a specific model and image.
        
        Args:
            model_name: Name of the model
            image_path: Path to the image
            
        Returns:
            Dictionary mapping class names to detection counts
        """
        if model_name not in self.inference_results:
            return {}
        
        if image_path not in self.inference_results[model_name]:
            return {}
        
        detections = self.inference_results[model_name][image_path]
        class_counts = Counter()
        
        for detection in detections:
            class_name = detection.get('class_name', 'unknown')
            class_counts[class_name] += 1
            
        return dict(class_counts)
    
    def generate_detection_table(self):
        """
        Generate a table showing detection counts for each image-model combination.
        
        Returns:
            pandas DataFrame if pandas is available, otherwise dict
        """
        # Create a structured table
        table_data = []
        
        for image_path in self.image_paths:
            row = {'Image': os.path.basename(image_path)}
            
            for model_name in self.model_names:
                counts = self.get_detection_counts(model_name, image_path)
                
                # Format detection counts as a readable string
                if counts:
                    count_str = ", ".join([f"{cls}: {cnt}" for cls, cnt in counts.items()])
                else:
                    count_str = "No detections"
                
                row[model_name] = count_str
                
            table_data.append(row)
        
        # Return pandas DataFrame if available, otherwise return raw data
        if PANDAS_AVAILABLE:
            return pd.DataFrame(table_data)
        else:
            return table_data
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """
        Generate summary statistics across all models and images.
        
        Returns:
            Dictionary containing various summary statistics
        """
        stats = {
            'total_images': len(self.image_paths),
            'total_models': len(self.model_names),
            'model_summaries': {},
            'class_distribution': defaultdict(lambda: defaultdict(int)),
            'detection_overlap': {}
        }
        
        # Per-model statistics
        for model_name in self.model_names:
            model_stats = {
                'total_detections': 0,
                'images_with_detections': 0,
                'avg_detections_per_image': 0.0,
                'class_counts': defaultdict(int),
                'confidence_stats': []
            }
            
            confidences = []
            
            for image_path in self.image_paths:
                detections = self.inference_results.get(model_name, {}).get(image_path, [])
                
                if detections:
                    model_stats['images_with_detections'] += 1
                    model_stats['total_detections'] += len(detections)
                    
                    for detection in detections:
                        class_name = detection.get('class_name', 'unknown')
                        model_stats['class_counts'][class_name] += 1
                        stats['class_distribution'][class_name][model_name] += 1
                        
                        if 'confidence' in detection:
                            confidences.append(detection['confidence'])
            
            if confidences:
                model_stats['confidence_stats'] = {
                    'mean': np.mean(confidences),
                    'std': np.std(confidences),
                    'min': np.min(confidences),
                    'max': np.max(confidences)
                }
            
            if model_stats['images_with_detections'] > 0:
                model_stats['avg_detections_per_image'] = (
                    model_stats['total_detections'] / model_stats['images_with_detections']
                )
            
            stats['model_summaries'][model_name] = model_stats
        
        # Detection overlap analysis
        for image_path in self.image_paths:
            image_classes = set()
            model_classes = {}
            
            for model_name in self.model_names:
                detections = self.inference_results.get(model_name, {}).get(image_path, [])
                classes = set(det.get('class_name', 'unknown') for det in detections)
                model_classes[model_name] = classes
                image_classes.update(classes)
            
            if len(self.model_names) > 1:
                # Calculate intersection and union
                all_models_classes = set.intersection(*model_classes.values()) if model_classes else set()
                any_model_classes = set.union(*model_classes.values()) if model_classes else set()
                
                overlap_ratio = len(all_models_classes) / len(any_model_classes) if any_model_classes else 0
                
                stats['detection_overlap'][os.path.basename(image_path)] = {
                    'common_classes': list(all_models_classes),
                    'total_unique_classes': len(any_model_classes),
                    'overlap_ratio': overlap_ratio
                }
        
        return stats
    
    def create_detection_mosaic(self, image_path: str, output_path: str, 
                              max_width: int = 1200, max_height: int = 800) -> str:
        """
        Create a mosaic showing detections from all models for a single image.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the mosaic
            max_width: Maximum width for each model's subplot
            max_height: Maximum height for each model's subplot
            
        Returns:
            Path to the saved mosaic image
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required for creating detection mosaics")
        
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image_rgb.shape[:2]
        
        # Calculate subplot layout
        num_models = len(self.model_names)
        if num_models == 0:
            raise ValueError("No models added to comparator")
        
        # Determine grid layout
        cols = min(3, num_models)  # Maximum 3 columns
        rows = (num_models + cols - 1) // cols
        
        # Create figure
        fig = plt.figure(figsize=(5 * cols, 4 * rows))
        gs = GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.2)
        
        # Color palette for different classes
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))
        class_colors = {class_name: colors[i] for i, class_name in enumerate(self.class_names)}
        
        for idx, model_name in enumerate(self.model_names):
            row = idx // cols
            col = idx % cols
            
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(image_rgb)
            ax.set_title(f"{model_name}", fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # Get detections for this model
            detections = self.inference_results.get(model_name, {}).get(image_path, [])
            
            # Draw bounding boxes
            for detection in detections:
                bbox = detection.get('bbox', [0, 0, 0, 0])
                class_name = detection.get('class_name', 'unknown')
                confidence = detection.get('confidence', 0.0)
                
                # Convert bbox coordinates to proper format
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    
                    # Check if coordinates are in percentage (0-1) or absolute pixels
                    if x <= 1.0 and y <= 1.0 and w <= 1.0 and h <= 1.0:
                        # Convert percentage coordinates to absolute pixels
                        x = x * img_width
                        y = y * img_height
                        w = w * img_width
                        h = h * img_height
                    
                    # Ensure coordinates are within image bounds
                    x = max(0, min(x, img_width))
                    y = max(0, min(y, img_height))
                    w = max(1, min(w, img_width - x))
                    h = max(1, min(h, img_height - y))
                    
                    # Get color for this class
                    color = class_colors.get(class_name, 'red')
                    
                    # Draw bounding box
                    rect = patches.Rectangle(
                        (x, y), w, h,
                        linewidth=2, edgecolor=color, facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # Add label with improved positioning
                    label = f"{class_name} ({confidence:.2f})"
                    label_y = max(10, y - 5)  # Ensure label is visible
                    ax.text(x, label_y, label, fontsize=8, color=color, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Add overall title
        fig.suptitle(f"Detection Comparison: {os.path.basename(image_path)}", 
                    fontsize=16, fontweight='bold')
        
        # Save the mosaic
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_all_mosaics(self, output_dir: str) -> List[str]:
        """
        Create detection mosaics for all images.
        
        Args:
            output_dir: Directory to save mosaic images
            
        Returns:
            List of paths to created mosaic images
        """
        os.makedirs(output_dir, exist_ok=True)
        mosaic_paths = []
        
        for image_path in self.image_paths:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            mosaic_path = os.path.join(output_dir, f"{image_name}_mosaic.png")
            
            try:
                created_path = self.create_detection_mosaic(image_path, mosaic_path)
                mosaic_paths.append(created_path)
                print(f"Created mosaic: {created_path}")
            except Exception as e:
                print(f"Failed to create mosaic for {image_path}: {e}")
        
        return mosaic_paths
    
    def save_results(self, output_dir: str) -> Dict[str, str]:
        """
        Save all comparison results to files.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Dictionary mapping result types to file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        
        # Save detection table
        table_data = self.generate_detection_table()
        table_path = os.path.join(output_dir, 'detection_table.csv')
        
        if PANDAS_AVAILABLE and hasattr(table_data, 'to_csv'):
            # Use pandas to save CSV
            table_data.to_csv(table_path, index=False)
        else:
            # Manually create CSV
            with open(table_path, 'w', newline='') as csvfile:
                if table_data:
                    writer = csv.DictWriter(csvfile, fieldnames=table_data[0].keys())
                    writer.writeheader()
                    writer.writerows(table_data)
        
        saved_files['detection_table'] = table_path
        
        # Save summary statistics
        stats = self.generate_summary_statistics()
        stats_path = os.path.join(output_dir, 'summary_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        saved_files['summary_statistics'] = stats_path
        
        # Save detailed results
        detailed_path = os.path.join(output_dir, 'detailed_results.json')
        with open(detailed_path, 'w') as f:
            json.dump(self.inference_results, f, indent=2, default=str)
        saved_files['detailed_results'] = detailed_path
        
        # Create mosaics
        mosaics_dir = os.path.join(output_dir, 'mosaics')
        mosaic_paths = self.create_all_mosaics(mosaics_dir)
        saved_files['mosaics'] = mosaics_dir
        
        return saved_files
    
    def generate_html_report(self, output_path: str) -> str:
        """
        Generate an HTML report with tables and embedded mosaics.
        
        Args:
            output_path: Path to save the HTML report
            
        Returns:
            Path to the saved HTML report
        """
        table_data = self.generate_detection_table()
        stats = self.generate_summary_statistics()
        
        # Get current timestamp
        if PANDAS_AVAILABLE:
            timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        else:
            import datetime
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>YOLO Models Inference Comparison</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .stats {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .stat-card {{ 
                    border: 1px solid #ddd; 
                    padding: 15px; 
                    border-radius: 5px; 
                    min-width: 200px; 
                    flex: 1;
                }}
                .mosaic-grid {{ 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                    gap: 20px; 
                    margin: 20px 0;
                }}
                .mosaic-item {{ text-align: center; }}
                .mosaic-item img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                .detection-cell {{ font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>YOLO Models Inference Comparison Report</h1>
                <p><strong>Total Images:</strong> {stats['total_images']}</p>
                <p><strong>Models Compared:</strong> {', '.join(self.model_names)}</p>
                <p><strong>Generated:</strong> {timestamp}</p>
            </div>
            
            <div class="section">
                <h2>Detection Summary</h2>
                <div class="stats">
        """
        
        # Add model summary cards
        for model_name, model_stats in stats['model_summaries'].items():
            html_content += f"""
                    <div class="stat-card">
                        <h3>{model_name}</h3>
                        <p><strong>Total Detections:</strong> {model_stats['total_detections']}</p>
                        <p><strong>Images with Detections:</strong> {model_stats['images_with_detections']}</p>
                        <p><strong>Avg Detections/Image:</strong> {model_stats['avg_detections_per_image']:.2f}</p>
                        <p><strong>Top Classes:</strong> {', '.join([f"{cls} ({cnt})" for cls, cnt in sorted(model_stats['class_counts'].items(), key=lambda x: x[1], reverse=True)[:3]])}</p>
                    </div>
            """
        
        html_content += """
                </div>
            </div>
            
            <div class="section">
                <h2>Detection Table</h2>
                <table>
                    <thead>
                        <tr>
        """
        
        # Add table headers - handle both DataFrame and list cases
        if PANDAS_AVAILABLE and hasattr(table_data, 'columns'):
            # DataFrame case
            for col in table_data.columns:
                html_content += f"<th>{col}</th>"
        elif table_data:
            # List case - use keys from first row
            for col in table_data[0].keys():
                html_content += f"<th>{col}</th>"
        
        html_content += """
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Add table rows - handle both DataFrame and list cases
        if PANDAS_AVAILABLE and hasattr(table_data, 'iterrows'):
            # DataFrame case
            for _, row in table_data.iterrows():
                html_content += "<tr>"
                for col in table_data.columns:
                    cell_value = row[col]
                    if col == 'Image':
                        html_content += f"<td><strong>{cell_value}</strong></td>"
                    else:
                        html_content += f"<td class='detection-cell'>{cell_value}</td>"
                html_content += "</tr>"
        elif table_data:
            # List case
            for row in table_data:
                html_content += "<tr>"
                for col, cell_value in row.items():
                    if col == 'Image':
                        html_content += f"<td><strong>{cell_value}</strong></td>"
                    else:
                        html_content += f"<td class='detection-cell'>{cell_value}</td>"
                html_content += "</tr>"
        
        html_content += """
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>Detection Mosaics</h2>
                <div class="mosaic-grid">
        """
        
        # Add mosaic images (assuming they're in a 'mosaics' subdirectory)
        for image_path in self.image_paths:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            mosaic_name = f"{image_name}_mosaic.png"
            html_content += f"""
                    <div class="mosaic-item">
                        <h3>{os.path.basename(image_path)}</h3>
                        <img src="mosaics/{mosaic_name}" alt="Mosaic for {image_name}">
                    </div>
            """
        
        html_content += """
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path