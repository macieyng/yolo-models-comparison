#!/usr/bin/env python3
"""
Main script for YOLO Model Comparison System
Compares YOLOv6, PP-YOLOE, YOLOv11, YOLOX, and YOLOv3 models on 100 images.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import time
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.data_utils import (
    load_config, setup_directories, ImageDataLoader, 
    TestDataGenerator, ModelWeightDownloader, COCODataset
)
from utils.evaluation import DetectionEvaluator, PerformanceComparator
from utils.base_model import BaseYOLOModel


def setup_logging(log_dir: str) -> logging.Logger:
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('yolo_comparison')
    logger.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, 'comparison.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def download_and_prepare_models(config: Dict[str, Any], weights_dir: str, 
                               logger: logging.Logger) -> Dict[str, str]:
    """Download and prepare all model weights."""
    logger.info("Starting model weight download...")
    
    downloader = ModelWeightDownloader(weights_dir)
    weight_paths = downloader.download_all_weights(config)
    
    logger.info(f"Downloaded {len(weight_paths)} model weights")
    for model_name, path in weight_paths.items():
        logger.info(f"  {model_name}: {path}")
    
    return weight_paths


def setup_test_data(config: Dict[str, Any], directories: Dict[str, str], 
                   logger: logging.Logger, custom_image_dir: str = None) -> tuple:
    """Set up test data (images and annotations)."""
    logger.info("Setting up test data...")
    
    # Load class names from config
    class_names = config['dataset']['class_names']
    num_images = config['dataset']['num_images']
    
    # Create image data loader
    image_loader = ImageDataLoader()
    
    # Use custom image directory if provided, otherwise use default
    if custom_image_dir is not None and os.path.exists(custom_image_dir):
        images_dir = custom_image_dir
        logger.info(f"Using custom image directory: {images_dir}")
    else:
        images_dir = directories['images']
        logger.info(f"Using default image directory: {images_dir}")
    
    # Check if images exist, if not create sample images (only for default directory)
    if not os.path.exists(images_dir) or len(os.listdir(images_dir)) == 0:
        if custom_image_dir:
            logger.error(f"Custom image directory '{custom_image_dir}' is empty or does not exist")
            raise ValueError(f"Custom image directory '{custom_image_dir}' is empty or does not exist")
        else:
            logger.info("No images found, downloading sample images...")
            downloaded_paths = image_loader.download_sample_images(images_dir, num_images)
            logger.info(f"Downloaded {len(downloaded_paths)} sample images")
    
    # Load images
    image_data = image_loader.load_images_from_directory(images_dir, limit=num_images)
    logger.info(f"Loaded {len(image_data)} images for testing")
    
    # Generate synthetic annotations for testing
    test_generator = TestDataGenerator(class_names)
    annotations_path = os.path.join(directories['annotations'], 'test_annotations.json')
    
    if not os.path.exists(annotations_path):
        logger.info("Generating synthetic annotations...")
        test_generator.create_test_dataset(images_dir, annotations_path, len(image_data))
        logger.info(f"Created annotations file: {annotations_path}")
    
    # Load annotations
    coco_dataset = COCODataset(images_dir, annotations_path)
    
    return image_data, coco_dataset


def create_model_instances(config: Dict[str, Any], weight_paths: Dict[str, str], 
                          logger: logging.Logger) -> List[BaseYOLOModel]:
    """Create instances of all YOLO models."""
    logger.info("Creating model instances...")
    
    models = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Import model implementations
    try:
        from models.yolov6_model import YOLOv6Model
        from models.yolox_model import YOLOXModel
        from models.yolov3_model import YOLOv3Model
        from models.ppyoloe_model import PPYOLOEModel
        from models.yolov11_model import YOLOv11Model
        
        model_classes = {
            'yolov6': YOLOv6Model,
            'yolox': YOLOXModel,
            'yolov3': YOLOv3Model,
            'ppyoloe': PPYOLOEModel,
            'yolov11': YOLOv11Model
        }
        
    except ImportError as e:
        logger.warning(f"Some model implementations not found: {e}")
        logger.warning("Will use mock models for demonstration")
        from models.mock_model import MockYOLOModel
        
        model_classes = {
            'yolov6': MockYOLOModel,
            'yolox': MockYOLOModel,
            'yolov3': MockYOLOModel,
            'ppyoloe': MockYOLOModel,
            'yolov11': MockYOLOModel
        }
    
    # Create model instances
    for model_type, model_config in config['models'].items():
        if model_type not in model_classes:
            logger.warning(f"Model type {model_type} not supported, skipping...")
            continue
        
        model_class = model_classes[model_type]
        
        for variant in model_config['variants']:
            model_key = f"{model_type}_{variant}"
            
            if model_key not in weight_paths:
                logger.warning(f"Weights not found for {model_key}, skipping...")
                continue
            
            try:
                model = model_class(
                    model_name=model_config['name'],
                    variant=variant,
                    weights_path=weight_paths[model_key],
                    input_size=tuple(model_config['input_size']),
                    conf_threshold=config['evaluation']['confidence_threshold'],
                    iou_threshold=config['evaluation']['iou_threshold'],
                    device=device
                )
                
                # Load model
                model.load_model()
                models.append(model)
                logger.info(f"Created model: {model}")
                
            except Exception as e:
                logger.error(f"Failed to create model {model_key}: {str(e)}")
    
    logger.info(f"Created {len(models)} model instances")
    return models


def run_model_evaluation(models: List[BaseYOLOModel], image_data: List[tuple], 
                        coco_dataset: COCODataset, config: Dict[str, Any], 
                        logger: logging.Logger) -> tuple:
    """Run evaluation on all models."""
    logger.info("Starting model evaluation...")
    
    # Create evaluator
    evaluator = DetectionEvaluator(config['dataset']['class_names'])
    comparator = PerformanceComparator()
    
    # Prepare ground truth data
    ground_truths = []
    for image_path, image in image_data:
        # Extract image ID from filename (simplified)
        image_id = int(os.path.basename(image_path).split('.')[0].split('_')[-1])
        
        # Get annotations for this image
        annotations = coco_dataset.get_image_annotations(image_id)
        
        # Convert to our format
        gt_objects = []
        for ann in annotations:
            gt_objects.append({
                'bbox': ann['bbox'],
                'class_id': ann['category_id'],
                'class_name': config['dataset']['class_names'][ann['category_id']]
            })
        
        ground_truths.append(gt_objects)
    
    # Evaluate each model
    results = {}
    
    for model in models:
        logger.info(f"Evaluating {model}...")
        
        try:
            # Run inference on all images
            model.reset_performance_stats()
            all_predictions = []
            
            for i, (image_path, image) in enumerate(image_data):
                logger.info(f"  Processing image {i+1}/{len(image_data)}: {os.path.basename(image_path)}")
                
                # Run prediction
                predictions, inference_time = model.predict(image)
                all_predictions.append(predictions)
            
            # Evaluate predictions
            eval_results = evaluator.evaluate_model(
                all_predictions, ground_truths, str(model)
            )
            
            # Get model info and performance stats
            model_info = model.get_model_info()
            performance_stats = model.get_performance_stats()
            
            # Add to comparator
            comparator.add_model_results(
                str(model), eval_results, model_info, performance_stats
            )
            
            results[str(model)] = {
                'evaluation': eval_results,
                'model_info': model_info,
                'performance': performance_stats
            }
            
            logger.info(f"  Completed {model}: mAP@0.5 = {eval_results['overall_metrics']['mAP@0.5']:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model}: {str(e)}")
            continue
    
    return results, comparator


def save_results(results: Dict[str, Any], comparator: PerformanceComparator, 
                directories: Dict[str, str], config: Dict[str, Any], 
                logger: logging.Logger, custom_output_file: str = None):
    """Save all results and generate reports."""
    logger.info("Saving results...")
    
    results_dir = directories['results']
    
    # Save individual model results (always save to default location)
    for model_name, model_results in results.items():
        output_file = os.path.join(results_dir, f"{model_name.replace(' ', '_')}_results.json")
        with open(output_file, 'w') as f:
            json.dump(model_results, f, indent=2)
        logger.info(f"Saved {model_name} results to {output_file}")
    
    # Generate comparison report
    comparison_report = comparator.generate_comparison_report(
        os.path.join(results_dir, 'comparison_report.json')
    )
    
    # Save CSV summary
    comparator.save_results_csv(os.path.join(results_dir, 'comparison_summary.csv'))
    
    # Generate HTML report
    html_report = generate_html_report(comparison_report, results)
    with open(os.path.join(results_dir, 'comparison_report.html'), 'w') as f:
        f.write(html_report)
    
    # Handle custom output file if specified
    if custom_output_file is not None:
        logger.info(f"Saving custom output to: {custom_output_file}")
        custom_output_dir = os.path.dirname(custom_output_file)
        if custom_output_dir:
            os.makedirs(custom_output_dir, exist_ok=True)
        
        # Determine output format based on file extension
        file_ext = os.path.splitext(custom_output_file)[1].lower()
        
        if file_ext == '.json':
            # Save combined results as JSON
            combined_results = {
                'comparison_report': comparison_report,
                'model_results': results,
                'config': config
            }
            with open(custom_output_file, 'w') as f:
                json.dump(combined_results, f, indent=2)
                
        elif file_ext == '.csv':
            # Save comparison summary as CSV
            comparator.save_results_csv(custom_output_file)
            
        elif file_ext == '.html':
            # Save HTML report
            with open(custom_output_file, 'w') as f:
                f.write(html_report)
                
        else:
            # Default to JSON format
            logger.warning(f"Unknown output format '{file_ext}', defaulting to JSON")
            combined_results = {
                'comparison_report': comparison_report,
                'model_results': results,
                'config': config
            }
            with open(custom_output_file, 'w') as f:
                json.dump(combined_results, f, indent=2)
        
        logger.info(f"Custom output saved to: {custom_output_file}")
    
    logger.info("Results saved successfully")


def generate_html_report(comparison_report: Dict[str, Any], results: Dict[str, Any]) -> str:
    """Generate HTML report for the comparison."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLO Model Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .model-results {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>YOLO Model Comparison Report</h1>
            <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Number of models evaluated: {comparison_report.get('summary', {}).get('num_models', 0)}</p>
        </div>
        
        <div class="section">
            <h2>Summary</h2>
            <ul>
                <li><strong>Best mAP@0.5:</strong> {comparison_report.get('summary', {}).get('best_map_50', {}).get('model', 'N/A')} 
                    ({comparison_report.get('summary', {}).get('best_map_50', {}).get('value', 0):.3f})</li>
                <li><strong>Best mAP@0.5:0.95:</strong> {comparison_report.get('summary', {}).get('best_map_50_95', {}).get('model', 'N/A')} 
                    ({comparison_report.get('summary', {}).get('best_map_50_95', {}).get('value', 0):.3f})</li>
                <li><strong>Fastest model:</strong> {comparison_report.get('summary', {}).get('fastest_model', {}).get('model', 'N/A')} 
                    ({comparison_report.get('summary', {}).get('fastest_model', {}).get('fps', 0):.1f} FPS)</li>
                <li><strong>Smallest model:</strong> {comparison_report.get('summary', {}).get('smallest_model', {}).get('model', 'N/A')} 
                    ({comparison_report.get('summary', {}).get('smallest_model', {}).get('params', 0):,} parameters)</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>Detailed Results</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>mAP@0.5</th>
                    <th>mAP@0.5:0.95</th>
                    <th>Avg FPS</th>
                    <th>Parameters</th>
                    <th>Model Size (MB)</th>
                </tr>
    """
    
    for model_name, model_results in results.items():
        eval_metrics = model_results['evaluation']['overall_metrics']
        model_info = model_results['model_info']
        performance = model_results['performance']
        
        html += f"""
                <tr>
                    <td>{model_name}</td>
                    <td>{eval_metrics.get('mAP@0.5', 0):.3f}</td>
                    <td>{eval_metrics.get('mAP@0.5:0.95', 0):.3f}</td>
                    <td>{performance.get('avg_fps', 0):.1f}</td>
                    <td>{model_info.get('total_parameters', 0):,}</td>
                    <td>{model_info.get('model_size_mb', 0):.1f}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            <ul>
    """
    
    for rec_type, rec_text in comparison_report.get('recommendations', {}).items():
        html += f"<li><strong>{rec_type.replace('_', ' ').title()}:</strong> {rec_text}</li>"
    
    html += """
            </ul>
        </div>
    </body>
    </html>
    """
    
    return html


def main():
    """Main function to run the YOLO model comparison."""
    parser = argparse.ArgumentParser(description='YOLO Model Comparison System')
    parser.add_argument('--config', '-c', type=str, default='configs/model_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', '-o', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--image-dir', '--image_dir', type=str, default=None,
                       help='Directory containing images for inference')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path for results (supports .json, .csv, .html)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup directories
    directories = setup_directories('.')
    
    # Setup logging
    logger = setup_logging(directories['logs'])
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("Starting YOLO Model Comparison System")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Download and prepare models
        weight_paths = download_and_prepare_models(config, directories['weights'], logger)
        
        # Setup test data
        image_data, coco_dataset = setup_test_data(config, directories, logger, args.image_dir)
        
        # Create model instances
        models = create_model_instances(config, weight_paths, logger)
        
        if not models:
            logger.error("No models were successfully created. Exiting.")
            return 1
        
        # Run evaluation
        results, comparator = run_model_evaluation(models, image_data, coco_dataset, config, logger)
        
        # Save results
        save_results(results, comparator, directories, config, logger, args.output)
        
        logger.info("YOLO Model Comparison completed successfully!")
        logger.info(f"Results saved to: {directories['results']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}")
        return 1


if __name__ == "__main__":
    import torch  # Import here to avoid issues if not installed
    sys.exit(main())