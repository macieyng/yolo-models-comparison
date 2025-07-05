#!/usr/bin/env python3
"""
Inference-focused YOLO Model Comparison Script
Compares inference results across different YOLO models without ground truth evaluation.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.data_utils import (
    load_config, setup_directories, ImageDataLoader, 
    ModelWeightDownloader
)
from utils.inference_comparator import InferenceComparator
from utils.base_model import BaseYOLOModel


def setup_logging(log_dir: str, verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('yolo_inference_comparison')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, 'inference_comparison.log'))
    fh.setLevel(logging.DEBUG if verbose else logging.INFO)
    fh.setFormatter(formatter)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
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


def load_images(image_dir: str, logger: logging.Logger, limit: Optional[int] = None) -> List[tuple]:
    """Load images from directory."""
    logger.info(f"Loading images from: {image_dir}")
    
    # Create image data loader
    image_loader = ImageDataLoader()
    
    # Load images
    image_data = image_loader.load_images_from_directory(image_dir, limit=limit)
    logger.info(f"Loaded {len(image_data)} images for inference")
    
    return image_data


def create_model_instances(config: Dict[str, Any], weight_paths: Dict[str, str], 
                          logger: logging.Logger) -> List[BaseYOLOModel]:
    """Create instances of all YOLO models."""
    logger.info("Creating model instances...")
    
    models = []
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
    except ImportError:
        device = 'cpu'
        logger.warning("PyTorch not available, using CPU mode")
    
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


def run_inference_comparison(models: List[BaseYOLOModel], image_data: List[tuple], 
                           config: Dict[str, Any], logger: logging.Logger) -> InferenceComparator:
    """Run inference comparison on all models."""
    logger.info("Starting inference comparison...")
    
    # Create inference comparator
    comparator = InferenceComparator(
        class_names=config['dataset']['class_names'],
        confidence_threshold=config['evaluation']['confidence_threshold']
    )
    
    # Run inference for each model on each image
    total_combinations = len(models) * len(image_data)
    combination_count = 0
    
    for model in models:
        logger.info(f"Running inference with {model}...")
        
        try:
            # Reset performance stats for this model
            model.reset_performance_stats()
            
            for i, (image_path, image) in enumerate(image_data):
                combination_count += 1
                logger.info(f"  [{combination_count}/{total_combinations}] Processing: {os.path.basename(image_path)}")
                
                # Run prediction
                predictions, inference_time = model.predict(image)
                
                # Add results to comparator
                comparator.add_model_results(str(model), image_path, predictions)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    logger.info(f"  Processed {i + 1}/{len(image_data)} images")
            
            logger.info(f"  Completed {model}")
            
        except Exception as e:
            logger.error(f"Failed to run inference with {model}: {str(e)}")
            continue
    
    logger.info("Inference comparison completed")
    return comparator


def save_results(comparator: InferenceComparator, output_path: str, 
                config: Dict[str, Any], logger: logging.Logger):
    """Save inference comparison results."""
    logger.info("Saving inference comparison results...")
    
    # Determine output format based on file extension
    if output_path:
        file_ext = os.path.splitext(output_path)[1].lower()
        output_dir = os.path.dirname(output_path)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        if file_ext == '.html':
            # Generate HTML report
            logger.info("Generating HTML report...")
            comparator.generate_html_report(output_path)
            logger.info(f"HTML report saved to: {output_path}")
            
        elif file_ext == '.csv':
            # Generate CSV table
            logger.info("Generating CSV table...")
            table_df = comparator.generate_detection_table()
            table_df.to_csv(output_path, index=False)
            logger.info(f"CSV table saved to: {output_path}")
            
        elif file_ext == '.json':
            # Generate JSON results
            logger.info("Generating JSON results...")
            results_data = {
                'detection_table': comparator.generate_detection_table().to_dict('records'),
                'summary_statistics': comparator.generate_summary_statistics(),
                'detailed_results': comparator.inference_results,
                'config': config
            }
            with open(output_path, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            logger.info(f"JSON results saved to: {output_path}")
            
        else:
            logger.warning(f"Unknown output format '{file_ext}', defaulting to JSON")
            with open(output_path, 'w') as f:
                json.dump(comparator.inference_results, f, indent=2, default=str)
    
    # Always save to default results directory as well
    results_dir = os.path.join(os.path.dirname(output_path) if output_path else '.', 'results')
    saved_files = comparator.save_results(results_dir)
    
    logger.info("Results saved to:")
    for result_type, file_path in saved_files.items():
        logger.info(f"  {result_type}: {file_path}")


def main():
    """Main function for inference comparison."""
    parser = argparse.ArgumentParser(description='YOLO Models Inference Comparison')
    parser.add_argument('--config', '-c', type=str, default='configs/model_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--image-dir', '--image_dir', type=str, required=True,
                       help='Directory containing images for inference')
    parser.add_argument('--output', type=str, default='inference_comparison.html',
                       help='Output file path (supports .html, .json, .csv)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximum number of images to process')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory '{args.image_dir}' does not exist")
        return 1
    
    if not os.path.isdir(args.image_dir):
        print(f"Error: '{args.image_dir}' is not a directory")
        return 1
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Setup directories
    directories = setup_directories('.')
    
    # Setup logging
    logger = setup_logging(directories['logs'], args.verbose)
    
    logger.info("Starting YOLO Models Inference Comparison")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Image directory: {args.image_dir}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Image limit: {args.limit or 'No limit'}")
    
    try:
        # Download and prepare models
        weight_paths = download_and_prepare_models(config, directories['weights'], logger)
        
        # Load images
        image_data = load_images(args.image_dir, logger, args.limit)
        
        if not image_data:
            logger.error("No images found in the specified directory")
            return 1
        
        # Create model instances
        models = create_model_instances(config, weight_paths, logger)
        
        if not models:
            logger.error("No models were successfully created. Exiting.")
            return 1
        
        # Run inference comparison
        comparator = run_inference_comparison(models, image_data, config, logger)
        
        # Save results
        save_results(comparator, args.output, config, logger)
        
        logger.info("Inference comparison completed successfully!")
        logger.info(f"Results saved to: {args.output}")
        
        # Print summary
        stats = comparator.generate_summary_statistics()
        logger.info("\nSummary:")
        logger.info(f"  Total images processed: {stats['total_images']}")
        logger.info(f"  Total models compared: {stats['total_models']}")
        
        for model_name, model_stats in stats['model_summaries'].items():
            logger.info(f"  {model_name}: {model_stats['total_detections']} detections across {model_stats['images_with_detections']} images")
        
        return 0
        
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())