#!/usr/bin/env python3
"""
Demo script for YOLO Model Comparison System
Runs a quick comparison with a smaller dataset for demonstration.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.data_utils import load_config, setup_directories
from main import (
    setup_logging, download_and_prepare_models, setup_test_data,
    create_model_instances, run_model_evaluation, save_results
)


def run_demo():
    """Run a quick demo of the comparison system."""
    print("=" * 60)
    print("YOLO Model Comparison System - Demo")
    print("=" * 60)
    
    # Load configuration
    config_path = "configs/model_config.yaml"
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        print("Please run setup.py first or create the configuration file.")
        return 1
    
    config = load_config(config_path)
    
    # Reduce dataset size for demo
    config['dataset']['num_images'] = 10
    print(f"Demo will use {config['dataset']['num_images']} images for quick testing")
    
    # Setup directories
    directories = setup_directories('.')
    
    # Setup logging
    logger = setup_logging(directories['logs'])
    logger.info("Starting YOLO Model Comparison Demo")
    
    try:
        print("\n1. Setting up test data...")
        # Setup test data (reduced dataset)
        image_data, coco_dataset = setup_test_data(config, directories, logger)
        print(f"   ✓ Loaded {len(image_data)} test images")
        
        print("\n2. Preparing models...")
        # For demo, we'll use mock models instead of downloading large weights
        print("   Using mock models for demonstration (no large downloads)")
        
        # Create mock weight paths (empty files)
        weights_dir = Path(directories['weights'])
        weights_dir.mkdir(exist_ok=True)
        
        weight_paths = {}
        for model_type, model_config in config['models'].items():
            for variant in model_config['variants']:
                model_key = f"{model_type}_{variant}"
                weight_path = weights_dir / f"{model_key}.pth"
                weight_path.touch()  # Create empty file
                weight_paths[model_key] = str(weight_path)
        
        print(f"   ✓ Prepared {len(weight_paths)} model configurations")
        
        print("\n3. Creating model instances...")
        # Create model instances
        models = create_model_instances(config, weight_paths, logger)
        print(f"   ✓ Created {len(models)} model instances")
        
        if not models:
            print("   ✗ No models were created. Check the setup.")
            return 1
        
        print("\n4. Running evaluation...")
        # Run evaluation
        results, comparator = run_model_evaluation(models, image_data, coco_dataset, config, logger)
        print(f"   ✓ Evaluated {len(results)} models")
        
        print("\n5. Generating reports...")
        # Save results
        save_results(results, comparator, directories, config, logger)
        print(f"   ✓ Results saved to {directories['results']}")
        
        print("\n6. Demo Results Summary:")
        print("-" * 40)
        
        # Display summary results
        for model_name, model_results in results.items():
            eval_metrics = model_results['evaluation']['overall_metrics']
            performance = model_results['performance']
            model_info = model_results['model_info']
            
            print(f"\n{model_name}:")
            print(f"  mAP@0.5: {eval_metrics.get('mAP@0.5', 0):.3f}")
            print(f"  mAP@0.5:0.95: {eval_metrics.get('mAP@0.5:0.95', 0):.3f}")
            print(f"  Avg FPS: {performance.get('avg_fps', 0):.1f}")
            print(f"  Parameters: {model_info.get('total_parameters', 0):,}")
            print(f"  Model Size: {model_info.get('model_size_mb', 0):.1f} MB")
            if model_info.get('is_mock', False):
                print("  (Mock model - for demonstration)")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        
        print(f"\nResults and reports are available in:")
        print(f"  - JSON reports: {directories['results']}")
        print(f"  - HTML report: {directories['results']}/comparison_report.html")
        print(f"  - CSV summary: {directories['results']}/comparison_summary.csv")
        print(f"  - Logs: {directories['logs']}/comparison.log")
        
        print(f"\nTo view the HTML report, open:")
        print(f"  file://{os.path.abspath(directories['results'])}/comparison_report.html")
        
        return 0
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\n✗ Demo failed: {str(e)}")
        return 1


def show_system_info():
    """Show system information and requirements."""
    print("\nSystem Information:")
    print("-" * 30)
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Check for key dependencies
    dependencies = ['torch', 'cv2', 'numpy', 'PIL', 'yaml', 'matplotlib']
    print(f"\nDependency Status:")
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ✗ {dep} (not installed)")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available (GPU: {torch.cuda.get_device_name(0)})")
        else:
            print(f"  ○ CUDA not available (CPU only)")
    except ImportError:
        print(f"  ○ PyTorch not installed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO Model Comparison Demo')
    parser.add_argument('--info', action='store_true', help='Show system information')
    
    args = parser.parse_args()
    
    if args.info:
        show_system_info()
        sys.exit(0)
    
    sys.exit(run_demo())