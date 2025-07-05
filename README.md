# YOLO Model Comparison System

A comprehensive system for comparing different YOLO (You Only Look Once) object detection models on a standardized dataset of 100 images. This system evaluates YOLOv6, PP-YOLOE, YOLOv11, YOLOX, and YOLOv3 models across multiple performance metrics.

## Features

### Supported Models
- **YOLOv6**: Meituan's industrial-grade object detection model with variants (n/s/m/l)
- **PP-YOLOE**: PaddlePaddle's efficient object detection model (s/m/l variants)
- **YOLOX**: Anchor-free YOLO model with multiple variants (nano/tiny/s/m/l/x)
- **YOLOv3**: Classic YOLO model with darknet weights support
- **YOLOv11**: Latest YOLO model with improved architecture (n/s/m/l variants)

### Evaluation Metrics
- **Accuracy Metrics**: mAP@0.5, mAP@0.5:0.95, precision, recall
- **Performance Metrics**: Inference time, FPS, memory usage
- **Model Complexity**: Parameters count, model size, FLOPs
- **Per-class Analysis**: Individual class performance breakdown

### Output Formats
- **JSON Reports**: Detailed results for each model
- **CSV Summary**: Tabular comparison data
- **HTML Report**: Interactive web-based comparison report
- **Visualization Charts**: Performance comparison graphs

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional but recommended)
- 8GB+ RAM
- 10GB+ disk space for models and data

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd yolo-model-comparison
   ```

2. **Run the setup script**
   ```bash
   python setup.py
   ```
   
   This will:
   - Install all required dependencies
   - Create necessary directories
   - Set up the project structure

3. **Verify installation**
   ```bash
   python main.py --help
   ```

## Usage

### Quick Start (Recommended)

**Single Command with Custom Images:**
```bash
# Run comparison with your own images
./run_yolo_comparison.sh /path/to/your/images results.html

# Examples:
./run_yolo_comparison.sh ~/Pictures/test_images output/report.html
./run_yolo_comparison.sh ./test_images results.json
./run_yolo_comparison.sh /data/images comparison.csv
```

The bash script will:
- Validate your image directory and check for supported formats
- Setup the environment and install dependencies automatically
- Run the complete comparison on all YOLO models
- Generate your requested output format (HTML, JSON, or CSV)
- Display progress and results summary

**Supported Input Formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
**Supported Output Formats:** `.html`, `.json`, `.csv`

### Basic Usage

Run the complete comparison system with default synthetic images:
```bash
python main.py
```

This will:
1. Download model weights (if not already present)
2. Generate/download test images
3. Run inference on all models
4. Evaluate performance metrics
5. Generate comprehensive reports

### Advanced Usage

#### With Custom Images
```bash
python main.py --image-dir /path/to/images --output results.html
```

#### Custom Configuration
```bash
python main.py --config custom_config.yaml
```

#### Verbose Output
```bash
python main.py --verbose
```

#### Custom Output Directory
```bash
python main.py --output-dir /path/to/results
```

### Configuration

The system is configured via `configs/model_config.yaml`. Key sections include:

#### Dataset Configuration
```yaml
dataset:
  num_images: 100
  image_size: 640
  class_names: [...]  # COCO class names
  num_classes: 80
```

#### Model Configuration
```yaml
models:
  yolov6:
    name: "YOLOv6"
    variants: ["yolov6n", "yolov6s", "yolov6m", "yolov6l"]
    weight_urls: {...}
    input_size: [640, 640]
```

#### Evaluation Configuration
```yaml
evaluation:
  confidence_threshold: 0.25
  iou_threshold: 0.45
  max_detections: 300
  metrics: ["precision", "recall", "mAP50", "mAP50-95", "fps"]
```

## Architecture

### Core Components

#### Base Model Class (`utils/base_model.py`)
- Abstract base class for all YOLO implementations
- Standardized interface for loading, inference, and evaluation
- Performance monitoring and statistics collection

#### Evaluation System (`utils/evaluation.py`)
- `DetectionEvaluator`: Computes object detection metrics
- `PerformanceComparator`: Compares models and generates reports
- Support for multiple IoU thresholds and per-class analysis

#### Data Utilities (`utils/data_utils.py`)
- `COCODataset`: Handles COCO format annotations
- `ImageDataLoader`: Loads and preprocesses images
- `TestDataGenerator`: Creates synthetic test data
- `ModelWeightDownloader`: Downloads model weights

#### Model Implementations (`models/`)
- Individual model implementations inheriting from `BaseYOLOModel`
- Mock model for testing when actual implementations aren't available
- Support for different model architectures and weight formats

### Bash Script Features

The `run_yolo_comparison.sh` script provides:

#### Automatic Setup
- Validates Python 3 and pip3 installation
- Installs required dependencies automatically
- Creates necessary directories
- Runs setup scripts if needed

#### Input Validation
- Checks if image directory exists and contains supported image files
- Validates output file format and creates output directories
- Provides clear error messages for common issues

#### Progress Monitoring
- Colored output for better readability
- Progress indicators during execution
- Execution time tracking
- Results summary with file sizes

#### Error Handling
- Graceful error handling with informative messages
- Automatic cleanup of temporary files
- Proper exit codes for scripting

#### Output Options
- **HTML Report**: Interactive web-based comparison
- **JSON Results**: Complete machine-readable data
- **CSV Summary**: Spreadsheet-compatible format
- Automatic format detection based on file extension

### Directory Structure
```
yolo-model-comparison/
├── configs/
│   └── model_config.yaml      # Configuration file
├── data/
│   ├── images/                # Test images
│   ├── annotations/           # COCO format annotations
│   ├── results/               # Evaluation results
│   └── visualizations/        # Generated charts
├── models/
│   ├── mock_model.py          # Mock model implementation
│   └── __init__.py
├── utils/
│   ├── base_model.py          # Base model class
│   ├── evaluation.py          # Evaluation utilities
│   ├── data_utils.py          # Data handling utilities
│   └── __init__.py
├── weights/                   # Downloaded model weights
├── logs/                      # System logs
├── main.py                    # Main execution script
├── setup.py                   # Setup script
├── run_yolo_comparison.sh     # One-click execution script
├── example_usage.sh           # Usage examples
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Results and Reports

### Output Files

After running the comparison, you'll find:

1. **Individual Model Results** (`data/results/`)
   - `{model_name}_results.json`: Detailed metrics for each model
   - Contains accuracy, performance, and complexity metrics

2. **Comparison Reports** (`data/results/`)
   - `comparison_report.json`: Complete comparison data
   - `comparison_summary.csv`: Tabular summary
   - `comparison_report.html`: Interactive HTML report

3. **Logs** (`logs/`)
   - `comparison.log`: Detailed execution logs
   - Error messages and debugging information

### Sample Results

The system provides comprehensive metrics:

```json
{
  "model_name": "YOLOv6-yolov6s",
  "overall_metrics": {
    "mAP@0.5": 0.456,
    "mAP@0.5:0.95": 0.312,
    "num_classes": 80,
    "num_images": 100
  },
  "performance": {
    "avg_fps": 45.2,
    "avg_inference_time": 0.022,
    "total_parameters": 18500000,
    "model_size_mb": 70.4
  }
}
```

## Technical Details

### Model Integration

To add a new YOLO model:

1. Create a new model class inheriting from `BaseYOLOModel`
2. Implement required methods:
   - `load_model()`: Load model weights
   - `preprocess_image()`: Image preprocessing
   - `forward()`: Model inference
   - `postprocess_predictions()`: Parse model outputs

3. Add model configuration to `configs/model_config.yaml`
4. Register the model in `main.py`

### Performance Considerations

- **Memory Usage**: Models are loaded sequentially to manage memory
- **Inference Speed**: Timing excludes data loading and preprocessing
- **Batch Processing**: Currently processes images individually for fair comparison
- **Device Support**: Automatic GPU detection and fallback to CPU

### Limitations

- Currently uses mock models for demonstration
- Synthetic test data generation (not real COCO validation set)
- Sequential processing (not parallelized)
- Limited to 100 images for comparison

## Contributing

### Development Setup

1. Clone the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Run tests: `python -m pytest tests/`

### Adding New Models

1. Implement the model class in `models/`
2. Add configuration in `configs/model_config.yaml`
3. Update model imports in `main.py`
4. Test with the mock system first

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Include comprehensive docstrings
- Add unit tests for new features

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or use CPU inference
   - Process models sequentially

2. **Model Weight Download Failures**
   - Check internet connection
   - Verify URLs in configuration
   - Try manual download

3. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path and virtual environment

### Getting Help

- Check the logs in `logs/comparison.log`
- Run with `--verbose` for detailed output
- Verify configuration file syntax
- Ensure all dependencies are installed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv6 team at Meituan
- PaddlePaddle team for PP-YOLOE
- Megvii team for YOLOX
- Joseph Redmon for original YOLO
- Ultralytics team for YOLOv11
- COCO dataset creators and maintainers

---

**Note**: This system currently uses mock models for demonstration purposes. For production use, integrate actual model implementations following the provided architecture.