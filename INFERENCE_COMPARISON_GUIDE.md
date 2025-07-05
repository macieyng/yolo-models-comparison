# YOLO Inference Comparison System

This system compares what different YOLO models detect on the same images, **without requiring ground truth annotations**. Instead of evaluating accuracy against known labels, it shows you what each model finds and creates visual comparisons.

## What This System Does

### 🔍 **Detection Comparison**
- Runs inference with each YOLO model on your images
- Shows what classes each model detects and how many
- Creates side-by-side visual comparisons

### 📊 **Results Tables**
- **Rows**: Your images
- **Columns**: YOLO models (YOLOv6, PP-YOLOE, YOLOv11, YOLOX, YOLOv3)
- **Values**: Detected classes and their counts (e.g., "person: 3, car: 2")

### 🎨 **Visual Mosaics**
- For each image, creates a mosaic showing detections from all models
- Different colors for different classes
- Bounding boxes with confidence scores

## Quick Start

### 🚀 **One-Command Execution**
```bash
# Basic usage - HTML report
./run_yolo_comparison.sh /path/to/your/images results.html

# JSON results for further processing
./run_yolo_comparison.sh /path/to/your/images results.json

# CSV table for spreadsheet analysis
./run_yolo_comparison.sh /path/to/your/images results.csv
```

### 📱 **Python Script Usage**
```bash
# Basic inference comparison
python inference_main.py --image-dir /path/to/images --output results.html

# Limit number of images
python inference_main.py --image-dir /path/to/images --output results.json --limit 50

# Verbose output
python inference_main.py --image-dir /path/to/images --output results.csv --verbose
```

### 🧪 **Try the Demo**
```bash
# Run interactive demo with sample images
python inference_demo.py
```

## Output Formats

### 📄 **HTML Report** (`.html`)
- **Interactive web report** with:
  - Summary statistics for each model
  - Detection table with all results
  - Visual mosaics for each image
  - Model performance comparison

### 📋 **CSV Table** (`.csv`)
- **Spreadsheet-compatible** format
- Easy to import into Excel, Google Sheets, etc.
- Rows = images, Columns = models
- Perfect for analysis and sharing

### 🔧 **JSON Results** (`.json`)
- **Complete machine-readable** data
- Detailed detection information
- Confidence scores and bounding boxes
- Ideal for further processing or integration

## Example Results

### Detection Table
```
Image          | YOLOv6-s           | YOLOv11-n          | YOLOX-s
---------------|--------------------|--------------------|------------------
image_001.jpg  | person: 2, car: 1  | person: 3, car: 1  | person: 2, dog: 1
image_002.jpg  | car: 3, truck: 1   | car: 2, truck: 1   | car: 4, bus: 1
image_003.jpg  | No detections      | person: 1          | person: 1, car: 1
```

### Visual Mosaics
Each image gets a mosaic showing:
- **Top-left**: YOLOv6 detections
- **Top-right**: YOLOv11 detections  
- **Bottom-left**: YOLOX detections
- **Bottom-right**: PP-YOLOE detections
- **Bottom-center**: YOLOv3 detections

## Key Features

### ✅ **No Ground Truth Required**
- Just point to your images directory
- No need to create annotations
- Works with any image collection

### ⚡ **Multiple Models**
- **YOLOv6**: Meituan's industrial-grade model
- **PP-YOLOE**: PaddlePaddle's efficient model
- **YOLOv11**: Latest YOLO with improved architecture
- **YOLOX**: Anchor-free YOLO variant
- **YOLOv3**: Classic YOLO model

### 🎯 **Flexible Output**
- Choose your preferred format (HTML/CSV/JSON)
- Automatic format detection from file extension
- Complete results always saved in `results/` directory

### 🔧 **Customizable**
- Set confidence thresholds
- Limit number of images processed
- Configure model variants
- Adjust visualization parameters

## System Architecture

### Core Components

1. **`inference_main.py`** - Main script for inference comparison
2. **`utils/inference_comparator.py`** - Comparison logic and visualization
3. **`run_yolo_comparison.sh`** - One-command bash script
4. **`inference_demo.py`** - Interactive demo with sample images

### Workflow

1. **Image Loading**: Load images from your directory
2. **Model Setup**: Initialize all YOLO models with weights
3. **Inference**: Run each model on each image
4. **Comparison**: Analyze detection differences
5. **Visualization**: Create mosaics and tables
6. **Export**: Save results in chosen format

## Advanced Usage

### Custom Configuration
```bash
# Use custom model configuration
python inference_main.py --config my_config.yaml --image-dir /path/to/images --output results.html
```

### Batch Processing
```bash
# Process multiple directories
for dir in /data/images/*/; do
    ./run_yolo_comparison.sh "$dir" "results_$(basename $dir).html"
done
```

### Integration with Other Tools
```python
# Load JSON results for further analysis
import json
with open('results.json', 'r') as f:
    data = json.load(f)

# Extract detection table
table = data['detection_table']
for row in table:
    print(f"Image: {row['Image']}")
    for model, detections in row.items():
        if model != 'Image':
            print(f"  {model}: {detections}")
```

## Comparison with Ground Truth Evaluation

| Feature | Inference Comparison | Ground Truth Evaluation |
|---------|---------------------|--------------------------|
| **Purpose** | Compare model outputs | Measure accuracy |
| **Requirements** | Just images | Images + annotations |
| **Metrics** | Detection counts, overlap | mAP, precision, recall |
| **Use Case** | Model comparison | Model validation |
| **Output** | Visual comparisons | Performance scores |

## Troubleshooting

### Common Issues

1. **No detections found**
   - Lower confidence threshold in config
   - Check image quality and size
   - Verify model weights are loaded

2. **Missing dependencies**
   - Install missing packages: `pip install -r requirements.txt`
   - Optional: `pip install pandas matplotlib opencv-python`

3. **Memory issues**
   - Use `--limit` to process fewer images
   - Process images in batches
   - Use CPU if GPU memory is limited

### Performance Tips

- **Use GPU**: Significantly faster inference
- **Batch processing**: Process similar-sized images together
- **Limit images**: Start with small batches for testing
- **Choose format**: HTML for viewing, CSV for analysis, JSON for processing

## Examples

### Real-World Scenarios

1. **Security Camera Analysis**
   ```bash
   ./run_yolo_comparison.sh /data/security_footage/ security_analysis.html
   ```

2. **Medical Imaging Comparison**
   ```bash
   python inference_main.py --image-dir /data/medical_scans/ --output medical_comparison.csv --limit 100
   ```

3. **Autonomous Vehicle Testing**
   ```bash
   ./run_yolo_comparison.sh /data/driving_scenarios/ av_model_comparison.json
   ```

4. **Wildlife Monitoring**
   ```bash
   python inference_main.py --image-dir /data/wildlife_photos/ --output wildlife_detections.html
   ```

---

## Next Steps

1. **Try the demo**: `python inference_demo.py`
2. **Run on your images**: `./run_yolo_comparison.sh /path/to/images results.html`
3. **Analyze results**: Open HTML report in browser or CSV in spreadsheet
4. **Compare models**: Look for patterns in detection differences
5. **Choose best model**: Based on what works best for your use case

This system helps you understand how different YOLO models perform on your specific images, making it easier to choose the right model for your application!