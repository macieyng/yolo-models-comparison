#!/bin/bash

# Example Usage Script for YOLO Model Comparison
# This script demonstrates how to use the run_yolo_comparison.sh script

echo "YOLO Model Comparison - Example Usage"
echo "====================================="
echo

# Example 1: Basic usage with a sample image directory
echo "Example 1: Basic usage with HTML output"
echo "Usage: ./run_yolo_comparison.sh /path/to/images results.html"
echo

# Example 2: Using JSON output format
echo "Example 2: Using JSON output format"
echo "Usage: ./run_yolo_comparison.sh /path/to/images results.json"
echo

# Example 3: Using CSV output format
echo "Example 3: Using CSV output format"
echo "Usage: ./run_yolo_comparison.sh /path/to/images results.csv"
echo

# Example 4: With custom output directory
echo "Example 4: With custom output directory"
echo "Usage: ./run_yolo_comparison.sh /path/to/images output/comparison_report.html"
echo

# Create a sample directory structure for demonstration
echo "Creating sample directory structure..."
mkdir -p sample_images
mkdir -p output

# Note: You would put your actual images in the sample_images directory
echo "Note: Place your images in the 'sample_images' directory"
echo "Supported formats: .jpg, .jpeg, .png, .bmp, .tiff"
echo

# Example commands you can run:
echo "Example commands you can run:"
echo "1. ./run_yolo_comparison.sh sample_images output/report.html"
echo "2. ./run_yolo_comparison.sh sample_images output/results.json"
echo "3. ./run_yolo_comparison.sh sample_images output/summary.csv"
echo

echo "For help with the script, run: ./run_yolo_comparison.sh"
echo "This will show the usage information."