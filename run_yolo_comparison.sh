#!/bin/bash

# YOLO Model Comparison Script
# Usage: ./run_yolo_comparison.sh <image_directory> <output_file>

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 <image_directory> <output_file>"
    echo ""
    echo "Arguments:"
    echo "  image_directory  Directory containing images for inference"
    echo "  output_file      Path for the output report (supports .json, .csv, .html)"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/images results.html"
    echo "  $0 ./test_images comparison_results.json"
    echo "  $0 ~/Pictures/dataset output/report.csv"
    echo ""
    echo "Supported image formats: .jpg, .jpeg, .png, .bmp, .tiff"
}

# Function to validate image directory
validate_image_directory() {
    local dir="$1"
    
    if [[ ! -d "$dir" ]]; then
        print_error "Image directory '$dir' does not exist"
        return 1
    fi
    
    # Check if directory contains any supported image files
    local image_count=$(find "$dir" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tiff" \) | wc -l)
    
    if [[ $image_count -eq 0 ]]; then
        print_error "No supported image files found in '$dir'"
        print_info "Supported formats: .jpg, .jpeg, .png, .bmp, .tiff"
        return 1
    fi
    
    print_info "Found $image_count image(s) in '$dir'"
    return 0
}

# Function to validate output file
validate_output_file() {
    local output_file="$1"
    local output_dir=$(dirname "$output_file")
    
    # Create output directory if it doesn't exist
    if [[ ! -d "$output_dir" ]]; then
        print_info "Creating output directory: $output_dir"
        mkdir -p "$output_dir"
    fi
    
    # Check if output directory is writable
    if [[ ! -w "$output_dir" ]]; then
        print_error "Output directory '$output_dir' is not writable"
        return 1
    fi
    
    # Validate output file extension
    local extension="${output_file##*.}"
    case "$extension" in
        json|csv|html)
            print_info "Output format: $extension"
            ;;
        *)
            print_warning "Unknown output format '$extension'. Supported: json, csv, html"
            print_info "Will attempt to use the specified format anyway"
            ;;
    esac
    
    return 0
}

# Function to check system requirements
check_requirements() {
    print_info "Checking system requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        return 1
    fi
    
    local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    print_info "Python version: $python_version"
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 is required but not installed"
        return 1
    fi
    
    return 0
}

# Function to setup environment
setup_environment() {
    print_info "Setting up environment..."
    
    # Check if requirements.txt exists
    if [[ ! -f "requirements.txt" ]]; then
        print_error "requirements.txt not found. Please ensure you're in the correct directory."
        return 1
    fi
    
    # Install requirements if not already installed
    print_info "Installing Python dependencies..."
    if ! pip3 install -r requirements.txt --quiet; then
        print_error "Failed to install Python dependencies"
        return 1
    fi
    
    # Run setup script if it exists
    if [[ -f "setup.py" ]]; then
        print_info "Running setup script..."
        python3 setup.py
    fi
    
    return 0
}

# Function to run the comparison
run_comparison() {
    local image_dir="$1"
    local output_file="$2"
    
    print_info "Starting YOLO model comparison..."
    print_info "Image directory: $image_dir"
    print_info "Output file: $output_file"
    
    # Create a temporary config file with the custom image directory
    local temp_config="temp_config.yaml"
    
    # Check if inference_main.py exists
    if [[ ! -f "inference_main.py" ]]; then
        print_error "inference_main.py not found. Please ensure you're in the correct directory."
        return 1
    fi
    
    # Run the comparison
    print_info "Running inference comparison (this may take several minutes)..."
    
    # Pass arguments to Python script
    if python3 inference_main.py --image-dir "$image_dir" --output "$output_file" --verbose; then
        print_success "Inference comparison completed successfully!"
        return 0
    else
        print_error "Inference comparison failed"
        return 1
    fi
}

# Function to show results summary
show_results() {
    local output_file="$1"
    
    if [[ -f "$output_file" ]]; then
        print_success "Results saved to: $output_file"
        
        # Show file size
        local file_size=$(du -h "$output_file" | cut -f1)
        print_info "File size: $file_size"
        
        # If HTML file, provide viewing instructions
        if [[ "$output_file" == *.html ]]; then
            print_info "To view the HTML report, open it in a web browser:"
            print_info "  firefox '$output_file' &"
            print_info "  google-chrome '$output_file' &"
            print_info "  or double-click the file in your file manager"
        fi
        
        # If JSON file, show a preview
        if [[ "$output_file" == *.json ]] && command -v jq &> /dev/null; then
            print_info "Preview of results:"
            echo "----------------------------------------"
            jq '.summary // .results[0] // .' "$output_file" 2>/dev/null | head -20
            echo "----------------------------------------"
        fi
    else
        print_error "Output file was not created"
        return 1
    fi
}

# Function to cleanup temporary files
cleanup() {
    print_info "Cleaning up temporary files..."
    
    # Remove any temporary config files
    if [[ -f "temp_config.yaml" ]]; then
        rm -f "temp_config.yaml"
    fi
    
    # Remove any temporary directories
    if [[ -d "temp_results" ]]; then
        rm -rf "temp_results"
    fi
}

# Main execution
main() {
    print_info "YOLO Model Comparison Script"
    print_info "============================"
    
    # Check arguments
    if [[ $# -ne 2 ]]; then
        print_error "Invalid number of arguments"
        show_usage
        exit 1
    fi
    
    local image_dir="$1"
    local output_file="$2"
    
    # Convert to absolute paths
    image_dir=$(realpath "$image_dir")
    output_file=$(realpath "$output_file")
    
    print_info "Input directory: $image_dir"
    print_info "Output file: $output_file"
    
    # Validate inputs
    if ! validate_image_directory "$image_dir"; then
        exit 1
    fi
    
    if ! validate_output_file "$output_file"; then
        exit 1
    fi
    
    # Check system requirements
    if ! check_requirements; then
        exit 1
    fi
    
    # Setup environment
    if ! setup_environment; then
        exit 1
    fi
    
    # Record start time
    local start_time=$(date +%s)
    
    # Run the comparison
    if run_comparison "$image_dir" "$output_file"; then
        # Calculate elapsed time
        local end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        local minutes=$((elapsed / 60))
        local seconds=$((elapsed % 60))
        
        print_success "Comparison completed in ${minutes}m ${seconds}s"
        
        # Show results
        show_results "$output_file"
        
        # Cleanup
        cleanup
        
        print_success "All done! Check your results at: $output_file"
        exit 0
    else
        print_error "Comparison failed"
        cleanup
        exit 1
    fi
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

# Run main function
main "$@"