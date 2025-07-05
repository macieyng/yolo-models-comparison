#!/usr/bin/env python3
"""
Demo script for YOLO Inference Comparison System
Shows how to compare detection results across different models.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from inference_main import main as inference_main

def create_sample_images():
    """Create sample images for testing."""
    print("Creating sample images...")
    
    # Create a temporary directory for sample images
    sample_dir = "./sample_images"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create some simple test images using PIL
    try:
        from PIL import Image, ImageDraw
        import random
        
        # Create 5 sample images with different content
        for i in range(5):
            # Create a blank image
            img = Image.new('RGB', (640, 480), color='lightblue')
            draw = ImageDraw.Draw(img)
            
            # Add some simple shapes to simulate objects
            for j in range(random.randint(1, 4)):
                # Random rectangle
                x1 = random.randint(50, 400)
                y1 = random.randint(50, 300)
                x2 = x1 + random.randint(50, 150)
                y2 = y1 + random.randint(50, 150)
                
                colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']
                color = random.choice(colors)
                
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                draw.text((x1, y1-20), f"Object {j+1}", fill=color)
            
            # Add image title
            draw.text((10, 10), f"Sample Image {i+1}", fill='black')
            
            # Save image
            img.save(os.path.join(sample_dir, f"sample_{i+1:02d}.jpg"))
        
        print(f"Created {5} sample images in {sample_dir}")
        return sample_dir
        
    except ImportError:
        print("PIL not available, creating placeholder images...")
        
        # Create placeholder text files instead
        for i in range(5):
            with open(os.path.join(sample_dir, f"sample_{i+1:02d}.txt"), 'w') as f:
                f.write(f"This is sample image {i+1}")
        
        print(f"Created {5} placeholder files in {sample_dir}")
        return sample_dir

def run_demo():
    """Run the inference comparison demo."""
    print("=" * 50)
    print("YOLO Inference Comparison Demo")
    print("=" * 50)
    
    # Create sample images
    sample_dir = create_sample_images()
    
    # Check if we have actual image files
    image_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    
    if not image_files:
        print("No image files found. Please install PIL (Pillow) to create sample images:")
        print("pip install Pillow")
        return sample_dir, "./demo_output"
    
    print(f"\nFound {len(image_files)} image files:")
    for img_file in image_files:
        print(f"  - {img_file}")
    
    # Run inference comparison
    print("\nRunning inference comparison...")
    
    # Create output directory
    output_dir = "./demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Example usage scenarios
    scenarios = [
        ("HTML Report", "demo_results.html"),
        ("JSON Results", "demo_results.json"),
        ("CSV Table", "demo_results.csv")
    ]
    
    for scenario_name, output_filename in scenarios:
        print(f"\n{'-' * 30}")
        print(f"Scenario: {scenario_name}")
        print(f"Output: {output_filename}")
        print(f"{'-' * 30}")
        
        output_path = os.path.join(output_dir, output_filename)
        
        # Simulate command line arguments
        original_argv = sys.argv
        sys.argv = [
            'inference_main.py',
            '--image-dir', sample_dir,
            '--output', output_path,
            '--limit', '5',
            '--verbose'
        ]
        
        try:
            result = inference_main()
            if result == 0:
                print(f"✓ Success! Results saved to: {output_path}")
                
                # Show file size
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    print(f"  File size: {file_size:,} bytes")
                    
                    # Show first few lines for text files
                    if output_filename.endswith('.csv'):
                        with open(output_path, 'r') as f:
                            lines = f.readlines()[:5]
                            print("  CSV preview:")
                            for line in lines:
                                print(f"    {line.strip()}")
                    
                    elif output_filename.endswith('.html'):
                        print("  HTML report generated - open in browser to view")
                    
                    elif output_filename.endswith('.json'):
                        print("  JSON results generated - contains detailed detection data")
                        
            else:
                print(f"✗ Failed with exit code: {result}")
                
        except Exception as e:
            print(f"✗ Error: {str(e)}")
        
        finally:
            # Restore original argv
            sys.argv = original_argv
    
    print(f"\n{'=' * 50}")
    print("Demo completed!")
    print(f"Sample images: {sample_dir}")
    print(f"Demo outputs: {output_dir}")
    print(f"{'=' * 50}")
    
    # Show usage examples
    print("\nUsage Examples:")
    print("===============")
    print("1. Run with your own images:")
    print(f"   python inference_main.py --image-dir /path/to/your/images --output results.html")
    print()
    print("2. Use the bash script:")
    print(f"   ./run_yolo_comparison.sh /path/to/your/images results.html")
    print()
    print("3. Limit number of images:")
    print(f"   python inference_main.py --image-dir {sample_dir} --output results.json --limit 10")
    
    return sample_dir, output_dir

if __name__ == "__main__":
    try:
        sample_dir, output_dir = run_demo()
        
        # Ask user if they want to clean up
        cleanup = input("\nClean up demo files? (y/n): ").lower().strip()
        if cleanup == 'y':
            if os.path.exists(sample_dir):
                shutil.rmtree(sample_dir)
                print(f"Cleaned up: {sample_dir}")
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
                print(f"Cleaned up: {output_dir}")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed: {str(e)}")
        sys.exit(1)