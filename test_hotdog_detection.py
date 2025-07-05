#!/usr/bin/env python3
"""
Test script to verify hot dog detection and coordinate system fixes.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.mock_model import MockYOLOModel
from utils.inference_comparator import InferenceComparator
import numpy as np

def test_mock_model_hotdog_detection():
    """Test that mock model generates hot dog detections with correct coordinates."""
    print("Testing Mock Model Hot Dog Detection")
    print("=" * 40)
    
    # Create a mock model
    model = MockYOLOModel(
        model_name="Test Model",
        variant="test",
        weights_path="dummy.pt",
        input_size=(640, 640),
        conf_threshold=0.5
    )
    
    # Load model
    model.load_model()
    
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Run inference multiple times to check hot dog detection rate
    hot_dog_count = 0
    total_detections = 0
    confidence_scores = []
    coordinate_formats = []
    
    for i in range(10):
        try:
            predictions, inference_time = model.predict(dummy_image)
            
            for detection in predictions:
                total_detections += 1
                
                # Check for hot dogs
                if detection['class_name'] == 'hot dog':
                    hot_dog_count += 1
                    confidence_scores.append(detection['confidence'])
                
                # Check coordinate format
                bbox = detection['bbox']
                x, y, w, h = bbox
                if all(0 <= coord <= 1 for coord in bbox):
                    coordinate_formats.append('percentage')
                else:
                    coordinate_formats.append('absolute')
                
                print(f"Detection {total_detections}: {detection['class_name']} "
                      f"(conf: {detection['confidence']:.2f}, "
                      f"bbox: [{x:.3f}, {y:.3f}, {w:.3f}, {h:.3f}])")
        
        except Exception as e:
            print(f"Error in iteration {i}: {e}")
    
    # Print results
    print(f"\nResults after 10 inference runs:")
    print(f"Total detections: {total_detections}")
    print(f"Hot dog detections: {hot_dog_count}")
    print(f"Hot dog detection rate: {hot_dog_count/total_detections*100:.1f}%" if total_detections > 0 else "No detections")
    print(f"Average hot dog confidence: {np.mean(confidence_scores):.2f}" if confidence_scores else "No hot dogs detected")
    print(f"Min confidence threshold: {model.conf_threshold}")
    
    # Check coordinate format
    percentage_coords = coordinate_formats.count('percentage')
    print(f"Percentage coordinates: {percentage_coords}/{total_detections} ({percentage_coords/total_detections*100:.1f}%)" if total_detections > 0 else "No coordinates to check")
    
    return hot_dog_count > 0, percentage_coords == total_detections

def test_inference_comparator():
    """Test the inference comparator with hot dog detections."""
    print("\nTesting Inference Comparator")
    print("=" * 40)
    
    # COCO class names
    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                  'hair drier', 'toothbrush']
    
    # Create comparator with 50% confidence threshold
    comparator = InferenceComparator(class_names, confidence_threshold=0.5)
    
    # Test detection data with hot dogs
    test_detections = [
        {
            'bbox': [0.2, 0.3, 0.3, 0.1],  # Percentage coordinates
            'confidence': 0.85,
            'class_id': 52,
            'class_name': 'hot dog'
        },
        {
            'bbox': [0.1, 0.1, 0.2, 0.15],
            'confidence': 0.45,  # Below threshold - should be filtered
            'class_id': 52,
            'class_name': 'hot dog'
        },
        {
            'bbox': [0.6, 0.4, 0.25, 0.2],
            'confidence': 0.75,
            'class_id': 52,
            'class_name': 'hot dog'
        }
    ]
    
    # Add results to comparator
    comparator.add_model_results("Test Model", "test_image.jpg", test_detections)
    
    # Get detection counts
    counts = comparator.get_detection_counts("Test Model", "test_image.jpg")
    print(f"Detection counts: {counts}")
    
    # Check that low confidence detection was filtered out
    expected_hot_dogs = 2  # Only 2 should pass the 50% threshold
    actual_hot_dogs = counts.get('hot dog', 0)
    
    print(f"Expected hot dogs after filtering: {expected_hot_dogs}")
    print(f"Actual hot dogs after filtering: {actual_hot_dogs}")
    print(f"Confidence threshold: {comparator.confidence_threshold}")
    
    return actual_hot_dogs == expected_hot_dogs

def main():
    """Run all tests."""
    print("Hot Dog Detection and Coordinate System Tests")
    print("=" * 50)
    
    try:
        # Test 1: Mock model hot dog detection
        has_hotdogs, correct_coordinates = test_mock_model_hotdog_detection()
        
        # Test 2: Inference comparator
        correct_filtering = test_inference_comparator()
        
        # Summary
        print("\nTest Summary")
        print("=" * 20)
        print(f"✓ Hot dog detection working: {has_hotdogs}")
        print(f"✓ Percentage coordinates: {correct_coordinates}")
        print(f"✓ Confidence filtering (50%): {correct_filtering}")
        
        if all([has_hotdogs, correct_coordinates, correct_filtering]):
            print("\n🎉 All tests passed! Hot dog detection is working correctly.")
            return True
        else:
            print("\n❌ Some tests failed. Check the output above for details.")
            return False
    
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)