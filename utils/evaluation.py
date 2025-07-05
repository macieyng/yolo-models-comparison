"""
Evaluation utilities for YOLO model comparison.
Includes metrics computation and performance analysis.
"""

import numpy as np
import json
import csv
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score


class DetectionEvaluator:
    """Evaluator for object detection metrics."""
    
    def __init__(self, class_names: List[str], iou_thresholds: List[float] = None):
        """
        Initialize the evaluator.
        
        Args:
            class_names: List of class names
            iou_thresholds: List of IoU thresholds for evaluation
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
    def compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Compute IoU between two bounding boxes.
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_detections(self, predictions: List[Dict[str, Any]], 
                        ground_truths: List[Dict[str, Any]], 
                        iou_threshold: float = 0.5) -> Tuple[List[bool], List[bool]]:
        """
        Match predictions to ground truths based on IoU.
        
        Args:
            predictions: List of prediction dictionaries
            ground_truths: List of ground truth dictionaries
            iou_threshold: IoU threshold for matching
            
        Returns:
            Tuple of (prediction_matches, gt_matches)
        """
        pred_matches = [False] * len(predictions)
        gt_matches = [False] * len(ground_truths)
        
        # Sort predictions by confidence (highest first)
        sorted_preds = sorted(enumerate(predictions), 
                            key=lambda x: x[1]['confidence'], reverse=True)
        
        for pred_idx, pred in sorted_preds:
            best_iou = 0.0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for gt_idx, gt in enumerate(ground_truths):
                if gt_matches[gt_idx]:  # Already matched
                    continue
                    
                if pred['class_id'] != gt['class_id']:  # Different class
                    continue
                
                iou = self.compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Match if IoU exceeds threshold
            if best_iou >= iou_threshold and best_gt_idx != -1:
                pred_matches[pred_idx] = True
                gt_matches[best_gt_idx] = True
        
        return pred_matches, gt_matches
    
    def compute_precision_recall(self, predictions: List[List[Dict[str, Any]]], 
                               ground_truths: List[List[Dict[str, Any]]],
                               class_id: int, iou_threshold: float = 0.5) -> Tuple[List[float], List[float]]:
        """
        Compute precision-recall curve for a specific class.
        
        Args:
            predictions: List of prediction lists (one per image)
            ground_truths: List of ground truth lists (one per image)
            class_id: Class ID to evaluate
            iou_threshold: IoU threshold for matching
            
        Returns:
            Tuple of (precision, recall) arrays
        """
        all_confidences = []
        all_matches = []
        total_gt = 0
        
        for pred_list, gt_list in zip(predictions, ground_truths):
            # Filter by class
            class_preds = [p for p in pred_list if p['class_id'] == class_id]
            class_gts = [g for g in gt_list if g['class_id'] == class_id]
            
            total_gt += len(class_gts)
            
            if len(class_preds) == 0:
                continue
            
            # Match predictions to ground truths
            pred_matches, _ = self.match_detections(class_preds, class_gts, iou_threshold)
            
            # Collect confidences and matches
            for pred, match in zip(class_preds, pred_matches):
                all_confidences.append(pred['confidence'])
                all_matches.append(match)
        
        if len(all_confidences) == 0:
            return [0.0], [0.0]
        
        # Sort by confidence
        sorted_indices = np.argsort(all_confidences)[::-1]
        sorted_matches = np.array(all_matches)[sorted_indices]
        
        # Compute precision and recall
        tp = np.cumsum(sorted_matches)
        fp = np.cumsum(1 - sorted_matches)
        
        precision = tp / (tp + fp)
        recall = tp / total_gt if total_gt > 0 else np.zeros_like(tp)
        
        return precision.tolist(), recall.tolist()
    
    def compute_ap(self, precision: List[float], recall: List[float]) -> float:
        """
        Compute Average Precision using the 11-point interpolation method.
        
        Args:
            precision: Precision values
            recall: Recall values
            
        Returns:
            Average Precision value
        """
        if len(precision) == 0 or len(recall) == 0:
            return 0.0
        
        # Add sentinel values
        precision = [0.0] + precision + [0.0]
        recall = [0.0] + recall + [1.0]
        
        # Compute maximum precision for each recall level
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        
        # Compute AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            # Find the maximum precision for recall >= t
            max_prec = 0.0
            for p, r in zip(precision, recall):
                if r >= t:
                    max_prec = max(max_prec, p)
            ap += max_prec
        
        return ap / 11.0
    
    def evaluate_model(self, predictions: List[List[Dict[str, Any]]], 
                      ground_truths: List[List[Dict[str, Any]]],
                      model_name: str = "") -> Dict[str, Any]:
        """
        Evaluate a model's performance across all classes and IoU thresholds.
        
        Args:
            predictions: List of prediction lists (one per image)
            ground_truths: List of ground truth lists (one per image)
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary containing evaluation metrics
        """
        results = {
            'model_name': model_name,
            'per_class_metrics': {},
            'overall_metrics': {}
        }
        
        # Compute per-class metrics
        class_aps = defaultdict(list)
        
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]
            class_metrics = {}
            
            for iou_threshold in self.iou_thresholds:
                precision, recall = self.compute_precision_recall(
                    predictions, ground_truths, class_id, iou_threshold
                )
                ap = self.compute_ap(precision, recall)
                class_aps[class_id].append(ap)
                class_metrics[f'AP@{iou_threshold:.2f}'] = ap
            
            # Compute mAP@0.5 and mAP@0.5:0.95
            class_metrics['AP@0.5'] = class_aps[class_id][0]
            class_metrics['AP@0.5:0.95'] = np.mean(class_aps[class_id])
            
            results['per_class_metrics'][class_name] = class_metrics
        
        # Compute overall metrics
        all_aps_50 = [class_aps[i][0] for i in range(self.num_classes)]
        all_aps_50_95 = [np.mean(class_aps[i]) for i in range(self.num_classes)]
        
        results['overall_metrics'] = {
            'mAP@0.5': np.mean(all_aps_50),
            'mAP@0.5:0.95': np.mean(all_aps_50_95),
            'num_classes': self.num_classes,
            'num_images': len(predictions)
        }
        
        return results


class PerformanceComparator:
    """Compare performance metrics across multiple models."""
    
    def __init__(self):
        self.model_results = {}
        self.model_info = {}
    
    def add_model_results(self, model_name: str, evaluation_results: Dict[str, Any], 
                         model_info: Dict[str, Any], performance_stats: Dict[str, Any]):
        """
        Add results for a model.
        
        Args:
            model_name: Name of the model
            evaluation_results: Results from DetectionEvaluator
            model_info: Model information (params, size, etc.)
            performance_stats: Performance statistics (inference time, FPS, etc.)
        """
        self.model_results[model_name] = {
            'evaluation': evaluation_results,
            'model_info': model_info,
            'performance': performance_stats
        }
    
    def generate_comparison_report(self, output_path: str = None) -> Dict[str, Any]:
        """
        Generate a comprehensive comparison report.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Comparison report dictionary
        """
        if not self.model_results:
            return {}
        
        report = {
            'summary': self._generate_summary(),
            'detailed_metrics': self._generate_detailed_metrics(),
            'performance_comparison': self._generate_performance_comparison(),
            'model_complexity': self._generate_model_complexity(),
            'recommendations': self._generate_recommendations()
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            'num_models': len(self.model_results),
            'models': list(self.model_results.keys()),
            'best_map_50': {'model': '', 'value': 0.0},
            'best_map_50_95': {'model': '', 'value': 0.0},
            'fastest_model': {'model': '', 'fps': 0.0},
            'smallest_model': {'model': '', 'params': float('inf')}
        }
        
        for model_name, results in self.model_results.items():
            eval_results = results['evaluation']
            perf_stats = results['performance']
            model_info = results['model_info']
            
            # Check best mAP
            map_50 = eval_results['overall_metrics']['mAP@0.5']
            if map_50 > summary['best_map_50']['value']:
                summary['best_map_50'] = {'model': model_name, 'value': map_50}
            
            map_50_95 = eval_results['overall_metrics']['mAP@0.5:0.95']
            if map_50_95 > summary['best_map_50_95']['value']:
                summary['best_map_50_95'] = {'model': model_name, 'value': map_50_95}
            
            # Check fastest model
            fps = perf_stats.get('avg_fps', 0)
            if fps > summary['fastest_model']['fps']:
                summary['fastest_model'] = {'model': model_name, 'fps': fps}
            
            # Check smallest model
            params = model_info.get('total_parameters', float('inf'))
            if params < summary['smallest_model']['params']:
                summary['smallest_model'] = {'model': model_name, 'params': params}
        
        return summary
    
    def _generate_detailed_metrics(self) -> Dict[str, Any]:
        """Generate detailed metric comparison."""
        detailed = {}
        
        for model_name, results in self.model_results.items():
            eval_results = results['evaluation']
            detailed[model_name] = {
                'overall_metrics': eval_results['overall_metrics'],
                'per_class_metrics': eval_results['per_class_metrics']
            }
        
        return detailed
    
    def _generate_performance_comparison(self) -> Dict[str, Any]:
        """Generate performance comparison."""
        comparison = {}
        
        for model_name, results in self.model_results.items():
            perf_stats = results['performance']
            comparison[model_name] = perf_stats
        
        return comparison
    
    def _generate_model_complexity(self) -> Dict[str, Any]:
        """Generate model complexity comparison."""
        complexity = {}
        
        for model_name, results in self.model_results.items():
            model_info = results['model_info']
            complexity[model_name] = {
                'total_parameters': model_info.get('total_parameters', 0),
                'model_size_mb': model_info.get('model_size_mb', 0),
                'input_size': model_info.get('input_size', []),
                'device': model_info.get('device', '')
            }
        
        return complexity
    
    def _generate_recommendations(self) -> Dict[str, str]:
        """Generate recommendations based on results."""
        recommendations = {}
        
        # Best accuracy
        best_acc = max(self.model_results.items(), 
                      key=lambda x: x[1]['evaluation']['overall_metrics']['mAP@0.5'])
        recommendations['best_accuracy'] = f"{best_acc[0]} - Highest mAP@0.5: {best_acc[1]['evaluation']['overall_metrics']['mAP@0.5']:.3f}"
        
        # Best speed
        best_speed = max(self.model_results.items(),
                        key=lambda x: x[1]['performance'].get('avg_fps', 0))
        recommendations['best_speed'] = f"{best_speed[0]} - Highest FPS: {best_speed[1]['performance'].get('avg_fps', 0):.1f}"
        
        # Best efficiency (accuracy/params)
        best_efficiency = max(self.model_results.items(),
                             key=lambda x: x[1]['evaluation']['overall_metrics']['mAP@0.5'] / 
                             max(x[1]['model_info'].get('total_parameters', 1), 1))
        recommendations['best_efficiency'] = f"{best_efficiency[0]} - Best accuracy per parameter"
        
        return recommendations
    
    def save_results_csv(self, output_path: str):
        """Save comparison results to CSV file."""
        rows = []
        
        for model_name, results in self.model_results.items():
            row = {
                'Model': model_name,
                'mAP@0.5': results['evaluation']['overall_metrics']['mAP@0.5'],
                'mAP@0.5:0.95': results['evaluation']['overall_metrics']['mAP@0.5:0.95'],
                'Avg_FPS': results['performance'].get('avg_fps', 0),
                'Avg_Inference_Time': results['performance'].get('avg_inference_time', 0),
                'Parameters': results['model_info'].get('total_parameters', 0),
                'Model_Size_MB': results['model_info'].get('model_size_mb', 0)
            }
            rows.append(row)
        
        with open(output_path, 'w', newline='') as csvfile:
            if rows:
                writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)