import json
import numpy as np
from collections import defaultdict

class TrackingMetrics:
    """Calculate tracking performance metrics"""
    
    def __init__(self):
        self.ground_truth = {}
        self.predictions = {}
        
    def add_ground_truth(self, frame_id, player_id, bbox):
        """Add ground truth annotation"""
        if frame_id not in self.ground_truth:
            self.ground_truth[frame_id] = {}
        self.ground_truth[frame_id][player_id] = bbox
    
    def add_prediction(self, frame_id, player_id, bbox):
        """Add prediction"""
        if frame_id not in self.predictions:
            self.predictions[frame_id] = {}
        self.predictions[frame_id][player_id] = bbox
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

def save_tracking_results(results, filename):
    """Save tracking results to JSON file"""
    serializable_results = {}
    for frame_id, players in results.items():
        serializable_results[str(frame_id)] = {}
        for player_id, data in players.items():
            serializable_results[str(frame_id)][str(player_id)] = {
                'bbox': data['bbox'].tolist() if isinstance(data['bbox'], np.ndarray) else data['bbox'],
                'confidence': float(data.get('confidence', 1.0)),
                'features': data.get('features', []).tolist() if isinstance(data.get('features', []), np.ndarray) else data.get('features', [])
            }
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
