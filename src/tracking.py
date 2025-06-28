import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Any, Optional
import cv2

class Track:
    def __init__(self, track_id: int, initial_detection: Dict[str, Any], 
                 initial_features: np.ndarray, max_age: int = 30):
        self.track_id = track_id
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.max_age = max_age
        
        # Detection history
        self.detections = deque(maxlen=10)
        self.detections.append(initial_detection)
        
        # Feature history
        self.features = deque(maxlen=10)
        self.features.append(initial_features)
        
        # Motion model
        self.position = initial_detection["bbox"].copy()
        self.velocity = np.zeros(2)
        
    def update(self, detection: Dict[str, Any], features: np.ndarray):
        self.detections.append(detection)
        self.features.append(features)
        
        # Update motion model
        prev_position = self.position
        self.position = detection["bbox"].copy()
        self.velocity = (self.position[:2] - prev_position[:2]) * 0.5 + self.velocity * 0.5
        
        self.hits += 1
        self.time_since_update = 0
    
    def predict(self) -> np.ndarray:
        """Predict next position based on motion model"""
        predicted = self.position.copy()
        predicted = predicted.astype(float)
        predicted[:2] += self.velocity
        return predicted
    
    def mark_missed(self):
        self.time_since_update += 1
        self.age += 1
    
    def is_confirmed(self) -> bool:
        return self.hits >= 3
    
    def is_dead(self) -> bool:
        return self.time_since_update > self.max_age
    
    def get_current_features(self) -> np.ndarray:
        """Get averaged features over last 5 detections"""
        if len(self.features) == 0:
            return None
        return np.mean(list(self.features), axis=0)
    
    def get_current_position(self) -> np.ndarray:
        return self.position.copy()

class PlayerTracker:
    def __init__(self, reid_model, config: dict):
        self.reid_model = reid_model
        self.tracks = {}
        self.next_id = 0
        self.config = config
        
    def _compute_cost_matrix(self, detections: List[Dict[str, Any]], 
                           features: List[np.ndarray]) -> np.ndarray:
        n_tracks = len(self.tracks)
        n_detections = len(detections)
        cost_matrix = np.zeros((n_tracks, n_detections))
        
        # Get track positions and features
        track_positions = []
        track_features = []
        for track in self.tracks.values():
            track_positions.append(track.predict())
            track_features.append(track.get_current_features())
        
        # Compute IoU costs
        for i, track_pos in enumerate(track_positions):
            for j, det in enumerate(detections):
                iou = self._iou(track_pos, det["bbox"])
                cost_matrix[i, j] = 1.0 - iou  # Convert IoU to cost
        
        # Compute appearance costs
        if n_tracks > 0 and n_detections > 0:
            feature_costs = np.zeros((n_tracks, n_detections))
            for i, track_feat in enumerate(track_features):
                for j, det_feat in enumerate(features):
                    similarity = self.reid_model.compute_similarity(track_feat, det_feat)
                    feature_costs[i, j] = 1.0 - similarity
            
            # Combine costs
            cost_matrix = (self.config["motion_weight"] * cost_matrix + 
                          self.config["appearance_weight"] * feature_costs)
        
        return cost_matrix
    
    def _iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute Intersection over Union (IoU) between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Compute area of intersection
        inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        
        # Compute area of both boxes
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        
        # Compute IoU
        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou
    
    def update(self, frame: np.ndarray, detections: List[Dict[str, Any]]):
        # Extract features for all detections
        features = []
        valid_detections = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            features.append(self.reid_model.extract_features(crop))
            valid_detections.append(det)
        
        detections = valid_detections
        
        # Initialize cost matrix if no tracks exist
        if len(self.tracks) == 0:
            for det, feat in zip(detections, features):
                self.tracks[self.next_id] = Track(self.next_id, det, feat)
                self.next_id += 1
            return
        
        # Compute cost matrix
        cost_matrix = self._compute_cost_matrix(detections, features)
        
        # Solve assignment problem
        track_ids = list(self.tracks.keys())
        detection_indices = list(range(len(detections)))
        
        if cost_matrix.size > 0:
            track_indices, det_indices = linear_sum_assignment(cost_matrix)
            
            # Process matches
            for i, j in zip(track_indices, det_indices):
                if cost_matrix[i, j] < self.config["max_cost"]:
                    track_id = track_ids[i]
                    self.tracks[track_id].update(detections[j], features[j])
            
            # Find unmatched tracks
            unmatched_tracks = set(range(len(track_ids))) - set(track_indices)
            for i in unmatched_tracks:
                track_id = track_ids[i]
                self.tracks[track_id].mark_missed()
            
            # Find unmatched detections
            unmatched_detections = set(range(len(detections))) - set(det_indices)
            for j in unmatched_detections:
                self.tracks[self.next_id] = Track(self.next_id, detections[j], features[j])
                self.next_id += 1
        else:
            # No tracks to match, create new tracks for all detections
            for det, feat in zip(detections, features):
                self.tracks[self.next_id] = Track(self.next_id, det, feat)
                self.next_id += 1
        
        # Remove dead tracks
        dead_tracks = [track_id for track_id, track in self.tracks.items() if track.is_dead()]
        for track_id in dead_tracks:
            del self.tracks[track_id]
    
    def get_tracks(self) -> Dict[int, Track]:
        return self.tracks
