import cv2
import numpy as np
import yaml
from typing import Dict, List  # Add this import
from tqdm import tqdm
from pathlib import Path
from scipy.optimize import linear_sum_assignment  # Add this import too
from .detection import PlayerDetector
from .reid import ReIDModel
from .visualization import Visualizer
from .utils import ensure_directory

class CrossCameraMapper:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.detector = PlayerDetector(self.config["model_path"])
        self.reid_model = ReIDModel()
        self.visualizer = Visualizer()
        self.output_dir = ensure_directory(self.config["output_dir"])
    
    def extract_features(self, video_path: str) -> Dict[int, List[Dict]]:
        """Extract features from all frames in a video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        features_data = {}
        
        for frame_idx in tqdm(range(frame_count), desc=f"Processing {Path(video_path).name}"):
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = self.detector.detect(frame, self.config["detection"]["conf_threshold"])
            frame_features = []
            
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                features = self.reid_model.extract_features(crop)
                frame_features.append({
                    "bbox": det["bbox"],
                    "features": features
                })
            
            features_data[frame_idx] = frame_features
        
        cap.release()
        return features_data
    
    def match_cameras(self, features1: Dict[int, List[Dict]], 
                     features2: Dict[int, List[Dict]]) -> Dict[int, int]:
        """Match players between two camera feeds"""
        # For simplicity, we'll match based on feature similarity in synchronized frames
        # In a real implementation, we'd use temporal alignment and spatial constraints
        mappings = {}
        similarity_threshold = self.config["matching"]["similarity_threshold"]
        
        # Find common frames (assuming same frame rate and start time)
        common_frames = set(features1.keys()) & set(features2.keys())
        
        for frame_idx in common_frames:
            frame_features1 = features1[frame_idx]
            frame_features2 = features2[frame_idx]
            
            # Create cost matrix
            cost_matrix = np.zeros((len(frame_features1), len(frame_features2)))
            
            for i, feat1 in enumerate(frame_features1):
                for j, feat2 in enumerate(frame_features2):
                    similarity = self.reid_model.compute_similarity(
                        feat1["features"], feat2["features"]
                    )
                    cost_matrix[i, j] = 1.0 - similarity  # Convert to cost
            
            # Solve assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < 1.0 - similarity_threshold:
                    # Create mapping (using indices as pseudo-IDs)
                    mappings[i] = j
        
        return mappings
    
    def process(self, video1_path: str, video2_path: str):
        # Extract features from both videos
        features1 = self.extract_features(video1_path)
        features2 = self.extract_features(video2_path)
        
        # Match players between cameras
        mappings = self.match_cameras(features1, features2)
        
        # Visualize results
        self.visualize_results(video1_path, video2_path, mappings, features1, features2)
    
    def visualize_results(self, video1_path: str, video2_path: str, 
                         mappings: Dict[int, int], 
                         features1: Dict[int, List[Dict]], 
                         features2: Dict[int, List[Dict]]):
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        frame_count = min(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), 
                         int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = min(cap1.get(cv2.CAP_PROP_FPS), cap2.get(cv2.CAP_PROP_FPS))
        
        # Prepare video writer
        output_path = Path(self.output_dir) / "cross_camera_mapping.mp4"
        width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        composite_width = width1 + width2
        composite_height = max(height1, height2)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, 
                            (composite_width, composite_height))
        
        for frame_idx in tqdm(range(frame_count), desc="Visualizing results"):
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2:
                break
            
            # Draw detections
            if frame_idx in features1:
                for det in features1[frame_idx]:
                    x1, y1, x2, y2 = det["bbox"]
                    cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if frame_idx in features2:
                for det in features2[frame_idx]:
                    x1, y1, x2, y2 = det["bbox"]
                    cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Create composite frame
            composite = np.zeros((composite_height, composite_width, 3), dtype=np.uint8)
            composite[:height1, :width1] = frame1
            composite[:height2, width1:width1+width2] = frame2
            
            # Draw mappings
            composite = self.visualizer.draw_cross_camera_mapping(frame1, frame2, mappings)
            out.write(composite)
        
        cap1.release()
        cap2.release()
        out.release()
        print(f"Cross-camera visualization saved to: {output_path}")
