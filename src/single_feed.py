import cv2
import numpy as np
import yaml
from tqdm import tqdm
from pathlib import Path
from .detection import PlayerDetector
from .reid import ReIDModel
from .tracking import PlayerTracker
from .visualization import Visualizer
from .utils import ensure_directory

class SingleFeedTracker:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.detector = PlayerDetector(self.config["model_path"])
        self.reid_model = ReIDModel()
        self.tracker = PlayerTracker(self.reid_model, self.config["tracking"])
        self.visualizer = Visualizer()
        self.output_dir = ensure_directory(self.config["output_dir"])
    
    def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Prepare video writer
        output_path = Path(self.output_dir) / f"tracked_{Path(video_path).name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        print(f"Processing video: {video_path}")
        print(f"Frames: {frame_count}, FPS: {fps:.2f}, Resolution: {width}x{height}")
        
        for frame_idx in tqdm(range(frame_count)):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect players
            detections = self.detector.detect(frame, self.config["detection"]["conf_threshold"])
            
            # Update tracker
            self.tracker.update(frame, detections)
            tracks = self.tracker.get_tracks()
            
            # Visualize results
            vis_frame = self.visualizer.draw_tracks(frame, tracks)
            out.write(vis_frame)
        
        cap.release()
        out.release()
        print(f"Tracking results saved to: {output_path}")
