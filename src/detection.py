import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils import ops
import numpy as np
from typing import List, Dict, Tuple, Any

class PlayerDetector:
    def __init__(self, model_path: str, device: str = "cuda:0" if torch.cuda.is_available() else "cpu"):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.device = device
        self.class_names = self.model.names
        self.player_class_id = self._get_class_id("player")
        self.ball_class_id = self._get_class_id("ball")
    
    def _get_class_id(self, class_name: str) -> int:
        for id, name in self.class_names.items():
            if name.lower() == class_name.lower():
                return id
        raise ValueError(f"Class '{class_name}' not found in model classes")
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Dict[str, Any]]:
        results = self.model(frame, verbose=False, conf=conf_threshold)
        detections = []
        
        for result in results:
            if result.boxes is None:
                return detections
                
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, class_id in zip(boxes, confs, class_ids):
                if class_id == self.player_class_id:
                    detections.append({
                        "bbox": box.astype(int),
                        "confidence": float(conf),
                        "class_id": int(class_id),
                        "class_name": self.class_names[class_id]
                    })
        
        return detections
    
    def detect_video(self, video_path: str, conf_threshold: float = 0.5) -> Dict[int, List[Dict[str, Any]]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Processing video: {video_path}")
        print(f"Frames: {frame_count}, FPS: {fps:.2f}, Resolution: {width}x{height}")
        
        all_detections = {}
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = self.detect(frame, conf_threshold)
            all_detections[frame_idx] = detections
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed frame {frame_idx}/{frame_count}")
        
        cap.release()
        return all_detections
