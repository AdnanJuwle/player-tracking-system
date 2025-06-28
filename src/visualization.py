import cv2
import numpy as np
from typing import Dict, List, Any, Optional

class Visualizer:
    def __init__(self, class_colors: Optional[Dict[str, tuple]] = None):
        self.class_colors = class_colors or {
            "player": (0, 255, 0),
            "ball": (255, 0, 0)
        }
        self.track_colors = {}
        
    def _get_track_color(self, track_id: int) -> tuple:
        if track_id not in self.track_colors:
            # Generate a random color for the track
            color = tuple(np.random.randint(0, 255, 3).tolist())
            self.track_colors[track_id] = color
        return self.track_colors[track_id]
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detections on a frame"""
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            class_name = det["class_name"]
            conf = det["confidence"]
            
            color = self.class_colors.get(class_name, (0, 0, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def draw_tracks(self, frame: np.ndarray, tracks: Dict[int, Any]) -> np.ndarray:
        """Draw tracks on a frame"""
        for track_id, track in tracks.items():
            if not track.is_confirmed():
                continue
                
            x1, y1, x2, y2 = track.get_current_position()
            color = self._get_track_color(track_id)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID
            label = f"ID: {track_id}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw trail
            positions = [d["bbox"] for d in track.detections]
            centers = [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for x1, y1, x2, y2 in positions]
            for i in range(1, len(centers)):
                cv2.line(frame, centers[i-1], centers[i], color, 2)
        
        return frame
    
    def draw_cross_camera_mapping(self, frame1: np.ndarray, frame2: np.ndarray, 
                                 mappings: Dict[int, int]) -> np.ndarray:
        """Visualize cross-camera mappings"""
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        # Create composite image
        composite = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        composite[:h1, :w1] = frame1
        composite[:h2, w1:w1+w2] = frame2
        
        # Draw mappings
        for track_id1, track_id2 in mappings.items():
            # Get positions (simplified for visualization)
            pos1 = (w1 // 2, h1 // 2)  # Should be actual positions
            pos2 = (w1 + w2 // 2, h2 // 2)
            
            color = self._get_track_color(track_id1)
            
            # Draw line between cameras
            cv2.line(composite, pos1, pos2, color, 2)
            
            # Draw IDs
            cv2.putText(composite, f"ID: {track_id1}", (pos1[0]-30, pos1[1]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(composite, f"ID: {track_id2}", (pos2[0]-30, pos2[1]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return composite
