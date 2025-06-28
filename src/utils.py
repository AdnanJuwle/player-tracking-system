import os
import cv2
import gdown
import numpy as np
from pathlib import Path
from typing import Union

def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, create if not"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def download_model(url: str, output_path: Union[str, Path]):
    """Download model from Google Drive"""
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    
    # Google Drive file ID extraction
    file_id = url.split("/")[-2]
    download_url = f"https://drive.google.com/uc?id={file_id}"
    
    gdown.download(download_url, str(output_path), quiet=False)
    print(f"Model downloaded to: {output_path}")

def video_to_frames(video_path: str, output_dir: str, frame_interval: int = 1):
    """Extract frames from a video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    output_dir = ensure_directory(output_dir)
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_path = output_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}") 
