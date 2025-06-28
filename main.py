# main.py (fixed)
import argparse
import yaml
from pathlib import Path
from src.single_feed import SingleFeedTracker
from src.cross_camera import CrossCameraMapper
from src.utils import download_model, ensure_directory

def main():
    # Hardcoded video paths (relative to project root)
    VIDEO_PATHS = {
        "single": "data/videos/15sec_input_720p.mp4",  # Fixed path
        "cross": {
            "video1": "data/videos/broadcast.mp4",     # Fixed path
            "video2": "data/videos/tacticam.mp4"       # Fixed typo and path
        }
    }

    # Hardcoded config paths
    CONFIG_PATHS = {
        "single": "configs/tracking.yaml",
        "cross": "configs/reid.yaml"
    }

    mode = "single"  # Default mode (change to "cross" for cross-camera tracking)
    
    # Load config
    config_path = CONFIG_PATHS[mode]
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Ensure model exists
    model_path = Path(config["model_path"])
    if not model_path.exists():
        print(f"Model not found at {model_path}. Downloading...")
        download_model(config["model_url"], model_path)

    # Run pipeline
    if mode == "single":
        print("Running single feed tracking...")
        tracker = SingleFeedTracker(config_path)
        tracker.process_video(VIDEO_PATHS["single"])
    elif mode == "cross":
        print("Running cross-camera mapping...")
        mapper = CrossCameraMapper(config_path)
        mapper.process(
            VIDEO_PATHS["cross"]["video1"],
            VIDEO_PATHS["cross"]["video2"]
        )

if __name__ == "__main__":
    main()
