

# Player Tracking and Re-Identification System

## Overview

This repository contains code for a football player tracking and re-identification system. The system supports:

- **Single-feed player tracking** using YOLOv11 for detection and DeepSORT for tracking.
- **Cross-camera player re-identification** based on appearance feature embeddings and matching across two video feeds.

---

## Features

- Player detection with YOLOv11.
- Appearance feature extraction using a ResNet50-based ReID model.
- Tracking using motion and appearance similarity combined with Hungarian assignment.
- Visualization of tracking results with consistent player IDs.

---

## Repository Structure

```
player-tracking-system/
├── configs/
│   ├── tracking.yaml
│   └── reid.yaml
├── data/
│   ├── videos/
│   │   ├── 15sec_input_720p.mp4
│   │   ├── broadcast.mp4
│   │   └── tacticam.mp4
│   └── models/
│       └── best.pt
├── outputs/
│   ├──single_feed/
│   │   ├── tracked_15sec_input_720p.mp4
├── src/                        # Source code
│   ├── detection.py            # YOLOv11 detection pipeline
│   ├── reid.py                 # Appearance feature extraction
│   ├── tracker.py              # Tracking and data association logic
│   ├── visualization.py        # Visualization utilities
│   └── main.py                 # Main entry point for running tasks
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── report.md                   # Project report
```

---

## Setup Instructions

1. **Clone the repository**

```
git clone https://github.com/yourusername/player-tracking-system.git
cd player-tracking-system
```

2. **Create and activate a virtual environment (optional but recommended)**

```
python3 -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
```

3. **Install dependencies**

```
pip install -r requirements.txt
```

---

## Running the Code

### Single-Feed Tracking

```
python3 main.py --mode single_feed --input data/videos/15sec_input_720p.mp4
```

This will detect and track players in a single video feed and save output to `outputs/single_feed/`.

### Cross-Camera Re-Identification (Experimental)

```
python3 main.py --mode cross_camera --input1 data/videos/broadcast.mp4 --input2 data/videos/tacticam.mp4
```

This attempts to identify players consistently across two camera feeds.  
**Note:** This feature is partially implemented and may not produce reliable results.

---

## Notes

- The trained YOLOv11 model weights file (`yolov11_weights.pt`) **is not included** in this repository due to its large size.
- You will need to download the weights separately from the official YOLOv11 release or request it from the project maintainer.
- Alternatively, you can train your own weights following the YOLOv11 documentation.

---

## Dependencies

- Python 3.8+
- PyTorch
- torchvision
- OpenCV
- NumPy
- tqdm

See `requirements.txt` for the full list.



