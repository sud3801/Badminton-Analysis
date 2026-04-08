# Automated Badminton Match Analysis using Deep Learning

## Project Overview

This project implements an automated badminton match analysis system using deep learning techniques. It processes badminton match videos to detect and track players and the shuttlecock, segment rallies, compute player metrics, and generate heatmaps. The system uses YOLOv8 for object detection and Kalman filtering for tracking, providing comprehensive analytics for badminton matches.

This is a major project developed for evaluation and demonstration purposes, showcasing computer vision and deep learning applications in sports analysis.

## Features

- **Object Detection**: Detect players and shuttlecock using YOLOv8
- **Shuttle Tracking**: Track shuttlecock trajectory using Kalman Filter
- **Player Tracking**: Multi-object tracking for players
- **Rally Segmentation**: Segment rallies and determine points
- **Player Metrics**: Compute distance traveled, speed, and other statistics
- **Heatmaps**: Generate player movement heatmaps
- **Homography**: Map 2D image coordinates to real-world court coordinates
- **Video Processing**: Handle single-camera (monocular) video input

## Project Structure

```
badminton-analysis/
│
├── data/
│   ├── datasets.yaml        # Dataset configuration
│   ├── annotations/         # Annotation data
│   ├── frames/              # Extracted video frames
│   ├── raw_videos/          # Input badminton match videos
│   └── shuttlecock/         # Shuttlecock detection dataset
│       ├── data.yaml        # Dataset YAML for YOLO training
│       ├── README.dataset.txt
│       ├── README.roboflow.txt
│       ├── test/
│       │   ├── images/
│       │   └── labels/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       └── valid/
│           ├── images/
│           └── labels/
│
├── models/
│   ├── weights/
│   │   └── shuttle_best.pt  # Best trained shuttlecock model
│   └── yolo/                # YOLO model files
│
├── outputs/
│   ├── heatmaps/            # Generated heatmaps
│   ├── logs/                # Log files
│   └── videos/              # Processed output videos
│
├── runs/
│   └── detect/              # YOLO detection runs
│       └── models/
│           └── yolo/
│               └── shuttle_detector/
│
├── src/
│   ├── main.py              # Main entry point
│   ├── yolov8n.pt           # YOLOv8 nano model
│   ├── analytics/
│   │   ├── heatmap.py       # Heatmap generation
│   │   ├── metrics.py       # Player metrics calculation
│   │   └── rally_detection.py # Rally segmentation
│   ├── configs/             # Configuration files
│   ├── detection/
│   │   └── yolo_detector.py # YOLO-based detection module
│   ├── tracking/
│   │   ├── player_tracker.py # Player tracking
│   │   └── shuttle_tracker.py # Shuttle tracking
│   └── utils/
│       ├── drawing.py       # Drawing and visualization
│       ├── homography.py    # Homography transformation
│       ├── roi_filter.py    # Region of interest filtering
│       ├── roi_selector.py  # Region of interest selection
│       └── video_utils.py   # Video processing utilities
│
├── folder-structure.txt     # Folder structure documentation
├── plan.txt                 # Project plan
├── requirements.txt         # Python dependencies
├── yolov8n.pt               # YOLOv8 nano model (duplicate)
└── README.md               # This file
```

## Installation

1. Clone or download the project repository.

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - On Windows: `.venv\Scripts\activate`
   - On macOS/Linux: `source .venv/bin/activate`

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

- ultralytics (YOLOv8)
- opencv-python
- numpy
- matplotlib
- scipy
- filterpy
- torch
- torchvision

## Usage

1. Place your badminton match video in the `data/raw_videos/` directory.

2. Run the main analysis script:
   ```bash
   python src/main.py
   ```

3. The processed outputs will be saved in the `outputs/` directory, including:
   - Analyzed videos with tracking overlays
   - Player metrics and statistics
   - Heatmaps of player movement
   - Rally segmentation data

## Dataset

The project uses a custom dataset for shuttlecock detection, sourced from Roboflow. The dataset is located in `data/shuttlecock/` with train, valid, and test splits.

## Training

To train the YOLOv8 model:
```bash
yolo train data=data/shuttlecock/data.yaml model=yolov8n.pt epochs=100
```

The trained model weights will be saved in `models/weights/shuttle_best.pt`.

## Results

The system successfully processes badminton videos to provide:
- Real-time player and shuttle tracking
- Accurate rally detection and segmentation
- Comprehensive player statistics including distance covered and speed
- Visual heatmaps showing player movement patterns
- Homography-corrected court coordinates for precise analysis

## Technologies Used

- **YOLOv8**: Object detection
- **Kalman Filter**: Trajectory tracking
- **OpenCV**: Computer vision operations
- **PyTorch**: Deep learning framework
- **NumPy/SciPy**: Numerical computations
- **Matplotlib**: Visualization

## Future Improvements

- Multi-camera support for 3D reconstruction
- Advanced In/Out detection using court lines
- Real-time processing optimization
- Integration with live streaming
- Player identification and team analysis

## License

This project is developed for educational and demonstration purposes.

## Acknowledgments

- Dataset sourced from Roboflow
- YOLOv8 by Ultralytics
- OpenCV community
- PyTorch framework