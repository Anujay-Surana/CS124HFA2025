# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CS124 Honors Project focused on real-time face detection, recognition, and tracking. The system uses OpenCV's YuNet face detector for detection, SFace for feature extraction, and a feature-enhanced centroid tracker to recognize and track multiple people across video frames, automatically organizing captured face crops by person ID.

## Environment Setup

### Python Virtual Environment

This project uses a Python virtual environment located at `.venv/` (or `venv/`).

Activate the virtual environment:
```bash
source .venv/bin/activate  # or source venv/bin/activate
```

Install dependencies:
```bash
pip install -r recog/requirements.txt
```

### Verify OpenCV Installation

Before running the main application, verify that OpenCV supports the YuNet interface:
```bash
python recog/test.py
```

This will confirm that `cv2.FaceDetectorYN_create` is available.

## Running the Application

### Run Face Detection and Tracking

From the project root:
```bash
python recog/main.py
```

This will:
- Open the default camera (source=0)
- Detect and track faces in real-time
- Save face crops to `recog/output/person_XXX/` directories
- Generate metadata.json files with timestamps and bounding box information
- Display a live preview window (press 'q' to quit)

To use a video file instead of the camera, modify `main.py` to call `run("path/to/video.mp4")`.

## Code Architecture

### Core Components

**main.py** - Entry point and main detection loop
- Initializes YuNet face detector from `recog/models/face_detection_yunet.onnx` (227KB)
- Initializes SFace face recognizer from `recog/models/face_recognition_sface_2021dec.onnx` (37MB)
- Manages video capture (defaults to CAP_AVFOUNDATION for macOS compatibility)
- Extracts 128-dimensional face feature vectors for each detected face
- Orchestrates detection, recognition, tracking, and output saving
- Saves face crops every 5 frames

**centroid_tracker.py** - Feature-enhanced multi-object tracking system
- Implements hybrid tracking using both position and facial features
- Assigns persistent IDs to detected faces across frames
- Uses cosine similarity on 128-dim feature vectors for face matching
- Combines position distance (30% weight) and feature similarity (70% weight)
- Exponential moving average (EMA) to update features over time (0.8 old + 0.2 new)
- Configurable parameters:
  - `max_disappeared=50`: frames before deregistration (increased for better persistence)
  - `max_distance=150`: maximum pixel distance for position matching
  - `min_feature_similarity=0.4`: minimum cosine similarity for feature matching

**utils.py** - Helper utilities
- `ensure_person_dir()`: Creates person-specific output directories (`output/person_XXX/`)
- `append_metadata()`: Manages JSON metadata files with detection info (timestamp, frame number, bbox)
- `now_ts()`: Generates timestamp strings

### Key Architecture Details

**Detection and Recognition Flow:**
1. YuNet detector processes each frame and returns face bounding boxes + landmarks
2. For each detected face:
   - SFace aligns the face using detected landmarks
   - Extracts a 128-dimensional feature vector representing the face
3. Both bounding boxes and feature vectors are passed to CentroidTracker.update()
4. Tracker matches new detections to existing tracked objects using hybrid matching
5. Each tracked face gets a persistent ID and updated bounding box
6. Face crops are saved to person-specific directories with metadata

**Feature-Based Tracking Algorithm:**
- **Position component**: Centroids computed as (x + w/2, y + h/2), distance matrix calculated
- **Feature component**: Cosine similarity computed between 128-dim feature vectors
- **Hybrid matching**: Combined cost = 0.3 × normalized_position + 0.7 × (1 - similarity)
- **Validation**: Matches must satisfy EITHER position < max_distance OR similarity > min_similarity
- **Feature update**: Exponential moving average smooths features over time
- **Deregistration**: Objects exceeding max_disappeared threshold are removed
- **New registration**: Unmatched detections become new tracked objects

**Output Structure:**
```
recog/output/
  person_000/
    000000.jpg
    000005.jpg
    metadata.json
  person_001/
    000010.jpg
    metadata.json
```

Each metadata.json contains an array of records with time, frame, and bbox fields.

## Model Files

Two ONNX models are required:

**Face Detection (YuNet):**
```
recog/models/face_detection_yunet.onnx (227KB)
```
- Detects faces and facial landmarks
- Already included in repository

**Face Recognition (SFace):**
```
recog/models/face_recognition_sface_2021dec.onnx (37MB)
```
- Extracts 128-dimensional feature vectors for face recognition
- Already included in repository
- Enables cross-session recognition (same person gets same ID even after leaving and returning)

## Dependencies

Key dependencies (see recog/requirements.txt):
- opencv-python >= 4.8.0
- opencv-contrib-python >= 4.8.0 (required for YuNet support)
- numpy >= 1.26.0
- imutils >= 0.5.4

Note: Both opencv-python and opencv-contrib-python are required because YuNet (FaceDetectorYN) is only available in the contrib package.
