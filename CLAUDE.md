# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CS124 Honors Project implementing real-time face and body detection with demographic analysis using OpenCV and deep neural networks. The system can detect faces and bodies in live camera feeds, images, or video files, and estimates age, gender, and ethnicity.

## Core Architecture

### Single-File Application
The entire application is contained in `DetectionMain.py` - a monolithic Python script that handles:
- Input source selection (camera, image file, or video file)
- Face detection using OpenCV's DNN module with Caffe models
- Body detection using HOG (Histogram of Oriented Gradients)
- Age classification (8 age ranges: 0-2, 4-6, 8-12, 15-20, 25-32, 38-43, 48-53, 60-100)
- Gender classification (Male/Female)
- Ethnicity estimation based on skin tone analysis in HSV and YCrCb color spaces

### Pre-trained Models
The `opencv_setup/` directory contains Caffe model files:
- `res10_300x300_ssd_iter_140000.caffemodel` + `deploy.prototxt`: Face detection (SSD-based)
- `gender_net.caffemodel` + `gender_deploy.prototxt`: Gender classification
- `age_net.caffemodel` + `age_deploy.prototxt`: Age classification

Note: Ethnicity estimation is implemented as a custom heuristic function (`estimate_ethnicity()`) rather than using a pre-trained model.

## Running the Application

```bash
python DetectionMain.py
```

The program will prompt for input source:
1. Live Camera (default camera index 0)
2. Image File (provide full path)
3. Video File (provide full path)

Press 'q' to quit when processing camera or video input. For images, press any key to close the display window.

## Key Configuration Parameters

- `confidence_threshold = 0.7`: Face detection confidence threshold (70%)
- `MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)`: Mean values for age/gender model preprocessing
- HOG parameters: `winStride=(8, 8)`, `padding=(4, 4)`, `scale=1.05`

## Important Implementation Details

### Face Detection Pipeline
1. Frame is resized to 300x300 for DNN processing
2. Blob is created with mean subtraction: `(104.0, 177.0, 123.0)`
3. Detection coordinates are scaled back to original frame dimensions
4. Confidence filtering applied (>70% threshold)

### Ethnicity Estimation Algorithm
Located in `estimate_ethnicity()` (DetectionMain.py:36-100):
- Analyzes HSV (Hue, Saturation, Value) and YCrCb color spaces
- Uses thresholds on average pixel values to classify into categories: African, South Asian, Hispanic/Latino, East Asian, Middle Eastern, Caucasian
- Returns confidence score (clamped between 30%-95%) based on how well values fit the category
- This is a heuristic approach with inherent limitations and biases

### Display Rendering
- Face bounding boxes: Green (0, 255, 0)
- Body bounding boxes: Blue (255, 0, 0)
- Text labels with colored backgrounds for visibility
- Large images automatically resized to fit within 1920x1080 display limits

## Development Considerations

When modifying this codebase:
- All model paths are relative to the repository root (use `opencv_setup/` prefix)
- The application expects BGR color format (OpenCV default)
- Face region extraction must check `face_img.size > 0` before processing to avoid errors
- Coordinate scaling is critical: detection happens at 300x300 but rendering happens at original resolution
- The ethnicity estimation function is sensitive to lighting conditions and should be used with caution
