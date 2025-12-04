# Model Files

This directory contains the pre-trained ONNX models for face detection and recognition.

## Required Models

### 1. face_detection_yunet.onnx (227KB) ✅
- **Purpose**: Face detection
- **Included in**: Git repository
- **Size**: 227KB (small enough for git)
- **Download**: Already included

### 2. face_recognition_sface_2021dec.onnx (37MB) ⬇️
- **Purpose**: Face recognition (feature extraction)
- **Included in**: NOT in git repository (too large)
- **Size**: 37MB
- **Download**: Automatically downloaded by `./setup.sh` or `./download_models.sh`

## Automatic Download

The large model file is automatically downloaded when you run:

```bash
./setup.sh
```

## Manual Download

If automatic download fails, you can download manually:

### Option 1: Use download script
```bash
./download_models.sh
```

### Option 2: Download directly
```bash
cd recog/models
curl -L https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx -o face_recognition_sface_2021dec.onnx
```

### Option 3: Browser download
Visit: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface

Download `face_recognition_sface_2021dec.onnx` and place it in this directory.

## Verify Models

Check if both models are present:

```bash
ls -lh recog/models/
```

Expected output:
```
face_detection_yunet.onnx (227K)
face_recognition_sface_2021dec.onnx (37M)
```

Or run:
```bash
./run.sh verify
```

## Why is SFace not in Git?

GitHub has a 100MB repository limit and recommends keeping files under 50MB. The 37MB SFace model is close to this limit, and we want to keep the repository lightweight. The model is automatically downloaded during setup.

## Model Sources

Both models are from OpenCV Zoo:
- https://github.com/opencv/opencv_zoo

Licensed under Apache 2.0.
