#!/bin/bash
# Download large model files that are not in git repository

set -e

MODELS_DIR="recog/models"
SFACE_MODEL="face_recognition_sface_2021dec.onnx"
SFACE_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

echo "========================================"
echo "Downloading Face Recognition Models"
echo "========================================"
echo ""

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

# Check if SFace model already exists
if [ -f "$MODELS_DIR/$SFACE_MODEL" ]; then
    echo "✓ $SFACE_MODEL already exists (37MB)"
    echo "  Skipping download"
else
    echo "Downloading $SFACE_MODEL (37MB)..."
    echo "This may take a few minutes depending on your connection..."
    echo ""

    if command -v curl &> /dev/null; then
        curl -L "$SFACE_URL" -o "$MODELS_DIR/$SFACE_MODEL"
    elif command -v wget &> /dev/null; then
        wget "$SFACE_URL" -O "$MODELS_DIR/$SFACE_MODEL"
    else
        echo "Error: Neither curl nor wget found"
        echo "Please install curl or wget, or download manually from:"
        echo "$SFACE_URL"
        exit 1
    fi

    echo ""
    echo "✓ Download complete"
fi

echo ""
echo "========================================"
echo "Verifying models..."
echo "========================================"
echo ""

# Check YuNet model (should be in git)
if [ -f "$MODELS_DIR/face_detection_yunet.onnx" ]; then
    echo "✓ face_detection_yunet.onnx found (227KB)"
else
    echo "⚠️  face_detection_yunet.onnx not found"
    echo "   This should be in the git repository"
fi

# Check SFace model
if [ -f "$MODELS_DIR/$SFACE_MODEL" ]; then
    SIZE=$(ls -lh "$MODELS_DIR/$SFACE_MODEL" | awk '{print $5}')
    echo "✓ $SFACE_MODEL found ($SIZE)"
else
    echo "❌ $SFACE_MODEL not found"
fi

echo ""
echo "========================================"
echo "All models ready!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Run ./setup.sh (if you haven't already)"
echo "2. Run ./run.sh to start the system"
echo ""
