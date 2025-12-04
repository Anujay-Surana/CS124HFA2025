#!/bin/bash
# Automated setup script for Face Recognition and Tracking System
# CS124 Honors Project

set -e  # Exit on error

echo "========================================"
echo "Face Recognition System - Setup"
echo "========================================"
echo ""

# Check Python version
echo "[1/6] Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3.8 or higher from https://www.python.org/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Found Python $PYTHON_VERSION"
echo ""

# Create virtual environment if it doesn't exist
echo "[2/6] Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "[3/6] Activating virtual environment..."
source .venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "[4/6] Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "[5/6] Installing dependencies..."
echo "This may take a few minutes (downloading ~100MB of packages)..."
pip install -r recog/requirements.txt --quiet

if [ $? -eq 0 ]; then
    echo "✓ All dependencies installed successfully"
else
    echo "❌ Error installing dependencies"
    exit 1
fi
echo ""

# Download model files
echo "[6/6] Downloading model files..."
./download_models.sh
echo ""

# Verify installation
echo "========================================"
echo "Verifying installation..."
echo "========================================"
echo ""

cd recog && python verify_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Setup completed successfully!"
    echo "========================================"
    echo ""
    echo "Next steps:"
    echo "1. Activate virtual environment:"
    echo "   source .venv/bin/activate"
    echo ""
    echo "2. Run the face recognition system:"
    echo "   ./run.sh"
    echo ""
    echo "3. Press 'q' to quit the application"
    echo ""
else
    echo ""
    echo "========================================"
    echo "⚠️  Setup completed with warnings"
    echo "========================================"
    echo ""
    echo "Please check the error messages above."
    echo "If you need help, see TROUBLESHOOTING.md"
    echo ""
fi
