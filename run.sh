#!/bin/bash
# Convenience script to run face recognition system

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found (.venv or venv)"
    exit 1
fi

# Check if verify flag is passed
if [ "$1" == "verify" ]; then
    echo "Running setup verification..."
    cd recog && python verify_setup.py
else
    echo "Starting face recognition system..."
    cd recog && python main.py
fi
