#!/bin/bash
# run_dashboard.sh - Quick start script for the dashboard

cd "$(dirname "$0")"

echo "========================================"
echo "Face Recognition Dashboard"
echo "========================================"
echo ""
echo "Installing Flask if needed..."
pip install flask > /dev/null 2>&1

echo "Starting dashboard server..."
echo ""
python dashboard.py
