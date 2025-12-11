#!/bin/bash
# Double-click this file to start the dashboard

cd "$(dirname "$0")"
source .venv/bin/activate
cd recog
python dashboard.py
