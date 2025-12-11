#!/usr/bin/env python3
# dashboard.py - Web dashboard for face recognition tracking data

import os
import json
import base64
from datetime import datetime
from flask import Flask, render_template, jsonify, request, send_from_directory
import numpy as np

app = Flask(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "templates")
STATIC_DIR = os.path.join(SCRIPT_DIR, "static")

# Create directories if they don't exist
os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


def load_person_data():
    """Load all person data from output directory."""
    persons = []

    if not os.path.exists(OUTPUT_DIR):
        return persons

    for dirname in sorted(os.listdir(OUTPUT_DIR)):
        if not dirname.startswith("person_"):
            continue

        person_dir = os.path.join(OUTPUT_DIR, dirname)
        person_id = int(dirname.split("_")[1])

        # Load metadata
        meta_path = os.path.join(person_dir, "metadata.json")
        metadata = []
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except:
                metadata = []

        # Load feature vector if exists
        feature_path = os.path.join(person_dir, "feature.npy")
        has_feature = os.path.exists(feature_path)

        # Get image files
        image_files = sorted([f for f in os.listdir(person_dir) if f.endswith(".jpg")])

        # Calculate statistics
        total_detections = len(metadata)
        avg_quality = sum(m.get("quality_score", 0) for m in metadata) / total_detections if total_detections > 0 else 0

        # Get first and last seen times
        first_seen = metadata[0].get("time", "Unknown") if metadata else "Unknown"
        last_seen = metadata[-1].get("time", "Unknown") if metadata else "Unknown"

        # Get thumbnail (first image)
        thumbnail = None
        if image_files:
            thumb_path = os.path.join(person_dir, image_files[0])
            if os.path.exists(thumb_path):
                with open(thumb_path, "rb") as f:
                    thumbnail = base64.b64encode(f.read()).decode('utf-8')

        persons.append({
            "id": person_id,
            "folder": dirname,
            "total_detections": total_detections,
            "avg_quality": round(avg_quality, 3),
            "first_seen": first_seen,
            "last_seen": last_seen,
            "has_feature": has_feature,
            "image_count": len(image_files),
            "thumbnail": thumbnail,
            "metadata": metadata,
            "images": image_files
        })

    return persons


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/persons')
def get_persons():
    """API endpoint to get all person data."""
    persons = load_person_data()

    # Apply filters if provided
    min_quality = request.args.get('min_quality', type=float)
    min_detections = request.args.get('min_detections', type=int)
    has_feature = request.args.get('has_feature', type=str)

    filtered_persons = persons

    if min_quality is not None:
        filtered_persons = [p for p in filtered_persons if p['avg_quality'] >= min_quality]

    if min_detections is not None:
        filtered_persons = [p for p in filtered_persons if p['total_detections'] >= min_detections]

    if has_feature == 'true':
        filtered_persons = [p for p in filtered_persons if p['has_feature']]
    elif has_feature == 'false':
        filtered_persons = [p for p in filtered_persons if not p['has_feature']]

    return jsonify(filtered_persons)


@app.route('/api/person/<int:person_id>')
def get_person_detail(person_id):
    """API endpoint to get detailed data for a specific person."""
    persons = load_person_data()
    person = next((p for p in persons if p['id'] == person_id), None)

    if person is None:
        return jsonify({"error": "Person not found"}), 404

    return jsonify(person)


@app.route('/api/stats')
def get_stats():
    """API endpoint to get overall statistics."""
    persons = load_person_data()

    total_persons = len(persons)
    total_detections = sum(p['total_detections'] for p in persons)
    avg_quality_all = sum(p['avg_quality'] * p['total_detections'] for p in persons) / total_detections if total_detections > 0 else 0
    persons_with_features = sum(1 for p in persons if p['has_feature'])

    # Quality distribution
    quality_ranges = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for p in persons:
        q = p['avg_quality']
        if q < 0.2:
            quality_ranges["0.0-0.2"] += 1
        elif q < 0.4:
            quality_ranges["0.2-0.4"] += 1
        elif q < 0.6:
            quality_ranges["0.4-0.6"] += 1
        elif q < 0.8:
            quality_ranges["0.6-0.8"] += 1
        else:
            quality_ranges["0.8-1.0"] += 1

    # Detection count distribution
    detection_ranges = {"1-5": 0, "6-10": 0, "11-20": 0, "21-50": 0, "50+": 0}
    for p in persons:
        d = p['total_detections']
        if d <= 5:
            detection_ranges["1-5"] += 1
        elif d <= 10:
            detection_ranges["6-10"] += 1
        elif d <= 20:
            detection_ranges["11-20"] += 1
        elif d <= 50:
            detection_ranges["21-50"] += 1
        else:
            detection_ranges["50+"] += 1

    return jsonify({
        "total_persons": total_persons,
        "total_detections": total_detections,
        "avg_quality": round(avg_quality_all, 3),
        "persons_with_features": persons_with_features,
        "quality_distribution": quality_ranges,
        "detection_distribution": detection_ranges
    })


@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve images from the output directory."""
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == '__main__':
    print("=" * 70)
    print("Face Recognition Dashboard")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Starting server at http://localhost:8080")
    print("Press Ctrl+C to stop")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=8080)
