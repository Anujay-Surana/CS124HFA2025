# web_api.py
"""
Flask-based web API for serving analytics data to a website dashboard.

Endpoints:
- GET /api/analytics/summary - Get overall analytics summary
- GET /api/analytics/persons - Get list of all persons with details
- GET /api/analytics/heatmap - Get heatmap data
- GET /api/analytics/live - Get current live stats (SSE)
- GET /api/person/<id> - Get specific person details
- GET /api/person/<id>/images - Get person's face crops
"""

from flask import Flask, jsonify, send_file, Response, request
from flask_cors import CORS
import json
import os
import numpy as np
import cv2
from io import BytesIO
import time
from datetime import datetime
from trait_filter import TraitFilter

app = Flask(__name__)
CORS(app)  # Enable CORS for web dashboard access

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
ANALYTICS_DIR = os.path.join(OUTPUT_DIR, "analytics")

# Initialize trait filter
trait_filter = TraitFilter(ANALYTICS_DIR)


def load_analytics_summary():
    """Load analytics summary from file."""
    summary_path = os.path.join(ANALYTICS_DIR, "analytics_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            return json.load(f)
    return {"total_frames": 0, "unique_persons": 0, "persons": []}


def load_person_metadata(person_id):
    """Load metadata for a specific person."""
    person_dir = os.path.join(OUTPUT_DIR, f"person_{person_id:03d}")
    metadata_path = os.path.join(person_dir, "metadata.json")

    if not os.path.exists(metadata_path):
        return None

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return metadata


def load_heatmap_data(heatmap_type="movement"):
    """Load heatmap data."""
    if heatmap_type == "movement":
        heatmap_path = os.path.join(ANALYTICS_DIR, "movement_heatmap.npy")
    else:
        heatmap_path = os.path.join(ANALYTICS_DIR, "density_heatmap.npy")

    if os.path.exists(heatmap_path):
        heatmap = np.load(heatmap_path)
        # Normalize to 0-1
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        return heatmap.tolist()

    return []


@app.route('/api/analytics/summary', methods=['GET'])
def get_analytics_summary():
    """Get overall analytics summary."""
    try:
        summary = load_analytics_summary()
        return jsonify({
            "success": True,
            "data": summary,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/analytics/persons', methods=['GET'])
def get_persons_list():
    """Get list of all detected persons."""
    try:
        summary = load_analytics_summary()
        persons = summary.get("persons", [])

        # Enrich with additional data
        enriched_persons = []
        for person in persons:
            person_id = person["id"]
            metadata = load_person_metadata(person_id)

            enriched_person = {
                **person,
                "image_count": len(metadata) if metadata else 0,
                "has_feature": os.path.exists(
                    os.path.join(OUTPUT_DIR, f"person_{person_id:03d}", "feature.npy")
                )
            }
            enriched_persons.append(enriched_person)

        return jsonify({
            "success": True,
            "data": enriched_persons,
            "count": len(enriched_persons)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/analytics/heatmap', methods=['GET'])
def get_heatmap():
    """Get heatmap data."""
    from flask import request

    try:
        heatmap_type = request.args.get('type', 'movement')  # 'movement' or 'density'
        heatmap_data = load_heatmap_data(heatmap_type)

        return jsonify({
            "success": True,
            "data": {
                "heatmap": heatmap_data,
                "type": heatmap_type
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/analytics/heatmap/image', methods=['GET'])
def get_heatmap_image():
    """Get heatmap as image."""
    from flask import request

    try:
        heatmap_type = request.args.get('type', 'movement')
        heatmap_data = np.array(load_heatmap_data(heatmap_type), dtype=np.float32)

        if heatmap_data.size == 0:
            return jsonify({"success": False, "error": "No heatmap data available"}), 404

        # Convert to 8-bit
        heatmap_8bit = (heatmap_data * 255).astype(np.uint8)

        # Apply Gaussian blur
        heatmap_8bit = cv2.GaussianBlur(heatmap_8bit, (15, 15), 0)

        # Apply colormap
        colored_heatmap = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)

        # Resize for better visibility
        target_size = (800, 600)
        colored_heatmap = cv2.resize(colored_heatmap, target_size, interpolation=cv2.INTER_LINEAR)

        # Encode as PNG
        _, buffer = cv2.imencode('.png', colored_heatmap)
        io_buf = BytesIO(buffer)

        return send_file(io_buf, mimetype='image/png')
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/person/<int:person_id>', methods=['GET'])
def get_person_details(person_id):
    """Get details for a specific person."""
    try:
        # Load from analytics summary
        summary = load_analytics_summary()
        person = next((p for p in summary.get("persons", []) if p["id"] == person_id), None)

        if not person:
            return jsonify({
                "success": False,
                "error": "Person not found"
            }), 404

        # Load metadata
        metadata = load_person_metadata(person_id)

        return jsonify({
            "success": True,
            "data": {
                **person,
                "images": metadata if metadata else []
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/person/<int:person_id>/images', methods=['GET'])
def get_person_images(person_id):
    """Get list of image URLs for a person."""
    try:
        person_dir = os.path.join(OUTPUT_DIR, f"person_{person_id:03d}")

        if not os.path.exists(person_dir):
            return jsonify({
                "success": False,
                "error": "Person not found"
            }), 404

        # Get all image files
        images = []
        for filename in os.listdir(person_dir):
            if filename.endswith('.jpg'):
                images.append({
                    "filename": filename,
                    "url": f"/api/person/{person_id}/image/{filename}"
                })

        return jsonify({
            "success": True,
            "data": images,
            "count": len(images)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/person/<int:person_id>/image/<filename>', methods=['GET'])
def get_person_image(person_id, filename):
    """Get a specific image for a person."""
    try:
        image_path = os.path.join(OUTPUT_DIR, f"person_{person_id:03d}", filename)

        if not os.path.exists(image_path):
            return jsonify({
                "success": False,
                "error": "Image not found"
            }), 404

        return send_file(image_path, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/stats/live', methods=['GET'])
def get_live_stats():
    """Get current live statistics (for real-time updates)."""
    try:
        summary = load_analytics_summary()

        # Calculate aggregate stats
        total_dwell_time = sum(p.get("dwell_time_seconds", 0) for p in summary.get("persons", []))
        total_visits = sum(p.get("visit_count", 0) for p in summary.get("persons", []))

        stats = {
            "unique_persons": summary.get("unique_persons", 0),
            "total_frames": summary.get("total_frames", 0),
            "total_dwell_time_seconds": total_dwell_time,
            "total_visits": total_visits,
            "average_dwell_time": total_dwell_time / summary.get("unique_persons", 1),
            "timestamp": datetime.now().isoformat()
        }

        return jsonify({
            "success": True,
            "data": stats
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/filter/persons', methods=['GET'])
def filter_persons():
    """Filter persons by demographic traits."""
    try:
        # Get query parameters
        age_ranges = request.args.get('age')
        genders = request.args.get('gender')
        ethnicities = request.args.get('ethnicity')

        # Parse comma-separated values
        age_list = age_ranges.split(',') if age_ranges else None
        gender_list = genders.split(',') if genders else None
        ethnicity_list = ethnicities.split(',') if ethnicities else None

        # Get filtered analytics
        results = trait_filter.get_filtered_analytics(
            age_ranges=age_list,
            genders=gender_list,
            ethnicities=ethnicity_list
        )

        return jsonify({
            "success": True,
            "data": results
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/demographics/summary', methods=['GET'])
def get_demographics():
    """Get demographic breakdown of all persons."""
    try:
        summary = trait_filter.get_demographic_summary()
        return jsonify({
            "success": True,
            "data": summary
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "success": True,
        "status": "running",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/', methods=['GET'])
def index():
    """API documentation."""
    return jsonify({
        "name": "Face Recognition Analytics API",
        "version": "1.0",
        "endpoints": {
            "GET /api/analytics/summary": "Get overall analytics summary",
            "GET /api/analytics/persons": "Get list of all persons",
            "GET /api/analytics/heatmap": "Get heatmap data (query: type=movement|density)",
            "GET /api/analytics/heatmap/image": "Get heatmap as image",
            "GET /api/person/<id>": "Get specific person details",
            "GET /api/person/<id>/images": "Get person's images list",
            "GET /api/person/<id>/image/<filename>": "Get specific image",
            "GET /api/stats/live": "Get live statistics",
            "GET /api/filter/persons": "Filter persons by traits (query: age, gender, ethnicity)",
            "GET /api/demographics/summary": "Get demographic breakdown",
            "GET /api/health": "Health check"
        }
    })


def run_api_server(host='0.0.0.0', port=5000, debug=False):
    """
    Run the Flask API server.

    Args:
        host: Host address
        port: Port number
        debug: Enable debug mode
    """
    print(f"Starting Face Recognition Analytics API server...")
    print(f"Server running at: http://{host}:{port}")
    print(f"API Documentation: http://{host}:{port}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Analytics directory: {ANALYTICS_DIR}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_api_server(debug=True)
