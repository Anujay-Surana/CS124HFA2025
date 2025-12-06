# Analytics and Heat Map Guide

This guide explains how to use the new analytics features including heat maps, movement tracking, and web API integration.

## Features Added

### 1. Real-Time Heat Maps
- **Movement Heat Maps**: Visualize where people move most frequently
- **Density Heat Maps**: Show crowd concentration areas
- Color-coded visualization (blue=low, red=high)
- Adjustable grid size for granularity

### 2. Analytics Tracking
- **Dwell Time**: Track how long each person stays in frame
- **Frequency**: Count visits and sessions per person
- **Trajectories**: Record movement paths
- **Area Zones**: Identify hot spots and traffic patterns

### 3. Age/Gender Detection
- Optional age range estimation (8 categories)
- Gender classification
- Requires downloading additional models (see below)

### 4. Web API & Dashboard Integration
- RESTful API for accessing analytics data
- JSON endpoints for web dashboards
- Real-time statistics updates
- Image serving for face crops and heat maps

---

## Quick Start

### Running with Analytics

```bash
# Activate virtual environment
source .venv/bin/activate  # or: source venv/bin/activate

# Run the analytics-enhanced version
python recog/main_with_analytics.py
```

### Keyboard Controls

While the application is running:
- **q** - Quit the application
- **h** - Toggle heat map overlay on/off
- **t** - Toggle trajectory lines on/off
- **m** - Switch heat map mode (movement ↔ density)
- **s** - Save current analytics to disk

### Output Structure

```
recog/output/
├── person_000/
│   ├── 000000.jpg          # Face crops
│   ├── 000005.jpg
│   ├── metadata.json       # Detection metadata
│   └── feature.npy         # Face feature vector
├── person_001/
│   └── ...
└── analytics/
    ├── movement_heatmap.npy    # Movement heat map data
    ├── density_heatmap.npy     # Density heat map data
    └── analytics_summary.json  # Complete analytics report
```

---

## Analytics Modules

### 1. Analytics Tracker (`analytics_tracker.py`)

Tracks comprehensive analytics for each detected person:

```python
from analytics_tracker import AnalyticsTracker

# Initialize with frame dimensions
analytics = AnalyticsTracker(
    frame_width=1920,
    frame_height=1080,
    grid_size=40  # 40x40 pixel cells
)

# Update each frame
analytics.update(tracked_objects, visible_objects)

# Get person statistics
dwell_time = analytics.get_person_dwell_time(person_id)
trajectory = analytics.get_person_trajectory(person_id)

# Get heat maps
movement_heatmap = analytics.get_movement_heatmap(normalize=True)
density_heatmap = analytics.get_density_heatmap(normalize=True)

# Export all data
analytics.export_heatmaps("output/analytics/")
```

**Tracked Metrics:**
- Position history (last 100 positions per person)
- First/last seen timestamps
- Total dwell time (seconds in frame)
- Visit count (number of appearances)
- Session count (distinct visits)
- Movement heat map (cumulative position density)
- Density heat map (people per grid cell)

### 2. Heat Map Visualizer (`heatmap_visualizer.py`)

Generates visual heat map overlays:

```python
from heatmap_visualizer import HeatmapVisualizer

visualizer = HeatmapVisualizer(alpha=0.4)

# Overlay heat map on frame
frame_with_heatmap = visualizer.overlay_heatmap(
    frame,
    heatmap,
    colormap=cv2.COLORMAP_JET
)

# Add legend
frame_with_legend = visualizer.add_legend_to_frame(
    frame_with_heatmap,
    "bottom-right"
)

# Draw trajectory
frame_with_path = visualizer.draw_trajectory(
    frame,
    trajectory,
    color=(0, 255, 255)
)

# Draw grid overlay
frame_with_grid = visualizer.draw_grid(frame, grid_size=40)
```

**Features:**
- Multiple color maps (JET, HOT, COOL, etc.)
- Gaussian blur smoothing
- Trajectory visualization
- Side-by-side comparisons
- Grid overlays with density values

### 3. Age/Gender Detector (`age_gender_detector.py`)

Optional age and gender detection:

```python
from age_gender_detector import AgeGenderDetector

detector = AgeGenderDetector()

if detector.is_available():
    # Detect from face crop
    results = detector.detect_age_gender(face_crop)

    print(f"Age: {results['age_range']}")
    print(f"Gender: {results['gender']}")
    print(f"Confidence: {results['gender_confidence']:.2f}")
```

**Age Ranges:**
- (0-2), (4-6), (8-12), (15-20)
- (25-32), (38-43), (48-53), (60-100)

**Note**: Requires downloading model files (see Installation section)

---

## Web API & Dashboard Integration

### Starting the API Server

```bash
# Run the Flask API server
python recog/web_api.py
```

Server will start at: `http://localhost:5000`

### API Endpoints

#### Get Analytics Summary
```bash
GET http://localhost:5000/api/analytics/summary
```

Response:
```json
{
  "success": true,
  "data": {
    "total_frames": 1500,
    "unique_persons": 3,
    "persons": [...]
  }
}
```

#### Get All Persons
```bash
GET http://localhost:5000/api/analytics/persons
```

#### Get Heat Map Data
```bash
GET http://localhost:5000/api/analytics/heatmap?type=movement
GET http://localhost:5000/api/analytics/heatmap?type=density
```

#### Get Heat Map Image
```bash
GET http://localhost:5000/api/analytics/heatmap/image?type=movement
```

Returns a PNG image of the heat map.

#### Get Person Details
```bash
GET http://localhost:5000/api/person/0
```

#### Get Person Images
```bash
GET http://localhost:5000/api/person/0/images
GET http://localhost:5000/api/person/0/image/000000.jpg
```

#### Get Live Statistics
```bash
GET http://localhost:5000/api/stats/live
```

### Example Web Dashboard (HTML/JavaScript)

```html
<!DOCTYPE html>
<html>
<head>
    <title>Analytics Dashboard</title>
</head>
<body>
    <h1>Face Recognition Analytics</h1>
    <div id="stats"></div>
    <img id="heatmap" src="http://localhost:5000/api/analytics/heatmap/image?type=movement">

    <script>
        // Fetch live stats
        fetch('http://localhost:5000/api/stats/live')
            .then(res => res.json())
            .then(data => {
                document.getElementById('stats').innerHTML =
                    `Unique Persons: ${data.data.unique_persons}`;
            });

        // Refresh heatmap every 5 seconds
        setInterval(() => {
            document.getElementById('heatmap').src =
                'http://localhost:5000/api/analytics/heatmap/image?type=movement&t=' +
                new Date().getTime();
        }, 5000);
    </script>
</body>
</html>
```

---

## Installation

### 1. Core Dependencies

```bash
pip install -r recog/requirements.txt
```

### 2. Age/Gender Models (Optional)

Download the following models for age/gender detection:

**Option A: Direct Downloads**
```bash
cd recog/models

# Age model
wget https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/age_net.caffemodel

# Gender model
wget https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/gender_net.caffemodel
```

**Prototxt files** (get from OpenCV samples or create manually):
- `age_deploy.prototxt`
- `gender_deploy.prototxt`

**Option B: From GitHub**

Visit: https://github.com/GilLevi/AgeGenderDeepLearning

Download all 4 files and place in `recog/models/`

**Models Size**: ~44MB total

---

## Configuration Options

### Grid Size

Adjust heat map granularity in `main_with_analytics.py`:

```python
run_with_analytics(
    source=0,
    grid_size=20   # Smaller = more detailed (default: 40)
)
```

**Recommended values:**
- 20 pixels: High detail (slow on large frames)
- 40 pixels: Balanced (default)
- 60 pixels: Coarse (faster)

### Heat Map Transparency

Adjust overlay opacity in `heatmap_visualizer.py`:

```python
visualizer = HeatmapVisualizer(alpha=0.4)  # 0=transparent, 1=opaque
```

### Trajectory Length

Control trajectory trail length:

```python
trajectory = analytics.get_person_trajectory(
    person_id,
    max_points=30  # Number of points to show
)
```

---

## Analytics Data Format

### analytics_summary.json

```json
{
  "total_frames": 1500,
  "unique_persons": 3,
  "persons": [
    {
      "id": 0,
      "first_seen": "2025-12-05 14:30:15",
      "last_seen": "2025-12-05 14:35:22",
      "dwell_time_seconds": 307.5,
      "visit_count": 1,
      "session_count": 1,
      "trajectory_points": 98
    }
  ]
}
```

### Heat Map Arrays (.npy files)

- **Format**: NumPy 2D array (float32)
- **Dimensions**: (grid_rows, grid_cols)
- **Values**: Cumulative counts or normalized 0-1

Load with:
```python
import numpy as np
heatmap = np.load('movement_heatmap.npy')
```

---

## Use Cases

### 1. Retail Analytics
- Track customer dwell time in store sections
- Identify high-traffic areas
- Analyze shopping patterns
- Measure engagement zones

### 2. Security Monitoring
- Detect unusual movement patterns
- Track person trajectories
- Identify loitering behavior
- Monitor crowd density

### 3. Event Management
- Crowd flow analysis
- Bottleneck detection
- Entry/exit tracking
- Occupancy heat maps

### 4. Smart Buildings
- Space utilization metrics
- Traffic pattern optimization
- Wait time analysis
- Queue management

---

## Performance Tips

### 1. Grid Size Optimization
- Larger grid = faster processing, less detail
- Start with 40-60 pixels for real-time use
- Use 20-30 pixels for post-processing analysis

### 2. Trajectory Smoothing
- Limit max_points to 20-30 for real-time display
- Use full history (100 points) for analysis

### 3. Heat Map Updates
- Update heat maps every N frames instead of every frame
- Export periodically with 's' key instead of continuous saving

### 4. Web API Optimization
- Cache analytics summary between requests
- Use SSE (Server-Sent Events) for live updates
- Implement rate limiting for production

---

## Troubleshooting

### Heat Map Not Showing
- Press 'h' to toggle heat map on
- Ensure analytics data exists (run for a few frames first)
- Check grid_size matches frame dimensions

### Web API 404 Errors
- Run `main_with_analytics.py` first to generate data
- Check that `output/analytics/` directory exists
- Verify analytics_summary.json was created

### Age/Gender Detection Fails
- Download model files (see Installation section)
- Check file paths in `recog/models/`
- Models are optional - system works without them

### Poor Performance
- Increase grid_size (40 → 60)
- Disable trajectories with 't' key
- Reduce max_points in trajectory calls
- Lower camera resolution

---

## API Rate Limits (Production)

For production deployments, add rate limiting:

```python
from flask_limiter import Limiter

limiter = Limiter(app, default_limits=["100 per minute"])

@app.route('/api/analytics/summary')
@limiter.limit("10 per minute")
def get_analytics_summary():
    # ...
```

---

## Next Steps

1. **Custom Analytics**: Extend `AnalyticsTracker` for domain-specific metrics
2. **Database Integration**: Store analytics in PostgreSQL/MongoDB
3. **Real-Time Dashboard**: Build React/Vue.js dashboard
4. **Alerts System**: Trigger alerts based on thresholds
5. **Multi-Camera**: Aggregate analytics across cameras
6. **Historical Analysis**: Compare analytics across time periods

---

## Support

For issues or questions:
- Check the main README.md
- Review CLAUDE.md for architecture details
- Examine example code in main_with_analytics.py
