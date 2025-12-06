# Analytics & Heat Map System - Complete Overview

## What's New

This project now includes comprehensive analytics and heat map generation capabilities for crowd density analysis, movement tracking, and characteristic identification.

## Features

### 1. Real-Time Heat Maps ✅
- **Movement Heat Maps**: Visualize where people move most frequently
- **Density Heat Maps**: Show crowd concentration in real-time
- Color-coded visualization (blue → green → yellow → red)
- Adjustable grid granularity
- Live overlay on video feed

### 2. Analytics Tracking ✅
- **Dwell Time**: Track how long each person stays in frame
- **Frequency Analysis**: Count visits and sessions per person
- **Movement Trajectories**: Record and visualize paths
- **Area Heat Zones**: Identify high-traffic areas
- Automatic data export to JSON/NumPy formats

### 3. Age & Gender Detection ✅
- Age range estimation (8 categories)
- Gender classification
- Confidence scoring
- Optional feature (requires model downloads)

### 4. Web API & Dashboard ✅
- RESTful API for data access
- Real-time statistics updates
- Beautiful HTML dashboard
- Image serving endpoints
- JSON data export

## File Structure

```
recog/
├── main_with_analytics.py       # Enhanced main with analytics
├── analytics_tracker.py          # Analytics computation engine
├── heatmap_visualizer.py         # Heat map rendering
├── age_gender_detector.py        # Age/gender detection
├── web_api.py                    # Flask REST API server
├── example_dashboard.html        # Web dashboard template
└── output/
    ├── person_XXX/               # Person-specific data
    │   ├── *.jpg                 # Face crops
    │   ├── metadata.json         # Detection metadata
    │   └── feature.npy           # Face embeddings
    └── analytics/
        ├── analytics_summary.json    # Complete analytics
        ├── movement_heatmap.npy      # Movement data
        └── density_heatmap.npy       # Density data
```

## Quick Start

### Option 1: Analytics Only (No Web Interface)

```bash
# Activate environment
source .venv/bin/activate

# Install dependencies
pip install -r recog/requirements.txt

# Run with analytics
python recog/main_with_analytics.py
```

**Controls:**
- `q` - Quit
- `h` - Toggle heat map overlay
- `t` - Toggle trajectory lines
- `m` - Switch heat map mode
- `s` - Save analytics to disk

### Option 2: Full System (With Web Dashboard)

**Terminal 1 - Run Detection:**
```bash
python recog/main_with_analytics.py
```

**Terminal 2 - Start API:**
```bash
python recog/web_api.py
```

**Terminal 3 - Open Dashboard:**
```bash
# macOS
open recog/example_dashboard.html

# Linux
xdg-open recog/example_dashboard.html

# Or just open in browser:
# file:///path/to/recog/example_dashboard.html
```

Dashboard connects to `http://localhost:5000` and auto-refreshes.

## Key Components

### 1. Analytics Tracker
Tracks comprehensive metrics:
```python
from analytics_tracker import AnalyticsTracker

analytics = AnalyticsTracker(width, height, grid_size=40)
analytics.update(tracked_objects, visible_objects)

# Get metrics
dwell_time = analytics.get_person_dwell_time(person_id)
heatmap = analytics.get_movement_heatmap(normalize=True)
summary = analytics.get_analytics_summary()
```

**Metrics Collected:**
- Position history (last 100 points/person)
- First/last seen timestamps
- Total dwell time in seconds
- Visit count (total appearances)
- Session count (distinct visits)
- Grid-based heat maps (movement & density)

### 2. Heat Map Visualizer
Renders heat maps on frames:
```python
from heatmap_visualizer import HeatmapVisualizer

visualizer = HeatmapVisualizer(alpha=0.4)

# Overlay heat map
frame = visualizer.overlay_heatmap(frame, heatmap, colormap=cv2.COLORMAP_JET)

# Add legend
frame = visualizer.add_legend_to_frame(frame, "bottom-right")

# Draw trajectories
frame = visualizer.draw_trajectory(frame, trajectory, color=(0,255,255))
```

**Features:**
- Multiple color maps (JET, HOT, COOL, etc.)
- Gaussian blur for smoothing
- Automatic normalization
- Legend generation
- Grid overlays

### 3. Web API
RESTful endpoints for dashboard integration:

```bash
# Health check
GET http://localhost:5000/api/health

# Analytics summary
GET http://localhost:5000/api/analytics/summary

# All persons
GET http://localhost:5000/api/analytics/persons

# Heat map data
GET http://localhost:5000/api/analytics/heatmap?type=movement

# Heat map image
GET http://localhost:5000/api/analytics/heatmap/image?type=density

# Person details
GET http://localhost:5000/api/person/0

# Person images
GET http://localhost:5000/api/person/0/images
GET http://localhost:5000/api/person/0/image/000000.jpg

# Live stats
GET http://localhost:5000/api/stats/live
```

### 4. Age/Gender Detection (Optional)
Requires downloading model files:

```python
from age_gender_detector import AgeGenderDetector

detector = AgeGenderDetector()

if detector.is_available():
    results = detector.detect_age_gender(face_crop)
    print(f"Age: {results['age_range']}")      # (25-32)
    print(f"Gender: {results['gender']}")      # Male/Female
    print(f"Confidence: {results['gender_confidence']:.2f}")
```

**Age Ranges:**
- (0-2), (4-6), (8-12), (15-20)
- (25-32), (38-43), (48-53), (60-100)

**Model Downloads:**
See [ANALYTICS_GUIDE.md](ANALYTICS_GUIDE.md) for download instructions (~44MB total)

## Data Formats

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
      "visit_count": 2,
      "session_count": 2,
      "trajectory_points": 98
    }
  ]
}
```

### Heat Map Arrays
- **Format**: NumPy 2D float32 arrays
- **Dimensions**: (grid_rows, grid_cols)
- **Values**: Normalized 0-1 or raw counts

Load with:
```python
import numpy as np
heatmap = np.load('output/analytics/movement_heatmap.npy')
```

## Configuration

### Grid Size
Controls heat map granularity (in `main_with_analytics.py`):
```python
run_with_analytics(source=0, grid_size=40)
```
- **20**: High detail, slower
- **40**: Balanced (default)
- **60**: Coarse, faster

### Heat Map Transparency
```python
visualizer = HeatmapVisualizer(alpha=0.4)  # 0.0 = transparent, 1.0 = opaque
```

### Trajectory Length
```python
trajectory = analytics.get_person_trajectory(person_id, max_points=30)
```

## Use Cases

### 1. Retail Analytics
- Track customer dwell time in store sections
- Identify high-traffic product areas
- Analyze shopping patterns
- Optimize store layout

### 2. Security & Surveillance
- Detect unusual movement patterns
- Track person trajectories
- Identify loitering behavior
- Monitor restricted areas

### 3. Event Management
- Crowd flow analysis
- Bottleneck detection
- Entry/exit tracking
- Occupancy monitoring

### 4. Smart Buildings
- Space utilization metrics
- Traffic pattern optimization
- Wait time analysis
- Resource allocation

## Performance

### Optimization Tips

**1. Grid Size**
- Larger grid = faster processing, less detail
- Use 40-60 for real-time, 20-30 for analysis

**2. Update Frequency**
- Update heat maps every N frames instead of every frame
- Save analytics periodically with 's' key

**3. Trajectory Length**
- Limit to 20-30 points for real-time display
- Full history (100 points) for analysis only

**4. Web API**
- Implement caching for frequently accessed data
- Use SSE for live updates
- Add rate limiting for production

### Typical Performance
- **720p**: 25-30 FPS (grid_size=40)
- **1080p**: 15-20 FPS (grid_size=40)
- **1080p**: 20-25 FPS (grid_size=60)

## API Examples

### Python
```python
import requests

# Get live stats
response = requests.get('http://localhost:5000/api/stats/live')
data = response.json()
print(f"Unique persons: {data['data']['unique_persons']}")

# Download heatmap image
response = requests.get('http://localhost:5000/api/analytics/heatmap/image?type=movement')
with open('heatmap.png', 'wb') as f:
    f.write(response.content)
```

### JavaScript
```javascript
// Fetch analytics summary
fetch('http://localhost:5000/api/analytics/summary')
    .then(res => res.json())
    .then(data => console.log(data.data));

// Auto-refresh heatmap
setInterval(() => {
    document.getElementById('heatmap').src =
        'http://localhost:5000/api/analytics/heatmap/image?t=' + Date.now();
}, 5000);
```

### cURL
```bash
# Get summary
curl http://localhost:5000/api/analytics/summary | jq

# Download heatmap
curl http://localhost:5000/api/analytics/heatmap/image?type=movement -o heatmap.png

# Get person details
curl http://localhost:5000/api/person/0 | jq
```

## Dashboard Features

The included HTML dashboard provides:

1. **Live Statistics**
   - Unique persons count
   - Total frames processed
   - Average dwell time
   - Total visits

2. **Interactive Heat Map**
   - Toggle movement/density modes
   - Auto-refresh every 5 seconds
   - Color-coded visualization

3. **Person Cards**
   - Individual statistics per person
   - Session tracking
   - Time-based metrics
   - Image counts

## Dependencies

```txt
# Core (already installed)
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0
numpy>=1.26.0
imutils>=0.5.4

# New (for analytics & web API)
flask>=2.3.0
flask-cors>=4.0.0
```

Install with:
```bash
pip install -r recog/requirements.txt
```

## Architecture

```
User Input (Camera/Video)
        ↓
    YuNet Detector → Face Detection
        ↓
    SFace Recognizer → Feature Extraction
        ↓
    CentroidTracker → Person Tracking
        ↓
    AnalyticsTracker → Metrics Collection
        ↓
    ├→ HeatmapVisualizer → Visual Overlay
    └→ Data Export → JSON/NPY files
                ↓
            Web API → Dashboard
```

## Troubleshooting

### Heat map not visible
1. Press 'h' to enable
2. Run for a few frames to collect data
3. Verify tracking is working (green boxes)

### Dashboard connection failed
1. Start API server: `python recog/web_api.py`
2. Verify: `curl http://localhost:5000/api/health`
3. Check CORS settings if accessing from different domain

### Poor performance
1. Increase grid_size (40 → 60)
2. Disable trajectories (press 't')
3. Lower camera resolution
4. Reduce trajectory max_points

### Flask not found
```bash
pip install flask flask-cors
```

## Documentation

- **[QUICK_START.md](QUICK_START.md)** - Get started in 5 minutes
- **[ANALYTICS_GUIDE.md](ANALYTICS_GUIDE.md)** - Comprehensive documentation
- **[CLAUDE.md](CLAUDE.md)** - Original architecture details

## Future Enhancements

Potential additions:
- Database integration (PostgreSQL/MongoDB)
- Advanced React/Vue.js dashboard
- Multi-camera aggregation
- Alert system based on thresholds
- Historical trend analysis
- Export to BI tools (Tableau, Power BI)
- Real-time streaming with WebSockets
- Custom analytics plugins

## License

Part of CS124 Honors Project - Face Recognition System

## Credits

Built on top of:
- OpenCV YuNet face detector
- OpenCV SFace recognizer
- Custom centroid tracking algorithm
- Flask web framework
