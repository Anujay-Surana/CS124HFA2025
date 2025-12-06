# Quick Start Guide - Analytics & Heat Maps

This guide gets you started with the new analytics features in under 5 minutes.

## Step 1: Install Dependencies

```bash
# Activate virtual environment
source .venv/bin/activate  # or: source venv/bin/activate

# Install new dependencies
pip install -r recog/requirements.txt
```

## Step 2: Run with Analytics

```bash
python recog/main_with_analytics.py
```

**Keyboard shortcuts:**
- `q` - Quit
- `h` - Toggle heat map
- `t` - Toggle trajectories
- `m` - Switch heat map mode (movement/density)
- `s` - Save analytics

## Step 3: View in Web Dashboard (Optional)

**Terminal 1 - Run analytics:**
```bash
python recog/main_with_analytics.py
```

**Terminal 2 - Start API server:**
```bash
python recog/web_api.py
```

**Terminal 3 - Open dashboard:**
```bash
open recog/example_dashboard.html
```

The dashboard will auto-update with live data at `http://localhost:5000`

## What You Get

### 1. Real-Time Visualization
- Color-coded heat maps (blue=low activity, red=high activity)
- Movement trajectories for each person
- Live statistics overlay

### 2. Analytics Data
- Dwell time per person
- Visit frequency
- Movement patterns
- Crowd density zones

### 3. Exported Data
All data saved to `recog/output/analytics/`:
- `analytics_summary.json` - Complete statistics
- `movement_heatmap.npy` - Movement heat map array
- `density_heatmap.npy` - Density heat map array

## Example Output

After running for a few minutes, you'll have:

```
recog/output/
├── person_000/
│   ├── 000000.jpg           # Best quality face crops
│   ├── 000005.jpg
│   ├── metadata.json        # Detection timestamps & bboxes
│   └── feature.npy          # 128-dim face features
├── analytics/
│   ├── analytics_summary.json
│   ├── movement_heatmap.npy
│   └── density_heatmap.npy
```

### Sample analytics_summary.json:
```json
{
  "unique_persons": 3,
  "total_frames": 1500,
  "persons": [
    {
      "id": 0,
      "dwell_time_seconds": 307.5,
      "visit_count": 2,
      "session_count": 2,
      "first_seen": "2025-12-05 14:30:15",
      "last_seen": "2025-12-05 14:35:22"
    }
  ]
}
```

## Web Dashboard Features

When you open the HTML dashboard, you'll see:

1. **Live Statistics Cards**
   - Unique persons detected
   - Total frames processed
   - Average dwell time
   - Total visits

2. **Heat Map Viewer**
   - Toggle between movement and density
   - Auto-refreshes every 5 seconds
   - Color-coded visualization

3. **Person Cards**
   - Individual person statistics
   - Session tracking
   - Time tracking
   - Image counts

## Common Use Cases

### Retail Store Analytics
```bash
# Run with larger grid for store layout
python -c "from recog.main_with_analytics import run_with_analytics; run_with_analytics(0, grid_size=60)"
```

### Security Monitoring
```bash
# Focus on trajectories
python recog/main_with_analytics.py
# Press 't' to show paths
# Press 'm' to switch to density view
```

### Event Crowd Analysis
```bash
# Run on video file
python -c "from recog.main_with_analytics import run_with_analytics; run_with_analytics('event_video.mp4')"
```

## Accessing Data via API

### Get Summary Stats
```bash
curl http://localhost:5000/api/stats/live
```

### Get Heat Map Image
```bash
curl http://localhost:5000/api/analytics/heatmap/image?type=movement -o heatmap.png
```

### Get All Persons
```bash
curl http://localhost:5000/api/analytics/persons
```

## Customization

### Change Grid Size
Edit `recog/main_with_analytics.py` line 361:
```python
run_with_analytics(0, show_heatmap=True, show_trajectories=True, grid_size=40)
```

- Smaller (20) = More detail, slower
- Larger (60) = Less detail, faster

### Change Heat Map Transparency
Edit `recog/main_with_analytics.py` line 58:
```python
visualizer = HeatmapVisualizer(alpha=0.4)  # 0.0 to 1.0
```

## Troubleshooting

### "No module named 'flask'"
```bash
pip install flask flask-cors
```

### Heat map doesn't show
1. Press `h` to toggle it on
2. Let it run for a few frames to collect data
3. Check that tracking is working (green boxes around faces)

### Dashboard shows "Failed to load"
1. Make sure API server is running: `python recog/web_api.py`
2. Check http://localhost:5000/api/health
3. Run main_with_analytics.py first to generate data

### Age/Gender not working
This feature requires additional model downloads (optional):
```bash
# See ANALYTICS_GUIDE.md for download links
```

## Next Steps

- Read [ANALYTICS_GUIDE.md](ANALYTICS_GUIDE.md) for detailed documentation
- Customize the dashboard HTML for your needs
- Integrate with your own backend system via the API
- Add database storage for historical analytics

## File Reference

| File | Purpose |
|------|---------|
| `main_with_analytics.py` | Main application with analytics |
| `analytics_tracker.py` | Analytics computation engine |
| `heatmap_visualizer.py` | Heat map rendering |
| `age_gender_detector.py` | Age/gender detection (optional) |
| `web_api.py` | REST API server |
| `example_dashboard.html` | Web dashboard template |

## Support

For issues:
1. Check [ANALYTICS_GUIDE.md](ANALYTICS_GUIDE.md)
2. Review [CLAUDE.md](CLAUDE.md) for architecture
3. Examine code comments in source files
