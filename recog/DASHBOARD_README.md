# Face Recognition Dashboard

A beautiful, real-time web dashboard for visualizing and analyzing your face recognition tracking data.

## Features

### Visual Analytics
- **Real-time Statistics**: Total persons tracked, total detections, average quality scores
- **Quality Distribution Chart**: See the distribution of face quality across all persons
- **Detection Count Chart**: Visualize how many detections each person category has
- **Auto-refresh**: Dashboard updates every 5 seconds with new data

### Person Cards
Each tracked person gets a card showing:
- Thumbnail image (first detected face)
- Person ID
- Total number of detections
- Number of saved images
- Average quality score with visual progress bar
- First and last seen timestamps
- Feature vector status

### Advanced Filtering
Filter the displayed persons by:
- **Minimum Quality Score**: Show only high-quality detections (0.0 - 1.0)
- **Minimum Detections**: Filter out persons with few appearances
- **Feature Vector Status**: Show only persons with saved face embeddings

## Quick Start

### 1. Install Flask (if not already installed)
```bash
cd /Users/abdullahbashir/Desktop/CS124_Honors_CV/CS124HFA2025
source .venv/bin/activate
pip install flask
```

### 2. Start the Dashboard
```bash
cd recog
python dashboard.py
```

### 3. Open in Browser
Navigate to: **http://localhost:5000**

The dashboard will automatically load data from your `output/` directory.

## Usage Examples

### Basic Usage
1. Run the face recognition system to collect data:
   ```bash
   python main.py
   ```

2. In a separate terminal, start the dashboard:
   ```bash
   python dashboard.py
   ```

3. Open http://localhost:5000 in your browser

### Filtering Examples

**Show only high-quality detections:**
- Set "Min Quality Score" to `0.5`
- Click "Apply Filters"

**Show frequent visitors:**
- Set "Min Detections" to `10`
- Click "Apply Filters"

**Show only persons with saved features:**
- Select "Has Feature Vector" → "Yes"
- Click "Apply Filters"

**Combine filters:**
- Min Quality: `0.6`
- Min Detections: `5`
- Has Feature: `Yes`
- Click "Apply Filters"

## API Endpoints

The dashboard provides REST API endpoints you can use programmatically:

### Get All Persons
```bash
GET /api/persons
```

Optional query parameters:
- `min_quality`: Minimum quality score (0.0-1.0)
- `min_detections`: Minimum detection count
- `has_feature`: Filter by feature vector status (true/false)

Example:
```bash
curl "http://localhost:5000/api/persons?min_quality=0.5&min_detections=10"
```

### Get Person Details
```bash
GET /api/person/<person_id>
```

Example:
```bash
curl "http://localhost:5000/api/person/0"
```

### Get Statistics
```bash
GET /api/stats
```

Returns overall statistics and distributions.

## Customization

### Change Port
Edit `dashboard.py` line 188:
```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Change 5000 to your port
```

### Change Auto-refresh Interval
Edit `templates/dashboard.html` line 475:
```javascript
setInterval(() => {
    loadStats();
    loadPersons();
}, 5000);  // Change 5000 (5 seconds) to your interval in milliseconds
```

### Customize Colors
The dashboard uses a purple gradient theme. To customize, edit the CSS in `templates/dashboard.html`:
- Background gradient: Lines 15-16
- Primary color: Search for `#667eea` and `#764ba2`

## Data Structure

The dashboard reads data from:
```
recog/output/
├── person_000/
│   ├── 000000.jpg
│   ├── feature.npy
│   └── metadata.json
├── person_001/
│   └── ...
```

Each `metadata.json` contains:
```json
[
  {
    "time": "2025-12-10 12:00:00",
    "frame": 0,
    "bbox": [100, 150, 80, 100],
    "quality_score": 0.75
  }
]
```

## Troubleshooting

### Dashboard shows "No persons found"
- Make sure you've run `main.py` to collect some data first
- Check that the `output/` directory exists and contains person folders

### Dashboard won't start
- Make sure Flask is installed: `pip install flask`
- Check if port 5000 is already in use
- Try a different port (see Customization section)

### Images not displaying
- Check that the `output/` directory has `.jpg` files
- Verify file permissions

### Data not updating
- Make sure the face recognition system is running
- Wait for the 5-second auto-refresh
- Try manually refreshing the browser

## Advanced Features

### Remote Access
To access the dashboard from other devices on your network:
1. Find your computer's IP address
2. The dashboard is already configured to listen on `0.0.0.0`
3. Access from other devices: `http://<your-ip>:5000`

### Production Deployment
For production use, consider:
- Using a production WSGI server (gunicorn, waitress)
- Setting up NGINX as a reverse proxy
- Enabling HTTPS
- Disabling debug mode

## Technologies Used

- **Backend**: Python Flask
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Charts**: Chart.js
- **Data**: JSON, NumPy

## Screenshots

The dashboard includes:
- 4 statistics cards at the top
- Filter controls for customization
- 2 interactive charts (bar chart and doughnut chart)
- Grid of person cards with thumbnails and details

Enjoy analyzing your face recognition data!
