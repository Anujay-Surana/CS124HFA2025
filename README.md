# Face Recognition and Tracking System

CS124 Honors Project - Real-time face detection, recognition, and tracking system.

## Features

- **Face Detection**: YuNet detector finds faces in video frames
- **Face Recognition**: SFace extracts unique 128-dim features for each face
- **Intelligent Tracking**: Hybrid position + feature matching
- **Persistent IDs**: Same person gets same ID even after leaving and returning to frame
- **Auto-organized Output**: Face crops saved in person-specific folders

## Quick Start for Team Members

### First Time Setup (One Command!)

```bash
./setup.sh
```

This automated script will:
- Check Python installation
- Create virtual environment
- Install all dependencies
- Verify everything works

**That's it!** Takes about 2-3 minutes.

### Running the System

```bash
# Activate virtual environment (if not already activated)
source .venv/bin/activate

# Run face recognition
./run.sh
```

Press 'q' to quit the application.

### Manual Setup (if automated setup fails)

```bash
# 1. Create virtual environment
python3 -m venv .venv

# 2. Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r recog/requirements.txt

# 4. Verify installation
./run.sh verify
```

**Having issues?** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for solutions.

## How It Works

### Architecture

1. **YuNet Detection**: Detects faces and facial landmarks in each frame
2. **SFace Recognition**: Extracts 128-dimensional feature vector for each detected face
3. **Hybrid Tracking**:
   - Combines position distance (30%) and feature similarity (70%)
   - Uses cosine similarity to compare face features
   - Exponential moving average smooths features over time
4. **Output**: Saves face crops to `recog/output/person_XXX/` directories

### Key Improvements Over Basic Tracking

**Before (position-only tracking):**
- Same person gets different IDs when they move quickly
- Same person gets new ID after leaving and returning
- Cannot distinguish between different people in similar positions

**After (feature-based recognition):**
- Same person keeps ID across entire session
- Robust to movement and temporary occlusion
- Can distinguish between different people even when close together

### Configuration

Edit `main.py` line 40-44 to adjust tracking parameters:

```python
ct = CentroidTracker(
    max_disappeared=50,          # Frames before ID removal (higher = more persistent)
    max_distance=150,            # Max pixel distance for position matching
    min_feature_similarity=0.4   # Min cosine similarity for face matching (0-1)
)
```

## Output Structure

```
recog/output/
├── person_000/
│   ├── 000000.jpg
│   ├── 000005.jpg
│   ├── 000010.jpg
│   └── metadata.json
├── person_001/
│   ├── 000015.jpg
│   └── metadata.json
└── ...
```

Each `metadata.json` contains:
```json
[
  {
    "time": "2025-12-01 22:00:00",
    "frame": 0,
    "bbox": [100, 150, 80, 100]
  }
]
```

## Models

- **YuNet** (227 KB): Face detection
- **SFace** (37 MB): Face recognition features

Both models are included in `recog/models/`.

## Requirements

- Python 3.x
- OpenCV >= 4.8.0 (with contrib)
- NumPy >= 1.26.0
- 8GB RAM recommended
- Webcam or video file

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions to common issues:

- Python installation issues
- Virtual environment problems
- Dependency installation errors
- Camera/webcam not working
- Performance issues
- Platform-specific problems (macOS/Windows/Linux)

Quick fixes for most common issues:

**"Module not found" errors:**
```bash
source .venv/bin/activate
pip install -r recog/requirements.txt
```

**"YuNet interface not available":**
```bash
pip install opencv-contrib-python>=4.8.0
```

**Camera not opening:**
- Check camera permissions in System Preferences (macOS) or Settings (Windows)
- Make sure no other app is using the camera

## Team Collaboration

### For New Team Members

1. Clone the repository
2. Run `./setup.sh`
3. Start coding!

### Project Structure

```
├── recog/              # Main application code
│   ├── main.py         # Entry point
│   ├── centroid_tracker.py  # Tracking algorithm
│   ├── utils.py        # Helper functions
│   └── models/         # Pre-trained models (included)
├── setup.sh            # Automated setup script
├── run.sh              # Convenience run script
├── README.md           # This file
├── TROUBLESHOOTING.md  # Detailed troubleshooting guide
└── CLAUDE.md           # Architecture documentation

```

### Development Tips

- All comments in English for team collaboration
- Virtual environment keeps dependencies isolated
- Models are pre-trained and included (no training needed)
- See `CLAUDE.md` for detailed architecture documentation

### Git Workflow

```bash
# Pull latest changes
git pull

# Create feature branch
git checkout -b feature/your-feature

# Make changes, then commit
git add .
git commit -m "Description of changes"

# Push to remote
git push origin feature/your-feature
```
