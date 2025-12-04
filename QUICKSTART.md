# Quick Start Guide for Team Members

Welcome! This guide will help you get the Face Recognition system running in 5 minutes.

## Prerequisites

You need:
- Python 3.8 or higher ([download here](https://www.python.org/downloads/))
- A webcam
- macOS, Windows, or Linux

## Setup (First Time Only)

### Step 1: Get the Code

```bash
# Clone the repository (replace with actual URL)
git clone <repository-url>
cd CS124HFA2025
```

### Step 2: Run Automated Setup

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r recog\requirements.txt
```

Wait 2-3 minutes for installation to complete.

### Step 3: Verify Installation

```bash
./run.sh verify
```

You should see "Setup verification complete!" with green checkmarks.

## Running the System

### Every Time You Want to Run:

**macOS/Linux:**
```bash
source .venv/bin/activate  # Activate virtual environment
./run.sh                   # Run the system
```

**Windows:**
```powershell
.venv\Scripts\activate     # Activate virtual environment
cd recog
python main.py             # Run the system
```

### What to Expect:

1. A window will open showing your webcam feed
2. When faces are detected, you'll see:
   - Green rectangles around faces
   - Person IDs (person_000, person_001, etc.)
3. Face crops are automatically saved to `recog/output/person_XXX/`
4. Press **'q'** to quit

## Common Issues

### "python3: command not found"
Install Python from https://www.python.org/downloads/

### "Virtual environment not found"
Run `./setup.sh` first

### "Module not found"
Make sure you activated the virtual environment:
```bash
source .venv/bin/activate  # You should see (.venv) in your prompt
```

### "Camera not opening"
- Check that no other app is using your camera
- Grant camera permissions to Terminal/PowerShell
- Try different camera: edit `main.py` line 147: `run(source=1)`

### Still Having Issues?
See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions.

## Project Structure

```
recog/
├── main.py                 # Main detection loop (start here!)
├── centroid_tracker.py     # Tracking algorithm
├── utils.py                # Helper functions
├── models/                 # Pre-trained models (included)
└── output/                 # Your face detection results go here
    ├── person_000/
    ├── person_001/
    └── ...
```

## Making Changes

1. **Always activate virtual environment first:**
   ```bash
   source .venv/bin/activate
   ```

2. **Make your changes** to the code

3. **Test your changes:**
   ```bash
   ./run.sh
   ```

4. **Commit and push:**
   ```bash
   git add .
   git commit -m "Description of your changes"
   git push
   ```

## Tips

- **First run:** May take a few seconds to initialize models
- **Performance:** Close other heavy applications for best FPS
- **Testing:** Point camera at multiple faces to see tracking in action
- **Output:** Check `recog/output/` folder for saved face crops

## Getting Help

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Read [README.md](README.md) for full documentation
3. Read [CLAUDE.md](CLAUDE.md) for architecture details
4. Ask the team!

## Next Steps

Once you're running successfully:

1. Read through `main.py` to understand the flow
2. Explore `centroid_tracker.py` to see the tracking algorithm
3. Try adjusting tracking parameters (see README.md)
4. Experiment with the code!

Happy coding!
