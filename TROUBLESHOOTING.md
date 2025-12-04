# Troubleshooting Guide

Common issues and solutions for the Face Recognition and Tracking System.

## Setup Issues

### 1. Python Not Found

**Error:**
```
bash: python3: command not found
```

**Solution:**
- **macOS:** Install from [python.org](https://www.python.org/downloads/) or use Homebrew:
  ```bash
  brew install python3
  ```
- **Windows:** Download installer from [python.org](https://www.python.org/downloads/)
  - Make sure to check "Add Python to PATH" during installation
- **Linux (Ubuntu/Debian):**
  ```bash
  sudo apt update
  sudo apt install python3 python3-pip python3-venv
  ```

### 2. Virtual Environment Creation Fails

**Error:**
```
Error: No module named venv
```

**Solution:**
Install the venv module:
```bash
# Ubuntu/Debian
sudo apt install python3-venv

# macOS (usually included, but if needed)
python3 -m pip install virtualenv
```

### 3. pip Install Fails

**Error:**
```
ERROR: Could not install packages due to an OSError
```

**Solutions:**

**Option 1 - Use sudo (not recommended):**
```bash
sudo pip install -r recog/requirements.txt
```

**Option 2 - User install (better):**
```bash
pip install --user -r recog/requirements.txt
```

**Option 3 - Fix virtual environment permissions:**
```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r recog/requirements.txt
```

## Runtime Issues

### 4. YuNet Interface Not Available

**Error:**
```
AttributeError: module 'cv2' has no attribute 'FaceDetectorYN_create'
```

**Solution:**
You need `opencv-contrib-python`, not just `opencv-python`:

```bash
source .venv/bin/activate
pip uninstall opencv-python opencv-contrib-python -y
pip install opencv-contrib-python>=4.8.0
```

**Verify fix:**
```bash
python recog/verify_setup.py
```

### 5. Camera Not Opening

**Error:**
```
Error: Failed to open camera
```

**Solutions:**

**macOS:**
1. Check camera permissions:
   - System Preferences → Security & Privacy → Camera
   - Allow Terminal/your IDE to access camera

2. Try different camera index in `main.py`:
   ```python
   run(source=1)  # or 2, 3, etc.
   ```

**Linux:**
1. Check camera permissions:
   ```bash
   ls -l /dev/video*
   sudo usermod -a -G video $USER
   # Log out and back in
   ```

2. Install v4l-utils:
   ```bash
   sudo apt install v4l-utils
   v4l2-ctl --list-devices
   ```

**Windows:**
1. Check camera isn't being used by another application
2. Check Windows privacy settings for camera access

### 6. Module Not Found Errors

**Error:**
```
ModuleNotFoundError: No module named 'cv2'
ModuleNotFoundError: No module named 'imutils'
```

**Solution:**
Make sure virtual environment is activated:
```bash
source .venv/bin/activate  # You should see (.venv) in your prompt
pip install -r recog/requirements.txt
```

**Check what's installed:**
```bash
pip list | grep opencv
pip list | grep numpy
```

### 7. Model Files Not Found

**Error:**
```
Error: face_detection_yunet.onnx not found
Error: face_recognition_sface_2021dec.onnx not found
```

**Solution:**
Model files should be in `recog/models/`. Check if they exist:
```bash
ls -lh recog/models/
```

Expected output:
```
face_detection_yunet.onnx (227KB)
face_recognition_sface_2021dec.onnx (37MB)
```

If missing, pull from git:
```bash
git pull
git lfs pull  # If using Git LFS
```

### 8. Performance Issues (Slow/Laggy)

**Symptoms:**
- Low FPS
- Delayed video
- System freezing

**Solutions:**

1. **Reduce resolution:**
   Edit `main.py` and add after line where cap is created:
   ```python
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   ```

2. **Increase save interval:**
   Change `main.py` line ~140:
   ```python
   if frame_count % 10 == 0:  # Save every 10 frames instead of 5
   ```

3. **Check system resources:**
   ```bash
   # macOS
   top

   # Linux
   htop
   ```
   - Ensure you have at least 2GB free RAM
   - Close other resource-intensive applications

### 9. Permission Denied on Scripts

**Error:**
```
bash: ./setup.sh: Permission denied
bash: ./run.sh: Permission denied
```

**Solution:**
```bash
chmod +x setup.sh run.sh
```

### 10. Import Error on Different OS

**Error (Windows):**
```
ImportError: DLL load failed
```

**Solution:**
Install Visual C++ Redistributable:
- Download from [Microsoft](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
- Install and restart

**Error (Linux):**
```
ImportError: libGL.so.1: cannot open shared object file
```

**Solution:**
```bash
sudo apt install libgl1-mesa-glx libglib2.0-0
```

## Verification

After fixing any issue, verify your setup:
```bash
source .venv/bin/activate
python recog/verify_setup.py
```

## Getting Help

If your issue isn't listed here:

1. **Check Python and package versions:**
   ```bash
   python3 --version
   pip list
   ```

2. **Run with verbose output:**
   ```bash
   python recog/main.py 2>&1 | tee debug.log
   ```

3. **Search error message:**
   - Google the exact error message
   - Check OpenCV documentation
   - Check project GitHub issues

4. **Contact team:**
   - Share your `debug.log` file
   - Include your OS and Python version
   - Describe what you were trying to do

## Platform-Specific Notes

### macOS
- Requires macOS 10.13 or later
- Camera permissions required (System Preferences)
- Works best with native Terminal app

### Windows
- Requires Windows 10 or later
- May need Visual C++ Redistributable
- Use PowerShell or Git Bash

### Linux
- Tested on Ubuntu 20.04+
- May need additional system packages
- Camera permissions via usermod

## Clean Reinstall

If all else fails, start fresh:

```bash
# 1. Deactivate virtual environment
deactivate

# 2. Remove virtual environment
rm -rf .venv venv

# 3. Run setup again
./setup.sh
```
