#!/usr/bin/env python3
# list_cameras.py - Find available camera devices

import cv2

def list_cameras(max_tested=10):
    """
    Test camera indices to find which ones are available.
    """
    print("Scanning for available cameras...\n")
    available_cameras = []

    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                backend = cap.getBackendName()

                print(f"✓ Camera {i}:")
                print(f"    Resolution: {width}x{height}")
                print(f"    FPS: {fps}")
                print(f"    Backend: {backend}")
                print()

                available_cameras.append(i)
            cap.release()

    if not available_cameras:
        print("No cameras found!")
    else:
        print(f"\nFound {len(available_cameras)} camera(s): {available_cameras}")
        print(f"\nTo use a specific camera, run:")
        print(f"  python main.py --source <camera_index>")
        print(f"\nFor example:")
        for cam_idx in available_cameras:
            print(f"  python main.py --source {cam_idx}")

    return available_cameras

if __name__ == "__main__":
    list_cameras()
