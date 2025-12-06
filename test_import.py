#!/usr/bin/env python3
"""Quick test to verify all modules load correctly without camera"""

print("Testing imports...")

try:
    import cv2
    print("✓ OpenCV imported")

    from recog.analytics_tracker import AnalyticsTracker
    print("✓ AnalyticsTracker imported")

    from recog.heatmap_visualizer import HeatmapVisualizer
    print("✓ HeatmapVisualizer imported")

    from recog.centroid_tracker import CentroidTracker
    print("✓ CentroidTracker imported")

    from recog.utils import ensure_person_dir, load_existing_persons
    print("✓ Utils imported")

    print("\nTesting model loading...")

    # Test YuNet detector
    detector = cv2.FaceDetectorYN_create(
        "recog/models/face_detection_yunet.onnx",
        "",
        (320, 320)
    )
    print("✓ YuNet face detector loaded")

    # Test SFace recognizer
    recognizer = cv2.FaceRecognizerSF_create(
        "recog/models/face_recognition_sface_2021dec.onnx",
        ""
    )
    print("✓ SFace face recognizer loaded")

    print("\nTesting analytics components...")

    # Test analytics tracker
    analytics = AnalyticsTracker(640, 480, grid_size=40)
    print("✓ AnalyticsTracker created")

    # Test visualizer
    visualizer = HeatmapVisualizer(alpha=0.4)
    print("✓ HeatmapVisualizer created")

    print("\n" + "="*50)
    print("✅ All components loaded successfully!")
    print("="*50)
    print("\nYou can now run:")
    print("  python3 recog/main_with_analytics.py")
    print("\n(Press 'q' to quit when the camera window opens)")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
