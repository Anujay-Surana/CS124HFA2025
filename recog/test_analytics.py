#!/usr/bin/env python3
"""
Test script to verify analytics system is working correctly.
Run this to check if all components are properly installed and functional.
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import cv2
        print("  ✓ OpenCV imported")
    except ImportError as e:
        print(f"  ✗ OpenCV import failed: {e}")
        return False

    try:
        import numpy as np
        print("  ✓ NumPy imported")
    except ImportError as e:
        print(f"  ✗ NumPy import failed: {e}")
        return False

    try:
        from analytics_tracker import AnalyticsTracker
        print("  ✓ AnalyticsTracker imported")
    except ImportError as e:
        print(f"  ✗ AnalyticsTracker import failed: {e}")
        return False

    try:
        from heatmap_visualizer import HeatmapVisualizer
        print("  ✓ HeatmapVisualizer imported")
    except ImportError as e:
        print(f"  ✗ HeatmapVisualizer import failed: {e}")
        return False

    try:
        from age_gender_detector import AgeGenderDetector
        print("  ✓ AgeGenderDetector imported")
    except ImportError as e:
        print(f"  ✗ AgeGenderDetector import failed: {e}")
        return False

    # Optional imports
    try:
        import flask
        print("  ✓ Flask imported (web API available)")
    except ImportError:
        print("  ⚠ Flask not found (web API unavailable - run: pip install flask flask-cors)")

    return True


def test_analytics_tracker():
    """Test analytics tracker functionality."""
    print("\nTesting AnalyticsTracker...")
    try:
        from analytics_tracker import AnalyticsTracker
        import numpy as np

        # Create tracker
        tracker = AnalyticsTracker(640, 480, grid_size=40)
        print("  ✓ AnalyticsTracker created")

        # Test update
        objects = {0: (100, 100), 1: (200, 200)}
        visible = {0: True, 1: True}
        tracker.update(objects, visible)
        print("  ✓ Tracker update works")

        # Test metrics
        dwell_time = tracker.get_person_dwell_time(0)
        print(f"  ✓ Dwell time calculation works (0s expected, got {dwell_time:.1f}s)")

        # Test heatmap generation
        heatmap = tracker.get_movement_heatmap()
        assert heatmap.shape == (12, 16), f"Unexpected heatmap shape: {heatmap.shape}"
        print(f"  ✓ Heat map generation works (shape: {heatmap.shape})")

        # Test summary
        summary = tracker.get_analytics_summary()
        assert summary['unique_persons'] == 2
        print(f"  ✓ Summary generation works ({summary['unique_persons']} persons)")

        return True
    except Exception as e:
        print(f"  ✗ AnalyticsTracker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_heatmap_visualizer():
    """Test heatmap visualizer functionality."""
    print("\nTesting HeatmapVisualizer...")
    try:
        from heatmap_visualizer import HeatmapVisualizer
        import numpy as np

        # Create visualizer
        visualizer = HeatmapVisualizer(alpha=0.4)
        print("  ✓ HeatmapVisualizer created")

        # Create dummy data
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        heatmap = np.random.rand(12, 16)
        print("  ✓ Test data created")

        # Test heatmap overlay
        result = visualizer.overlay_heatmap(frame, heatmap)
        assert result.shape == (480, 640, 3)
        print(f"  ✓ Heat map overlay works (shape: {result.shape})")

        # Test legend creation
        legend = visualizer.create_legend()
        print(f"  ✓ Legend creation works (shape: {legend.shape})")

        # Test trajectory drawing
        trajectory = [(100, 100), (150, 150), (200, 200)]
        result = visualizer.draw_trajectory(frame.copy(), trajectory)
        print("  ✓ Trajectory drawing works")

        return True
    except Exception as e:
        print(f"  ✗ HeatmapVisualizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_age_gender_detector():
    """Test age/gender detector (will pass even if models not available)."""
    print("\nTesting AgeGenderDetector...")
    try:
        from age_gender_detector import AgeGenderDetector, SimpleAgeGenderEstimator
        import numpy as np

        # Try to create detector
        detector = AgeGenderDetector()
        print("  ✓ AgeGenderDetector created")

        if detector.is_available():
            print("  ✓ Age/Gender models are available")
        else:
            print("  ⚠ Age/Gender models not found (optional feature)")
            print("    See ANALYTICS_GUIDE.md for download instructions")

        # Test with dummy data
        dummy_face = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect_age_gender(dummy_face)
        print(f"  ✓ Detection function works")
        print(f"    Result: Age={result['age_range']}, Gender={result['gender']}")

        return True
    except Exception as e:
        print(f"  ✗ AgeGenderDetector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_models():
    """Check if required model files exist."""
    print("\nChecking model files...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")

    required_models = [
        "face_detection_yunet.onnx",
        "face_recognition_sface_2021dec.onnx"
    ]

    optional_models = [
        "age_deploy.prototxt",
        "age_net.caffemodel",
        "gender_deploy.prototxt",
        "gender_net.caffemodel"
    ]

    all_good = True
    for model in required_models:
        path = os.path.join(models_dir, model)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✓ {model} found ({size_mb:.1f}MB)")
        else:
            print(f"  ✗ {model} NOT FOUND (required)")
            all_good = False

    print("\nOptional models:")
    for model in optional_models:
        path = os.path.join(models_dir, model)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✓ {model} found ({size_mb:.1f}MB)")
        else:
            print(f"  ⚠ {model} not found (optional for age/gender)")

    return all_good


def test_web_api():
    """Test if web API can be imported (not started)."""
    print("\nTesting Web API...")
    try:
        from web_api import app, load_analytics_summary
        print("  ✓ Web API module imported successfully")
        print("  ℹ To test API fully, run: python recog/web_api.py")
        return True
    except ImportError as e:
        print(f"  ✗ Web API import failed: {e}")
        print("    Install Flask: pip install flask flask-cors")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Analytics System Test Suite")
    print("="*60)

    results = {
        "Imports": test_imports(),
        "Models": test_models(),
        "AnalyticsTracker": test_analytics_tracker(),
        "HeatmapVisualizer": test_heatmap_visualizer(),
        "AgeGenderDetector": test_age_gender_detector(),
        "WebAPI": test_web_api()
    }

    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s} {status}")

    print("="*60)

    all_critical_passed = all([
        results["Imports"],
        results["Models"],
        results["AnalyticsTracker"],
        results["HeatmapVisualizer"]
    ])

    if all_critical_passed:
        print("\n✓ All critical tests passed!")
        print("\nYou can now run:")
        print("  python recog/main_with_analytics.py")
        if results["WebAPI"]:
            print("  python recog/web_api.py")
        return 0
    else:
        print("\n✗ Some critical tests failed.")
        print("Please fix the issues above before running the system.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
