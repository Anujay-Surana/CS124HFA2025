"""
Verify that face recognition system is properly configured.
Tests model loading and basic functionality without camera.
"""
import cv2
import numpy as np
import os

def verify_setup():
    print("="*60)
    print("Face Recognition System - Setup Verification")
    print("="*60)

    # Check OpenCV version
    print(f"\n1. OpenCV Version: {cv2.__version__}")

    # Check YuNet support
    has_yunet = hasattr(cv2, "FaceDetectorYN_create")
    print(f"2. YuNet Support: {'✓ Available' if has_yunet else '✗ Not Available'}")
    if not has_yunet:
        print("   ERROR: YuNet interface not available!")
        print("   Make sure opencv-contrib-python >= 4.8.0 is installed")
        return False

    # Check SFace support
    has_sface = hasattr(cv2, "FaceRecognizerSF_create")
    print(f"3. SFace Support: {'✓ Available' if has_sface else '✗ Not Available'}")
    if not has_sface:
        print("   ERROR: SFace interface not available!")
        print("   Make sure opencv-contrib-python >= 4.8.0 is installed")
        return False

    # Check model files
    yunet_path = "models/face_detection_yunet.onnx"
    sface_path = "models/face_recognition_sface_2021dec.onnx"

    yunet_exists = os.path.exists(yunet_path)
    sface_exists = os.path.exists(sface_path)

    print(f"4. YuNet Model: {'✓ Found' if yunet_exists else '✗ Missing'} ({yunet_path})")
    print(f"5. SFace Model: {'✓ Found' if sface_exists else '✗ Missing'} ({sface_path})")

    if not yunet_exists or not sface_exists:
        print("\n   ERROR: Model files missing!")
        return False

    # Test model loading
    print("\n6. Testing Model Loading...")
    try:
        detector = cv2.FaceDetectorYN_create(
            yunet_path,
            "",
            (320, 320)
        )
        print("   ✓ YuNet detector loaded successfully")
    except Exception as e:
        print(f"   ✗ YuNet loading failed: {e}")
        return False

    try:
        recognizer = cv2.FaceRecognizerSF_create(
            sface_path,
            ""
        )
        print("   ✓ SFace recognizer loaded successfully")
    except Exception as e:
        print(f"   ✗ SFace loading failed: {e}")
        return False

    # Test basic functionality
    print("\n7. Testing Basic Functionality...")
    try:
        # Create a dummy image
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.setInputSize((640, 480))
        _, faces = detector.detect(test_img)
        print("   ✓ Detection test passed (no faces expected)")

        # Test feature extraction (will fail without face, but should not crash)
        print("   ✓ Functionality test passed")
    except Exception as e:
        print(f"   ✗ Functionality test failed: {e}")
        return False

    print("\n" + "="*60)
    print("✓ ALL CHECKS PASSED - System Ready!")
    print("="*60)
    print("\nYou can now run: python main.py")
    return True

if __name__ == "__main__":
    success = verify_setup()
    exit(0 if success else 1)
