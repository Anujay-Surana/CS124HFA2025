# main.py
import os
import cv2
import numpy as np
import json
from centroid_tracker import CentroidTracker
from utils import ensure_person_dir, append_metadata, now_ts, save_person_feature, load_existing_persons

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

def run(source=0):
    """
    Run face detection, recognition, and tracking on video source.

    Args:
        source: Video source (0 for default camera, or path to video file)
    """
    # 1. Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Initialize YuNet face detector
    detector_model_path = os.path.join(SCRIPT_DIR, "models", "face_detection_yunet.onnx")
    detector = cv2.FaceDetectorYN_create(
        detector_model_path,
        "",
        (320, 320),
        score_threshold=0.75,  # Higher to reduce partial face detections
        nms_threshold=0.7,  # Very high to aggressively merge overlapping boxes
        top_k=5000
    )

    # 3. Initialize SFace face recognizer
    recognizer_model_path = os.path.join(SCRIPT_DIR, "models", "face_recognition_sface_2021dec.onnx")
    recognizer = cv2.FaceRecognizerSF_create(
        recognizer_model_path,
        ""
    )

    # 4. Load existing persons from previous sessions
    print("Loading existing persons from output directory...")
    known_persons = load_existing_persons(OUTPUT_DIR)
    if known_persons:
        print(f"Found {len(known_persons)} known person(s): {list(known_persons.keys())}")
    else:
        print("No existing persons found. Starting fresh.")

    # 5. Initialize tracker with feature-based matching
    # max_disappeared: frames before deregistration (increased for better persistence)
    # max_distance: maximum pixel distance for position matching
    # min_feature_similarity: minimum cosine similarity for face feature matching
    ct = CentroidTracker(
        max_disappeared=50,
        max_distance=250,  # Increased to allow more movement
        min_feature_similarity=0.25,  # Very low threshold for flexible matching
        known_persons=known_persons  # Pre-register known persons
    )

    # 6. Open video source
    cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
    frame_idx = 0
    print("Face Recognition System Started")
    print("Press 'q' to quit")
    print("="*50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces using YuNet
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)

        rects = []
        features = []
        face_scores = []  # Store detection scores for quality assessment

        if faces is not None:
            # Additional post-processing: filter overlapping detections
            # Calculate IoU (Intersection over Union) and keep only non-overlapping boxes
            def calculate_iou(box1, box2):
                """Calculate IoU between two boxes (x, y, w, h)"""
                x1, y1, w1, h1 = box1
                x2, y2, w2, h2 = box2

                # Calculate intersection
                xi1 = max(x1, x2)
                yi1 = max(y1, y2)
                xi2 = min(x1 + w1, x2 + w2)
                yi2 = min(y1 + h1, y2 + h2)

                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

                # Calculate union
                box1_area = w1 * h1
                box2_area = w2 * h2
                union_area = box1_area + box2_area - inter_area

                return inter_area / union_area if union_area > 0 else 0

            # Filter faces with high overlap (IoU > 0.3)
            filtered_faces = []
            for i, face in enumerate(faces):
                keep = True
                for j, other_face in enumerate(filtered_faces):
                    box1 = face[:4].astype(int)
                    box2 = other_face[:4].astype(int)
                    iou = calculate_iou(box1, box2)

                    if iou > 0.3:  # High overlap detected
                        # Keep the one with higher detection score (face[14])
                        if len(face) > 14 and len(other_face) > 14:
                            if face[14] <= other_face[14]:  # Lower score, discard
                                keep = False
                                break
                        else:
                            keep = False
                            break

                if keep:
                    filtered_faces.append(face)

            # Process filtered faces
            for face in filtered_faces:
                # Extract bounding box
                x, y, ww, hh = face[:4].astype(int)
                rects.append((x, y, ww, hh))

                # Extract detection score (confidence)
                detection_score = face[14] if len(face) > 14 else 0.5
                face_scores.append(detection_score)

                # Extract face feature using SFace
                try:
                    # Align face for feature extraction
                    aligned_face = recognizer.alignCrop(frame, face)

                    # Extract 128-dim feature vector
                    feature = recognizer.feature(aligned_face)
                    feature = feature.flatten()  # Convert to 1D array
                    features.append(feature)
                except Exception as e:
                    # If feature extraction fails, append None
                    print(f"Feature extraction failed: {e}")
                    features.append(None)

        # Update tracker with both bounding boxes and features
        _, bboxes = ct.update(rects, features if features else None)

        # Draw results and save face crops (only for visible objects)
        visible_count = 0
        for idx, (pid, bbox) in enumerate(bboxes.items()):
            # Only draw and save if the object is visible in current frame
            if not ct.visible.get(pid, False):
                continue

            visible_count += 1
            x, y, w, h = bbox

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw person ID label with similarity score
            similarity = ct.similarities.get(pid, 0.0)
            if similarity > 0:
                label = f"Person {pid} (sim: {similarity:.2f})"
            else:
                label = f"Person {pid} (NEW)"
            cv2.putText(frame, label, (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Save face crops - keep only top 10 best quality images
            if frame_idx % 5 == 0:  # Still sample every 5 frames
                pdir = ensure_person_dir(OUTPUT_DIR, pid)
                crop = frame[y:y+h, x:x+w]

                if crop.size > 0:
                    # Calculate quality score for this image
                    # Score = face_size * detection_confidence * sharpness
                    face_area = w * h

                    # Get detection score for this face
                    # Find which detection this corresponds to
                    detection_idx = 0
                    for i, rect in enumerate(rects):
                        if rect == (x, y, w, h):
                            detection_idx = i
                            break

                    detection_conf = face_scores[detection_idx] if detection_idx < len(face_scores) else 0.5

                    # Calculate sharpness using Laplacian variance
                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray_crop, cv2.CV_64F).var()

                    # Normalize and combine scores
                    # face_area normalized by image size, sharpness normalized to 0-1 range
                    normalized_area = face_area / (w * h * 100)  # Rough normalization
                    normalized_sharpness = min(sharpness / 500.0, 1.0)  # Cap at 500

                    quality_score = (0.4 * normalized_area +
                                   0.3 * detection_conf +
                                   0.3 * normalized_sharpness)

                    # Load existing metadata to check current saved images
                    metadata_path = os.path.join(pdir, "metadata.json")
                    existing_images = []

                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, "r", encoding="utf-8") as f:
                                existing_images = json.load(f)
                        except:
                            existing_images = []

                    # If we have less than 10 images, just add this one
                    if len(existing_images) < 10:
                        img_path = os.path.join(pdir, f"{frame_idx:06d}.jpg")
                        cv2.imwrite(img_path, crop)
                        append_metadata(pdir, {
                            "time": now_ts(),
                            "frame": int(frame_idx),
                            "bbox": [int(x), int(y), int(w), int(h)],
                            "quality_score": float(quality_score)
                        })
                    else:
                        # Find the image with lowest quality score
                        min_score = float('inf')
                        min_idx = -1

                        for i, img_data in enumerate(existing_images):
                            img_score = img_data.get("quality_score", 0)
                            if img_score < min_score:
                                min_score = img_score
                                min_idx = i

                        # If new image is better than worst existing image, replace it
                        if quality_score > min_score:
                            # Delete old image
                            old_frame = existing_images[min_idx]["frame"]
                            old_img_path = os.path.join(pdir, f"{old_frame:06d}.jpg")
                            if os.path.exists(old_img_path):
                                os.remove(old_img_path)

                            # Save new image
                            img_path = os.path.join(pdir, f"{frame_idx:06d}.jpg")
                            cv2.imwrite(img_path, crop)

                            # Update metadata
                            existing_images[min_idx] = {
                                "time": now_ts(),
                                "frame": int(frame_idx),
                                "bbox": [int(x), int(y), int(w), int(h)],
                                "quality_score": float(quality_score)
                            }

                            # Write back metadata
                            with open(metadata_path, "w", encoding="utf-8") as f:
                                json.dump(existing_images, f, ensure_ascii=False, indent=2)

                    # Save the current feature vector for this person
                    if pid in ct.features and ct.features[pid] is not None:
                        save_person_feature(pdir, ct.features[pid])

        # Display frame info
        info_text = f"Frame: {frame_idx} | Tracked: {visible_count} person(s) | Detections: {len(rects)}"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Display debug info for tracked objects
        y_offset = 60
        for pid in sorted(ct.objects.keys()):
            if ct.visible.get(pid, False):
                sim = ct.similarities.get(pid, 0.0)
                disappeared = ct.disappeared.get(pid, 0)
                debug_text = f"P{pid}: sim={sim:.2f}, disappeared={disappeared}"
                cv2.putText(frame, debug_text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += 25

        # Show video
        cv2.imshow("Face Recognition + Tracking", frame)
        frame_idx += 1

        # Handle quit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("="*50)
    print(f"Session ended. Total frames: {frame_idx}")
    print(f"Output saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    # Run with default camera (0)
    # For video file: run("input.mp4")
    run(0)
