# main_with_analytics.py
import os
import cv2
import numpy as np
import json
from centroid_tracker import CentroidTracker
from analytics_tracker import AnalyticsTracker
from heatmap_visualizer import HeatmapVisualizer
from utils import ensure_person_dir, append_metadata, now_ts, save_person_feature, load_existing_persons

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

def run_with_analytics(source=0, show_heatmap=True, show_trajectories=True, grid_size=40):
    """
    Run face detection, recognition, and tracking with real-time analytics and heat maps.

    Args:
        source: Video source (0 for default camera, or path to video file)
        show_heatmap: Display heat map overlay
        show_trajectories: Display movement trajectories
        grid_size: Size of grid cells for heat map (pixels)
    """
    # 1. Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    analytics_dir = os.path.join(OUTPUT_DIR, "analytics")
    os.makedirs(analytics_dir, exist_ok=True)

    # 2. Initialize YuNet face detector
    detector_model_path = os.path.join(SCRIPT_DIR, "models", "face_detection_yunet.onnx")
    detector = cv2.FaceDetectorYN_create(
        detector_model_path,
        "",
        (320, 320),
        score_threshold=0.75,
        nms_threshold=0.7,
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

    # 5. Initialize tracker
    ct = CentroidTracker(
        max_disappeared=50,
        max_distance=250,
        min_feature_similarity=0.25,
        known_persons=known_persons
    )

    # 6. Open video source
    cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)

    # Get frame dimensions
    ret, test_frame = cap.read()
    if not ret:
        print("Error: Could not read from video source")
        return

    frame_h, frame_w = test_frame.shape[:2]

    # 7. Initialize analytics tracker
    analytics = AnalyticsTracker(frame_w, frame_h, grid_size=grid_size)

    # 8. Initialize heatmap visualizer
    visualizer = HeatmapVisualizer(alpha=0.4)

    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    print("Face Recognition System with Analytics Started")
    print("Press 'q' to quit")
    print("Press 'h' to toggle heatmap")
    print("Press 't' to toggle trajectories")
    print("Press 's' to save analytics")
    print("=" * 50)

    # Visualization toggles
    heatmap_enabled = show_heatmap
    trajectories_enabled = show_trajectories
    heatmap_mode = "movement"  # "movement" or "density"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Create a copy for visualization
        display_frame = frame.copy()

        # Detect faces using YuNet
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)

        rects = []
        features = []
        face_scores = []

        if faces is not None:
            # Filter overlapping detections (same as original main.py)
            def calculate_iou(box1, box2):
                x1, y1, w1, h1 = box1
                x2, y2, w2, h2 = box2
                xi1 = max(x1, x2)
                yi1 = max(y1, y2)
                xi2 = min(x1 + w1, x2 + w2)
                yi2 = min(y1 + h1, y2 + h2)
                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                box1_area = w1 * h1
                box2_area = w2 * h2
                union_area = box1_area + box2_area - inter_area
                return inter_area / union_area if union_area > 0 else 0

            filtered_faces = []
            for i, face in enumerate(faces):
                keep = True
                for j, other_face in enumerate(filtered_faces):
                    box1 = face[:4].astype(int)
                    box2 = other_face[:4].astype(int)
                    iou = calculate_iou(box1, box2)
                    if iou > 0.3:
                        if len(face) > 14 and len(other_face) > 14:
                            if face[14] <= other_face[14]:
                                keep = False
                                break
                        else:
                            keep = False
                            break
                if keep:
                    filtered_faces.append(face)

            # Process filtered faces
            for face in filtered_faces:
                x, y, ww, hh = face[:4].astype(int)
                rects.append((x, y, ww, hh))
                detection_score = face[14] if len(face) > 14 else 0.5
                face_scores.append(detection_score)

                try:
                    aligned_face = recognizer.alignCrop(frame, face)
                    feature = recognizer.feature(aligned_face)
                    feature = feature.flatten()
                    features.append(feature)
                except Exception as e:
                    print(f"Feature extraction failed: {e}")
                    features.append(None)

        # Update tracker
        objects, bboxes = ct.update(rects, features if features else None)

        # Update analytics
        analytics.update(objects, ct.visible)

        # Draw trajectories if enabled
        if trajectories_enabled:
            for pid in objects.keys():
                if ct.visible.get(pid, False):
                    trajectory = analytics.get_person_trajectory(pid, max_points=30)
                    if len(trajectory) > 1:
                        # Use different colors for different people
                        color_idx = pid % 6
                        colors = [
                            (0, 255, 255),   # Yellow
                            (255, 0, 255),   # Magenta
                            (0, 255, 0),     # Green
                            (255, 128, 0),   # Orange
                            (128, 0, 255),   # Purple
                            (0, 128, 255)    # Light blue
                        ]
                        display_frame = visualizer.draw_trajectory(
                            display_frame, trajectory, color=colors[color_idx], thickness=2
                        )

        # Draw bounding boxes and labels
        visible_count = 0
        for pid, bbox in bboxes.items():
            if not ct.visible.get(pid, False):
                continue

            visible_count += 1
            x, y, w, h = bbox

            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Get analytics for this person
            dwell_time = analytics.get_person_dwell_time(pid)
            similarity = ct.similarities.get(pid, 0.0)

            # Create label with analytics
            if similarity > 0:
                label = f"P{pid} | {dwell_time:.1f}s | sim:{similarity:.2f}"
            else:
                label = f"P{pid} (NEW) | {dwell_time:.1f}s"

            cv2.putText(display_frame, label, (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save face crops (same as original)
            if frame_idx % 5 == 0:
                pdir = ensure_person_dir(OUTPUT_DIR, pid)
                crop = frame[y:y+h, x:x+w]

                if crop.size > 0:
                    # Quality scoring (same as original)
                    face_area = w * h
                    detection_idx = 0
                    for i, rect in enumerate(rects):
                        if rect == (x, y, w, h):
                            detection_idx = i
                            break
                    detection_conf = face_scores[detection_idx] if detection_idx < len(face_scores) else 0.5
                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
                    normalized_area = face_area / (w * h * 100)
                    normalized_sharpness = min(sharpness / 500.0, 1.0)
                    quality_score = (0.4 * normalized_area + 0.3 * detection_conf + 0.3 * normalized_sharpness)

                    # Load existing metadata
                    metadata_path = os.path.join(pdir, "metadata.json")
                    existing_images = []
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, "r", encoding="utf-8") as f:
                                existing_images = json.load(f)
                        except:
                            existing_images = []

                    # Keep top 10 quality images
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
                        min_score = float('inf')
                        min_idx = -1
                        for i, img_data in enumerate(existing_images):
                            img_score = img_data.get("quality_score", 0)
                            if img_score < min_score:
                                min_score = img_score
                                min_idx = i
                        if quality_score > min_score:
                            old_frame = existing_images[min_idx]["frame"]
                            old_img_path = os.path.join(pdir, f"{old_frame:06d}.jpg")
                            if os.path.exists(old_img_path):
                                os.remove(old_img_path)
                            img_path = os.path.join(pdir, f"{frame_idx:06d}.jpg")
                            cv2.imwrite(img_path, crop)
                            existing_images[min_idx] = {
                                "time": now_ts(),
                                "frame": int(frame_idx),
                                "bbox": [int(x), int(y), int(w), int(h)],
                                "quality_score": float(quality_score)
                            }
                            with open(metadata_path, "w", encoding="utf-8") as f:
                                json.dump(existing_images, f, ensure_ascii=False, indent=2)

                    # Save feature
                    if pid in ct.features and ct.features[pid] is not None:
                        save_person_feature(pdir, ct.features[pid])

        # Apply heatmap overlay if enabled
        if heatmap_enabled:
            if heatmap_mode == "movement":
                heatmap = analytics.get_movement_heatmap(normalize=True)
            else:
                heatmap = analytics.get_density_heatmap(normalize=True)

            display_frame = visualizer.overlay_heatmap(
                display_frame, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET
            )
            display_frame = visualizer.add_legend_to_frame(display_frame, "bottom-right")

        # Display frame info
        info_text = f"Frame: {frame_idx} | Tracked: {visible_count} | Total Unique: {analytics.get_analytics_summary()['unique_persons']}"
        cv2.putText(display_frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Display mode info
        mode_text = f"Heatmap: {'ON' if heatmap_enabled else 'OFF'} ({heatmap_mode}) | Trajectories: {'ON' if trajectories_enabled else 'OFF'}"
        cv2.putText(display_frame, mode_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # Display analytics summary (top-right corner)
        y_offset = 90
        for pid in sorted(ct.objects.keys())[:5]:  # Show top 5
            if ct.visible.get(pid, False):
                dwell = analytics.get_person_dwell_time(pid)
                visits = analytics.visit_count[pid]
                sessions = analytics.session_count[pid]
                stats_text = f"P{pid}: {dwell:.1f}s | V:{visits} | S:{sessions}"
                cv2.putText(display_frame, stats_text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += 25

        # Show video
        cv2.imshow("Face Recognition + Analytics", display_frame)
        frame_idx += 1

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            heatmap_enabled = not heatmap_enabled
            print(f"Heatmap: {'ON' if heatmap_enabled else 'OFF'}")
        elif key == ord('t'):
            trajectories_enabled = not trajectories_enabled
            print(f"Trajectories: {'ON' if trajectories_enabled else 'OFF'}")
        elif key == ord('m'):
            heatmap_mode = "density" if heatmap_mode == "movement" else "movement"
            print(f"Heatmap mode: {heatmap_mode}")
        elif key == ord('s'):
            print("Saving analytics...")
            analytics.export_heatmaps(analytics_dir)
            print(f"Analytics saved to {analytics_dir}/")

    # Save final analytics
    print("\nSaving final analytics...")
    analytics.export_heatmaps(analytics_dir)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    print("=" * 50)
    print(f"Session ended. Total frames: {frame_idx}")
    print(f"Output saved to: {OUTPUT_DIR}/")
    print(f"Analytics saved to: {analytics_dir}/")

    # Print analytics summary
    summary = analytics.get_analytics_summary()
    print("\nAnalytics Summary:")
    print(f"  Unique persons detected: {summary['unique_persons']}")
    for person in summary['persons']:
        print(f"\n  Person {person['id']}:")
        print(f"    First seen: {person['first_seen']}")
        print(f"    Last seen: {person['last_seen']}")
        print(f"    Dwell time: {person['dwell_time_seconds']:.1f}s")
        print(f"    Visits: {person['visit_count']}")
        print(f"    Sessions: {person['session_count']}")

if __name__ == "__main__":
    # Run with default camera
    # Options: show_heatmap, show_trajectories, grid_size
    run_with_analytics(0, show_heatmap=True, show_trajectories=True, grid_size=40)
