# main.py

# Configurations
# Live webcam
# python main.py

# Webcam + record
## python main.py --record

# Process video file
# python main.py --source videos/myvideo.mp4

# Full demo: process video, record output, and generate person clips
# python main.py --source input.mp4 --record --make-person-videos --person-fps 12
#

import os
import cv2
import numpy as np
import json
import argparse
from centroid_tracker import CentroidTracker
from utils import ensure_person_dir, append_metadata, now_ts, save_person_feature, load_existing_persons

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
VIDEO_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "videos")
PERSON_VIDEOS_DIR = os.path.join(VIDEO_OUTPUT_DIR, "person_clips")


def create_person_videos(fps=10):
    """Create a video clip for each person using their saved high-quality face crops."""
    os.makedirs(PERSON_VIDEOS_DIR, exist_ok=True)
    print(f"\nCreating per-person highlight videos → {PERSON_VIDEOS_DIR}")

    for person_folder in os.listdir(OUTPUT_DIR):
        if not person_folder.startswith("person_"):
            continue

        person_dir = os.path.join(OUTPUT_DIR, person_folder)
        image_files = [f for f in os.listdir(person_dir) if f.endswith(".jpg")]
        if not image_files:
            continue

        # Read and sort images by frame number
        images = []
        for img_file in sorted(image_files):
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)

        if not images:
            continue

        h, w = images[0].shape[:2]
        video_path = os.path.join(PERSON_VIDEOS_DIR, f"{person_folder}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

        print(f"  → {person_folder}.mp4 ({len(images)} frames @ {fps}fps)")

        for img in images:
            out.write(img)

        out.release()
        print(f"     Saved: {video_path}")

    print("All person highlight videos created!")


def run(source=0, save_video=False, output_video_name=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

    # Video recording setup
    video_writer = None
    if save_video:
        name = output_video_name or f"session_{now_ts().replace(':', '-').replace(' ', '_')}.mp4"
        video_path = os.path.join(VIDEO_OUTPUT_DIR, name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        print(f"Recording session → {video_path}")

    # === Load Models ===
    detector_model_path = os.path.join(SCRIPT_DIR, "models", "face_detection_yunet.onnx")
    detector = cv2.FaceDetectorYN_create(
        detector_model_path, "", (320, 320), score_threshold=0.75, nms_threshold=0.7, top_k=5000
    )

    recognizer_model_path = os.path.join(SCRIPT_DIR, "models", "face_recognition_sface_2021dec.onnx")
    recognizer = cv2.FaceRecognizerSF_create(recognizer_model_path, "")

    # === Load Known Persons ===
    print("Loading existing persons from output directory...")
    known_persons = load_existing_persons(OUTPUT_DIR)
    if known_persons:
        print(f"Found {len(known_persons)} known person(s): {list(known_persons.keys())}")
    else:
        print("No existing persons found. Starting fresh.")

    ct = CentroidTracker(
        max_disappeared=50,
        max_distance=250,
        min_feature_similarity=0.25,
        known_persons=known_persons
    )

    # === Open Video Source ===
    if isinstance(source, str) and os.path.isfile(source):
        print(f"Opening video file: {source}")
    else:
        source = 0
        print("Opening default webcam (source=0)")

    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Cannot open video source!")
        return

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0:
        fps_in = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if save_video:
        video_writer = cv2.VideoWriter(video_path, fourcc, fps_in, (width, height))

    frame_idx = 0
    print("Face Recognition System Started")
    print("Press 'q' to quit")
    print("=" * 70)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)

        rects = []
        features = []
        face_scores = []

        if faces is not None:
            # === Post-process: filter overlapping detections ===
            def calculate_iou(box1, box2):
                x1, y1, w1, h1 = box1
                x2, y2, w2, h2 = box2
                xi1 = max(x1, x2)
                yi1 = max(y1, y2)
                xi2 = min(x1 + w1, x2 + w2)
                yi2 = min(y1 + h1, y2 + h2)
                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                union_area = w1 * h1 + w2 * h2 - inter_area
                return inter_area / union_area if union_area > 0 else 0

            filtered_faces = []
            for face in faces:
                keep = True
                box1 = face[:4].astype(int)
                for other in filtered_faces:
                    box2 = other[:4].astype(int)
                    if calculate_iou(box1, box2) > 0.3:
                        if len(face) > 14 and len(other) > 14 and face[14] <= other[14]:
                            keep = False
                            break
                        elif len(face) <= 14:
                            keep = False
                            break
                if keep:
                    filtered_faces.append(face)

            # === Process each face ===
            for face in filtered_faces:
                x, y, ww, hh = face[:4].astype(int)
                rects.append((x, y, ww, hh))
                detection_score = face[14] if len(face) > 14 else 0.5
                face_scores.append(detection_score)

                try:
                    aligned = recognizer.alignCrop(frame, face)
                    feature = recognizer.feature(aligned).flatten()
                    features.append(feature)
                except Exception as e:
                    print(f"Feature extraction failed: {e}")
                    features.append(None)

        # === Update tracker ===
        _, bboxes = ct.update(rects, features if features else None)

        # === Draw & Save ===
        visible_count = 0
        for pid, bbox in bboxes.items():
            if not ct.visible.get(pid, False):
                continue
            visible_count += 1
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            sim = ct.similarities.get(pid, 0.0)
            label = f"Person {pid} (sim: {sim:.2f})" if sim > 0 else f"Person {pid} (NEW)"
            cv2.putText(frame, label, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Save best face crops (every 5 frames)
            if frame_idx % 5 == 0:
                pdir = ensure_person_dir(OUTPUT_DIR, pid)
                crop = frame[y:y+h, x:x+w]
                if crop.size == 0:
                    continue

                # Quality score
                area = w * h
                conf = face_scores[[i for i, r in enumerate(rects) if r == (x,y,w,h)][0]] if any(r == (x,y,w,h) for r in rects) else 0.5
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                quality = 0.4 * (area / (w * h * 100)) + 0.3 * conf + 0.3 * min(sharpness / 500.0, 1.0)

                meta_path = os.path.join(pdir, "metadata.json")
                existing = []
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            existing = json.load(f)
                    except:
                        existing = []

                if len(existing) < 10:
                    img_path = os.path.join(pdir, f"{frame_idx:06d}.jpg")
                    cv2.imwrite(img_path, crop)
                    append_metadata(pdir, {"time": now_ts(), "frame": frame_idx, "bbox": [x,y,w,h], "quality_score": quality})
                else:
                    worst_idx = min(range(len(existing)), key=lambda i: existing[i].get("quality_score", 0))
                    if quality > existing[worst_idx].get("quality_score", 0):
                        old_frame = existing[worst_idx]["frame"]
                        old_path = os.path.join(pdir, f"{old_frame:06d}.jpg")
                        if os.path.exists(old_path):
                            os.remove(old_path)
                        new_path = os.path.join(pdir, f"{frame_idx:06d}.jpg")
                        cv2.imwrite(new_path, crop)
                        existing[worst_idx] = {"time": now_ts(), "frame": frame_idx, "bbox": [x,y,w,h], "quality_score": quality}
                        with open(meta_path, "w", encoding="utf-8") as f:
                            json.dump(existing, f, indent=2, ensure_ascii=False)

                if pid in ct.features and ct.features[pid] is not None:
                    save_person_feature(pdir, ct.features[pid])

        # === Info Overlay ===
        info = f"Frame: {frame_idx} | Tracked: {visible_count} | Detections: {len(rects)}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        y_off = 60
        for pid in sorted(ct.objects.keys()):
            if ct.visible.get(pid, False):
                sim = ct.similarities.get(pid, 0.0)
                disp = ct.disappeared.get(pid, 0)
                cv2.putText(frame, f"P{pid}: sim={sim:.2f}, gone={disp}", (10, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_off += 25

        # === Record & Show ===
        if video_writer:
            video_writer.write(frame)
        cv2.imshow("Face Recognition + Tracking", frame)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # === Cleanup ===
    cap.release()
    if video_writer:
        video_writer.release()
        print(f"Session video saved → {video_path}")
    cv2.destroyAllWindows()
    print("=" * 70)
    print(f"Session ended. Processed {frame_idx} frames.")
    print(f"Data saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition & Tracking System")
    parser.add_argument("--source", type=str, default=0,
                        help="0 = webcam, or path to video file (e.g. videos/test.mp4)")
    parser.add_argument("--record", action="store_true", help="Record output video")
    parser.add_argument("--output", type=str, default=None, help="Custom output video name")
    parser.add_argument("--make-person-videos", action="store_true",
                        help="Generate a video for each person from their saved face crops")
    parser.add_argument("--person-fps", type=int, default=10, help="FPS for person highlight videos")

    args = parser.parse_args()

    try:
        source = int(args.source)
    except:
        source = args.source

    run(source=source, save_video=args.record, output_video_name=args.output)

    if args.make_person_videos:
        create_person_videos(fps=args.person_fps)