# main.py
import os
import cv2
import numpy as np
from centroid_tracker import CentroidTracker
from utils import ensure_person_dir, append_metadata, now_ts

OUTPUT_DIR = "output"

def run(source=0):
    # 1. Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Initialize YuNet detector
    detector = cv2.FaceDetectorYN_create(
        "models/face_detection_yunet.onnx",  # Path to YuNet model
        "",
        (320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000
    )

    # 3. Initialize tracker
    ct = CentroidTracker(max_disappeared=20)

    # 4. Open video source (0 = default camera)
    cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
    frame_idx = 0
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)

        rects = []
        if faces is not None:
            for f in faces:
                x, y, ww, hh = f[:4].astype(int)
                rects.append((x, y, ww, hh))

        # Update tracker
        objects, bboxes = ct.update(rects)

        # Draw results and save face crops
        for pid, bbox in bboxes.items():
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {pid}", (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Save every few frames
            if frame_idx % 5 == 0:
                pdir = ensure_person_dir(OUTPUT_DIR, pid)
                img_path = os.path.join(pdir, f"{frame_idx:06d}.jpg")
                crop = frame[y:y+h, x:x+w]
                if crop.size > 0:
                    cv2.imwrite(img_path, crop)
                    append_metadata(pdir, {
                        "time": now_ts(),
                        "frame": int(frame_idx),
                        "bbox": [int(x), int(y), int(w), int(h)]
                    })

        cv2.imshow("YuNet Face Detection + Tracking", frame)
        frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # You can use run(0) for camera or run("input.mp4") for video file
    run(0)