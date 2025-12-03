import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from collections import defaultdict, deque
import os
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
import sys
import math
import time
import csv
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)


INPUT_SOURCE = r"C:\Users\Ende\Downloads\815657\P1E_S1\P1E_S1_C1\P1E_S1_C1"
OUTPUT_VIDEO = "store_entry_final.mp4"
CSV_FILENAME = "demographics_log.csv"

CONFIDENCE_THRESHOLD = 0.45
SKIP_FRAMES = 1
PAD_WIDE = 0.60
PAD_TIGHT = 0.10

SCORE_TO_LOCK = 3.0
CONSENSUS_RATIO = 0.60
FORCED_LOCK_FRAMES = 12

MIN_QUALITY_THRESHOLD = 0.30
MIN_FACE_CONFIDENCE = 0.60
MIN_FACE_SIZE = 50
MAX_FACE_ANGLE = 35.0

CALIBRATE_MALE = 0.80  
CALIBRATE_FEMALE = 1.20 

ENABLE_PREPROCESSING = True
CLAHE_CLIP_LIMIT = 2.0
GAUSSIAN_KERNEL = 3

TEMPORAL_SMOOTHING_WINDOW = 15
MAX_AGE_LIST_SIZE = 30

ADAPTIVE_SKIP_ENABLED = True
MIN_SKIP_FRAMES = 0
MAX_SKIP_FRAMES = 3

COLOR_MALE = (255, 0, 0)
COLOR_FEMALE = (203, 192, 255)
COLOR_ANALYZING = (0, 255, 255)
COLOR_UNTRACKED = (100, 100, 100)
COLOR_QUALITY_INDICATOR = (0, 255, 0)

# Dataclasses
@dataclass
class DemographicData:
    gender: str
    race: str
    age_bucket: str

@dataclass
class Accumulator:
    gender_history: List[str] = field(default_factory=list)
    race_history: List[str] = field(default_factory=list)
    gender_score: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    race_score: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    age_list: List[int] = field(default_factory=list)
    age_quality_list: List[float] = field(default_factory=list)
    last_quality: float = 0.0
    frame_count: int = 0
    ema_gender_prob: Dict[str, float] = field(default_factory=lambda: {'Male': 0.0, 'Female': 0.0})
    ema_quality: float = 0.0
    recent_genders: deque = field(default_factory=lambda: deque(maxlen=TEMPORAL_SMOOTHING_WINDOW))

def ensure_video_source(source: str, output_name: str) -> Optional[str]:
    if os.path.isfile(source): return source
    if not os.path.exists(source): return None
    images = sorted([img for img in os.listdir(source) if img.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not images: return None
    if os.path.exists(output_name): return output_name
    print(f"Converting {len(images)} images to video...")
    first = cv2.imread(os.path.join(source, images[0]))
    h, w = first.shape[:2]
    out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))
    frames_to_process = min(len(images), 2500)
    for i, img_name in enumerate(images[:frames_to_process]):
        img = cv2.imread(os.path.join(source, img_name))
        if img is not None: out.write(img)
    out.release()
    return output_name

def load_models() -> Tuple[YOLO, cv2.dnn_Net]:
    print("Loading models...")
    try:
        yolo_model = YOLO('yolov8s.pt')
        print("  ✓ YOLO model loaded (Small version)")
    except Exception as e:
        print(f"Error loading YOLO: {e}")
        sys.exit(1)
    try:
        faceNet = cv2.dnn.readNet("opencv_setup/res10_300x300_ssd_iter_140000.caffemodel", "opencv_setup/deploy.prototxt")
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("  ✓ Face detection using CUDA")
        else:
            print("  ✓ Face detection using CPU")
    except Exception as e:
        print(f"Error loading face model: {e}")
        sys.exit(1)
    return yolo_model, faceNet

yolo_model, faceNet = load_models()

# Face detection: box
def get_face_box(img_crop: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
    if img_crop is None or img_crop.size == 0: return 0.0, None
    h, w = img_crop.shape[:2]
    blob = cv2.dnn.blobFromImage(img_crop, 1.0, (300, 300), [104, 117, 123], False, False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    best_conf = 0.0
    best_box = None
    try:
        dets = detections[0, 0, :, :]
        for i in range(dets.shape[0]):
            conf = float(dets[i, 2])
            if conf > best_conf:
                box = dets[i, 3:7] * np.array([w, h, w, h])
                box = box.astype(int)
                best_conf = conf
                best_box = box
    except Exception:
        try:
            flat = detections.flatten()
            return 0.0, None
        except Exception:
            return 0.0, None

    if best_conf > MIN_FACE_CONFIDENCE and best_box is not None:
        return best_conf, best_box
    return 0.0, None


def preprocess_face(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0 or not ENABLE_PREPROCESSING: return img
    h, w = img.shape[:2]
    if h < 112:
        scale = 112 / h
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    processed = img.copy()
    lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
    l = clahe.apply(l)
    processed = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    if GAUSSIAN_KERNEL > 0:
        processed = cv2.GaussianBlur(processed, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
    return processed

# quality scoring
def calculate_quality_score(face_img: np.ndarray) -> float:
    if face_img is None or face_img.size == 0: return 0.0

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    w = gray.shape[1]
    mid = w // 2
    left_var = np.var(gray[:, :mid]) if mid > 0 else 0.0
    right_var = np.var(gray[:, mid:]) if mid > 0 else 0.0
    diff = abs(left_var - right_var) / max(left_var + right_var, 1.0)
    angle = min(diff * 80.0, 90.0)

    if angle > MAX_FACE_ANGLE: return 0.0

    h, w = face_img.shape[:2]
    blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
    avg_brightness = np.mean(gray)

    size_score = min(h / 140.0, 1.0)
    blur_score = min(blur_val / 180.0, 1.0)
    bright_score = 0.2 if (avg_brightness < 40 or avg_brightness > 220) else 1.0
    angle_score = 1.0 - (angle / MAX_FACE_ANGLE)

    return (size_score * 0.3 + blur_score * 0.4 + bright_score * 0.1 + angle_score * 0.2)

# demographic analysis

def parse_deepface_gender_result(gender_res) -> Tuple[float, float]:
    """
    Return (male_prob, female_prob) in [0,1].
    Accepts DeepFace.analyze outputs in different return formats.
    """
    male_p = female_p = 0.0
    try:
        # DeepFace sometimes returns a list of dicts, sometimes a dict
        entry = gender_res[0] if isinstance(gender_res, list) and len(gender_res) > 0 else gender_res
        g = entry.get('gender', None)
        if isinstance(g, dict):
            # Keys might be 'Man'/'Woman' or 'man'/'woman'
            keys = {k.lower(): k for k in g.keys()}
            if 'man' in keys and 'woman' in keys:
                male_p = g[keys['man']] / 100.0
                female_p = g[keys['woman']] / 100.0
            else:
                # sometimes DeepFace returns string label and 'gender_val' or similar
                pass
        else:
            # some versions return 'gender' string and 'gender_confidence' maybe
            label = str(entry.get('gender', '')).lower()
            if label in ['man', 'male']:
                male_p = float(entry.get('gender_confidence', 0.0)) / 100.0 if entry.get('gender_confidence') else 0.9
                female_p = 1.0 - male_p
            elif label in ['woman', 'female']:
                female_p = float(entry.get('gender_confidence', 0.0)) / 100.0 if entry.get('gender_confidence') else 0.9
                male_p = 1.0 - female_p
    except Exception:
        pass
    s = max(male_p + female_p, 1e-6)
    male_p /= s
    female_p /= s
    return male_p, female_p

def analyze_demographics_unified(img_crop_wide: np.ndarray, img_crop_tight: np.ndarray) -> Optional[Dict]:
    """
    Uses WIDE crop for Gender (context) and TIGHT crop for Age/Race (features).
    Returns probabilities instead of forcing a label decision here.
    """
    if img_crop_wide is None or img_crop_wide.size == 0: return None

    try:
        # GENDER: wide crop raw image (no preprocessing)
        gender_res = DeepFace.analyze(
            img_crop_wide,
            actions=['gender'],
            detector_backend='opencv',
            enforce_detection=False,
            align=True,
            silent=True
        )
        male_p, female_p = parse_deepface_gender_result(gender_res)

        # calibration multipliers (keeps values in 0-1 after renormalizing)
        male_p *= CALIBRATE_MALE
        female_p *= CALIBRATE_FEMALE
        denom = max(male_p + female_p, 1e-6)
        male_p /= denom
        female_p /= denom

        # RACE & AGE on TIGHT
        processed_tight = preprocess_face(img_crop_tight)
        other_res = DeepFace.analyze(
            processed_tight,
            actions=['age', 'race'],
            detector_backend='opencv',
            enforce_detection=False,
            align=True,
            silent=True
        )
        entry = other_res[0] if isinstance(other_res, list) and len(other_res) > 0 else other_res
        race = entry.get('dominant_race', entry.get('race', 'Unknown'))
        age = int(entry.get('age', 0) or 0)

        return {
            'male_prob': male_p,
            'female_prob': female_p,
            'race': race,
            'age_raw': age,
            'frame_confidence': max(male_p, female_p)
        }

    except Exception as e:
        return None


def apply_temporal_smoothing(age_list: List[int], quality_list: List[float]) -> int:
    if not age_list: return 0
    combined = sorted(zip(quality_list, age_list), key=lambda x: x[0], reverse=True)
    top_n = max(1, len(combined) // 2)
    best_entries = combined[:top_n]
    weighted_ages = []
    for q, age in best_entries:
        count = max(1, int(q * 10))
        weighted_ages.extend([age] * count)
    return int(np.median(weighted_ages))

def get_age_bucket(age: int) -> str:
    if age <= 0: return "Unknown"
    base = 5 * (age // 5)
    return f"{base}-{base+5}"

def extract_face_crops(upper_body: np.ndarray, face_box: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fx1, fy1, fx2, fy2 = face_box
    fx1, fy1, fx2, fy2 = int(fx1), int(fy1), int(fx2), int(fy2)
    fh, fw = fy2 - fy1, fx2 - fx1
    h, w = upper_body.shape[:2]

    pw = int(fw * PAD_WIDE)
    pwy = int(fh * PAD_WIDE)
    wx1, wy1 = max(0, fx1 - pw), max(0, fy1 - pwy)
    wx2, wy2 = min(w, fx2 + pw), min(h, fy2 + pwy)
    img_wide = upper_body[wy1:wy2, wx1:wx2]

    pt = int(fw * PAD_TIGHT)
    pty = int(fh * PAD_TIGHT)
    tx1, ty1 = max(0, fx1 - pt), max(0, fy1 - pty)
    tx2, ty2 = min(w, fx2 + pt), min(h, fy2 + pty)
    img_tight = upper_body[ty1:ty2, tx1:tx2]

    return img_wide, img_tight


def draw_results(frame, locked_list, locked_data, analyzing, accumulators):
    for tid, box in locked_list:
        if tid in locked_data:
            data = locked_data[tid]
            x1, y1, x2, y2 = box
            color = COLOR_MALE if data.gender == "Male" else COLOR_FEMALE
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{tid} {data.gender} {data.age_bucket}", (x1, y1-25), 0, 0.6, color, 2)
            cv2.putText(frame, f"{data.race} (LOCKED)", (x1, y1-5), 0, 0.6, color, 2)

    for tid, box, acc, qual in analyzing:
        x1, y1, x2, y2 = box
        if acc:
            if acc.gender_score:
                top_g = max(acc.gender_score, key=acc.gender_score.get)
                score = acc.gender_score[top_g]
                label = f"{top_g} ({int(score*10)}%)"
            else:
                top_g = "?"
                score = 0
                label = "Scan..."

            prog = min(score / SCORE_TO_LOCK, 1.0)
            cv2.rectangle(frame, (x1, y1-30), (x1+int((x2-x1)*prog), y1-25), COLOR_ANALYZING, -1)
            cv2.putText(frame, label, (x1, y1-5), 0, 0.5, COLOR_ANALYZING, 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ANALYZING if acc else COLOR_UNTRACKED, 2)


def main():
    video_path = ensure_video_source(INPUT_SOURCE, OUTPUT_VIDEO)
    if not video_path:
        print("No valid video source.")
        return

    cap = cv2.VideoCapture(video_path)
    print(f"Processing {video_path}...")

    accumulators: Dict[int, Accumulator] = {}
    locked_data: Dict[int, DemographicData] = {}
    last_check: Dict[int, int] = {}
    frame_count = 0

    # --- CSV INITIALIZATION ---
    print(f"Logging demographics to: {CSV_FILENAME}")
    with open(CSV_FILENAME, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Track_ID", "Gender", "Race", "Age_Bucket", "Raw_Age", "Gender_Score", "Race_Score"])
    
    logged_ids = set() # Keep track of who we already saved to avoid duplicates

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        results = yolo_model.track(frame, persist=True, classes=0, conf=CONFIDENCE_THRESHOLD, verbose=False, tracker="bytetrack.yaml")

        if len(results) == 0 or results[0].boxes.id is None:
            cv2.imshow('Final Analysis', frame)
            if cv2.waitKey(1) == ord('q'): break
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        draw_locked = []
        draw_analyzing = []
        faces_to_process = []

        for box, track_id in zip(boxes, ids):
            if track_id in locked_data:
                draw_locked.append((track_id, box))
                continue

            if track_id not in accumulators:
                accumulators[track_id] = Accumulator()

            acc = accumulators[track_id]
            quality = acc.last_quality if acc else 0.0

            # EMA-based skip decision
            skip = SKIP_FRAMES
            if ADAPTIVE_SKIP_ENABLED:
                q = acc.ema_quality if acc.ema_quality is not None else quality
                skip = int(MAX_SKIP_FRAMES * q) if q > 0.5 else MIN_SKIP_FRAMES

            last_frame = last_check.get(track_id, 0)
            if (frame_count - last_frame) <= skip:
                draw_analyzing.append((track_id, box, acc, quality))
                continue

            x1, y1, x2, y2 = box
            h_body = y2 - y1
            upper_body = frame[y1:y1+int(h_body*0.45), x1:x2]
            if upper_body.size == 0:
                draw_analyzing.append((track_id, box, acc, quality))
                continue

            conf, face_box = get_face_box(upper_body)
            if conf < MIN_FACE_CONFIDENCE or face_box is None:
                draw_analyzing.append((track_id, box, acc, quality))
                continue

            img_wide, img_tight = extract_face_crops(upper_body, face_box)
            if img_tight is None or img_tight.size == 0 or img_tight.shape[0] < MIN_FACE_SIZE:
                draw_analyzing.append((track_id, box, acc, quality))
                continue

            quality = calculate_quality_score(img_tight)
            acc.last_quality = quality
            alpha_q = 0.25
            acc.ema_quality = (alpha_q * quality) + ((1 - alpha_q) * acc.ema_quality) if acc.ema_quality else quality

            if quality <= MIN_QUALITY_THRESHOLD:
                draw_analyzing.append((track_id, box, acc, quality))
                continue

            faces_to_process.append((track_id, box, img_wide, img_tight, quality))
            draw_analyzing.append((track_id, box, acc, quality))

        # Batch processing
        for track_id, box, img_w, img_t, quality in faces_to_process:
            last_check[track_id] = frame_count
            res = analyze_demographics_unified(img_w, img_t)

            if not res:
                continue

            acc = accumulators[track_id]
            acc.frame_count += 1

            male_p = res['male_prob']
            female_p = res['female_prob']
            frame_conf = res.get('frame_confidence', max(male_p, female_p))
            predicted_gender = 'Male' if male_p >= female_p else 'Female'

            # Update EMA of gender probs
            alpha = 0.35
            acc.ema_gender_prob['Male'] = alpha * male_p + (1 - alpha) * acc.ema_gender_prob.get('Male', 0.0)
            acc.ema_gender_prob['Female'] = alpha * female_p + (1 - alpha) * acc.ema_gender_prob.get('Female', 0.0)

            # update gender_score using weighted frame confidence * quality
            acc.gender_score['Male'] += acc.ema_gender_prob['Male'] * quality * frame_conf
            acc.gender_score['Female'] += acc.ema_gender_prob['Female'] * quality * frame_conf

            # history for consensus
            acc.recent_genders.append(predicted_gender)
            acc.gender_history.append(predicted_gender)

            # race & age updates
            race = res.get('race', 'Unknown')
            acc.race_score[race] += quality
            acc.race_history.append(race)
            age_raw = res.get('age_raw', 0)
            acc.age_list.append(age_raw)
            acc.age_quality_list.append(quality)
            if len(acc.age_list) > MAX_AGE_LIST_SIZE:
                acc.age_list = acc.age_list[-MAX_AGE_LIST_SIZE:]
                acc.age_quality_list = acc.age_quality_list[-MAX_AGE_LIST_SIZE:]

            # lock to prevent stuttering
            top_g = max(acc.gender_score, key=acc.gender_score.get)
            score_g = acc.gender_score[top_g]
            top_r = max(acc.race_score, key=acc.race_score.get) if acc.race_score else "Unknown"
            score_r = acc.race_score[top_r] if acc.race_score else 0.0

            total_votes = len(acc.gender_history) if acc.gender_history else 1
            agreement = acc.gender_history.count(top_g) / total_votes
            recent_agreement = (list(acc.recent_genders).count(top_g) / (len(acc.recent_genders) if len(acc.recent_genders) else 1))

            can_lock_normal = (score_g >= SCORE_TO_LOCK and score_r >= SCORE_TO_LOCK and agreement >= CONSENSUS_RATIO and recent_agreement >= (CONSENSUS_RATIO - 0.1))
            can_lock_forced = (acc.frame_count >= FORCED_LOCK_FRAMES and score_g > 1.0)

            if can_lock_normal or can_lock_forced:
                final_age = apply_temporal_smoothing(acc.age_list, acc.age_quality_list)
                locked_data[track_id] = DemographicData(
                    gender=top_g,
                    race=top_r,
                    age_bucket=get_age_bucket(final_age)
                )
                print(f"🔒 LOCKED ID {track_id}: {top_g}, {top_r}, Age {final_age}")

                # CSV logging
                if track_id not in logged_ids:
                    try:
                        bucket = get_age_bucket(final_age)
                        with open(CSV_FILENAME, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                track_id,
                                top_g,
                                top_r,
                                bucket,
                                final_age,
                                round(score_g, 2),
                                round(score_r, 2)
                            ])
                        logged_ids.add(track_id)
                        print(f"  -> Saved to {CSV_FILENAME}")
                    except Exception as e:
                        print(f"CSV Error: {e}")


        draw_results(frame, draw_locked, locked_data, draw_analyzing, accumulators)
        cv2.imshow('Final Analysis', frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
