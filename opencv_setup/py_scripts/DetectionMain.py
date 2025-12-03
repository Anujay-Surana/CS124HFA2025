import cv2
import numpy as np
import os
from collections import deque

# Pre-trained DNN face detector model from OpenCV
modelFile = "opencv_setup/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "opencv_setup/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Gender classification model
genderProto = "opencv_setup/gender_deploy.prototxt"
genderModel = "opencv_setup/gender_net.caffemodel"
genderNet = cv2.dnn.readNetFromCaffe(genderProto, genderModel)
genderList = ['Male', 'Female']

# Age classification model
ageProto = "opencv_setup/age_deploy.prototxt"
ageModel = "opencv_setup/age_net.caffemodel"
ageNet = cv2.dnn.readNetFromCaffe(ageProto, ageModel)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Body detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Model mean values for preprocessing
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Set confidence threshold
confidence_threshold = 0.5

# Temporal smoothing
face_tracker = {}
next_face_id = 0
SMOOTHING_FRAMES = 7


class FaceTracker:
    def __init__(self, face_id):
        self.id = face_id
        self.age_history = deque(maxlen=SMOOTHING_FRAMES)
        self.gender_history = deque(maxlen=SMOOTHING_FRAMES)
        self.ethnicity_history = deque(maxlen=SMOOTHING_FRAMES)
        self.last_position = None
        self.frames_missing = 0
        
    def update(self, position, age, gender, ethnicity):
        self.last_position = position
        self.age_history.append(age)
        self.gender_history.append(gender)
        self.ethnicity_history.append(ethnicity)
        self.frames_missing = 0
        
    def get_smoothed_predictions(self):
        if len(self.age_history) > 0:
            age = max(set(self.age_history), key=self.age_history.count)
            gender = max(set(self.gender_history), key=self.gender_history.count)
            ethnicity = max(set(self.ethnicity_history), key=self.ethnicity_history.count)
            return age, gender, ethnicity
        return "Unknown", "Unknown", "Unknown"


def match_face_to_tracker(x, y, w, h):
    global face_tracker, next_face_id
    
    center_x = x + w // 2
    center_y = y + h // 2
    
    min_dist = float('inf')
    matched_id = None
    
    for face_id, tracker in list(face_tracker.items()):
        if tracker.last_position is not None:
            tx, ty, tw, th = tracker.last_position
            tcx = tx + tw // 2
            tcy = ty + th // 2
            dist = np.sqrt((center_x - tcx)**2 + (center_y - tcy)**2)
            
            if dist < w * 0.6 and dist < min_dist:
                min_dist = dist
                matched_id = face_id
    
    if matched_id is None:
        matched_id = next_face_id
        face_tracker[matched_id] = FaceTracker(matched_id)
        next_face_id += 1
    
    return matched_id


def cleanup_trackers():
    global face_tracker
    to_remove = []
    for face_id, tracker in face_tracker.items():
        tracker.frames_missing += 1
        if tracker.frames_missing > 30:
            to_remove.append(face_id)
    for face_id in to_remove:
        del face_tracker[face_id]


def estimate_ethnicity(face_img):
    """
    Improved ethnicity estimation
    """
    try:
        if face_img.size == 0 or face_img.shape[0] < 30 or face_img.shape[1] < 30:
            return "Unknown", 0.0

        face_h, face_w = face_img.shape[:2]
        
        # Extract forehead region
        forehead_top = int(face_h * 0.18)
        forehead_bottom = int(face_h * 0.48)
        forehead_left = int(face_w * 0.22)
        forehead_right = int(face_w * 0.78)
        
        forehead = face_img[forehead_top:forehead_bottom, forehead_left:forehead_right]
        
        if forehead.size == 0:
            return "Unknown", 0.0
        
        # Color space analysis
        lab = cv2.cvtColor(forehead, cv2.COLOR_BGR2LAB)
        ycrcb = cv2.cvtColor(forehead, cv2.COLOR_BGR2YCrCb)
        
        L = np.mean(lab[:, :, 0])
        a = np.mean(lab[:, :, 1])
        b = np.mean(lab[:, :, 2])
        
        Cr = np.mean(ycrcb[:, :, 1])
        Cb = np.mean(ycrcb[:, :, 2])
        
        # Classification with better thresholds
        if L < 88:
            ethnicity = "African"
            confidence = 0.73
        elif L < 92 and Cr > 145:
            ethnicity = "African"
            confidence = 0.68
        elif L < 135:
            if Cr > 144:
                ethnicity = "South Asian"
                confidence = 0.70
            elif b > 133 or Cr > 138:
                ethnicity = "Hispanic/Latino"
                confidence = 0.66
            else:
                ethnicity = "Middle Eastern"
                confidence = 0.62
        elif L < 168:
            if b > 134:
                ethnicity = "East Asian"
                confidence = 0.67
            elif Cr > 137:
                ethnicity = "Hispanic/Latino"
                confidence = 0.64
            else:
                ethnicity = "Middle Eastern"
                confidence = 0.60
        else:
            if Cb < 118:
                ethnicity = "Caucasian"
                confidence = 0.73
            else:
                ethnicity = "East Asian"
                confidence = 0.63
        
        return ethnicity, confidence
        
    except Exception as e:
        return "Unknown", 0.0


def get_input_source():
    print("\nChoose input source:")
    print("1. Live Camera")
    print("2. Image File")
    print("3. Video File")
    
    while True:
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            return "camera", 0
        elif choice == "2":
            while True:
                file_path = input("Enter the full path to your image file: ").strip()
                if os.path.exists(file_path):
                    print(f"Found image: {file_path}")
                    return "image", file_path
                else:
                    print("File not found. Please check the path and try again.")
                    retry = input("Try again? (y/n): ").strip().lower()
                    if retry != 'y':
                        break
        elif choice == "3":
            while True:
                file_path = input("Enter the full path to your video file: ").strip()
                if os.path.exists(file_path):
                    print(f"Found video: {file_path}")
                    return "video", file_path
                else:
                    print("File not found. Please check the path and try again.")
                    retry = input("Try again? (y/n): ").strip().lower()
                    if retry != 'y':
                        break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def preprocess_face(face_img):
    """
    Smart preprocessing
    """
    # CLAHE on LAB
    lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced


def predict_age_gender_robust(face_img):
    """
    Make predictions with multiple variations
    """
    try:
        # Original
        original = face_img
        # Enhanced
        enhanced = preprocess_face(face_img)
        # Gamma adjusted
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        gamma_adj = cv2.LUT(face_img, table)
        
        faces = [original, enhanced, gamma_adj]
        
        gender_votes = []
        age_votes = []
        
        for face in faces:
            # Gender
            blob_gender = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob_gender)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            gender_conf = float(genderPreds[0].max())
            gender_votes.append((gender, gender_conf))
            
            # Age
            blob_age = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            ageNet.setInput(blob_age)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            age_conf = float(agePreds[0].max())
            age_votes.append((age, age_conf))
        
        # Weighted voting
        gender_counts = {}
        for gender, conf in gender_votes:
            gender_counts[gender] = gender_counts.get(gender, 0) + conf
        final_gender = max(gender_counts, key=gender_counts.get)
        final_gender_conf = max([conf for g, conf in gender_votes if g == final_gender])
        
        age_counts = {}
        for age, conf in age_votes:
            age_counts[age] = age_counts.get(age, 0) + conf
        final_age = max(age_counts, key=age_counts.get)
        final_age_conf = max([conf for a, conf in age_votes if a == final_age])
        
        return final_age, final_age_conf, final_gender, final_gender_conf
        
    except Exception as e:
        return "Unknown", 0.0, "Unknown", 0.0


def detect_faces_and_bodies(frame, use_smoothing=True):
    """
    Process frame for face and body detection
    """
    (h, w) = frame.shape[:2]

    # Face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False)
    net.setInput(blob)
    detections = net.forward()

    # Body detection
    bodies, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)

    for (x, y, w_body, h_body) in bodies:
        cv2.rectangle(frame, (x, y), (x + w_body, y + h_body), (255, 0, 0), 2)
        cv2.putText(frame, "Body", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if use_smoothing:
        cleanup_trackers()

    # Process faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            face_width = endX - startX
            face_height = endY - startY

            if face_width < 40 or face_height < 40:
                continue

            face_id = None
            if use_smoothing:
                face_id = match_face_to_tracker(startX, startY, face_width, face_height)

            # Good padding amount
            pad_x = int(face_width * 0.30)
            pad_y = int(face_height * 0.35)
            
            pad_startX = max(0, startX - pad_x)
            pad_startY = max(0, startY - pad_y)
            pad_endX = min(w, endX + pad_x)
            pad_endY = min(h, endY + pad_y)

            face_img_padded = frame[pad_startY:pad_endY, pad_startX:pad_endX].copy()
            face_img_tight = frame[startY:endY, startX:endX].copy()

            gender = "Unknown"
            gender_conf = 0.0
            age = "Unknown"
            age_conf = 0.0
            ethnicity = "Unknown"
            eth_conf = 0.0

            if face_img_padded.size > 0 and face_img_padded.shape[0] > 60 and face_img_padded.shape[1] > 60:
                try:
                    # Robust prediction
                    age, age_conf, gender, gender_conf = predict_age_gender_robust(face_img_padded)
                    
                    # Ethnicity
                    ethnicity, eth_conf = estimate_ethnicity(face_img_tight)

                    if use_smoothing and face_id is not None:
                        face_tracker[face_id].update(
                            (startX, startY, face_width, face_height), 
                            age, gender, ethnicity
                        )
                        age, gender, ethnicity = face_tracker[face_id].get_smoothed_predictions()

                except Exception as e:
                    pass

            # Draw bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Labels (no face confidence)
            gender_text = f"{gender}: {gender_conf * 100:.1f}%"
            age_text = f"Age {age}: {age_conf * 100:.1f}%"
            ethnicity_text = f"{ethnicity}: {eth_conf * 100:.1f}%"

            # Draw text
            y_offset = startY - 10 if startY > 90 else endY + 20
            texts = [gender_text, age_text, ethnicity_text]
            colors = [(255, 200, 0), (255, 100, 255), (0, 200, 255)]

            for idx, (text, color) in enumerate(zip(texts, colors)):
                if y_offset < 90:
                    y_pos = y_offset + idx * 25
                else:
                    y_pos = y_offset - (len(texts) - idx - 1) * 25

                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                cv2.rectangle(frame, (startX, y_pos - text_h - 4), (startX + text_w, y_pos + 4), color, -1)
                cv2.putText(frame, text, (startX, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame


# Get input source
input_type, input_source = get_input_source()

if input_type == "camera":
    print("Face Detection Started. Press 'q' to quit.")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
elif input_type == "image":
    print(f"Processing image: {input_source}")
    cap = None
elif input_type == "video":
    print(f"Processing video: {input_source}")
    cap = cv2.VideoCapture(input_source)

# Process based on input type
if input_type == "image":
    print(f"Attempting to load image: {input_source}")
    frame = cv2.imread(input_source)
    if frame is None:
        print(f"Error: Could not load image from {input_source}")
        print("Please check that the file exists and is a valid image format.")
        exit(1)
    
    print(f"Successfully loaded image. Dimensions: {frame.shape[1]}x{frame.shape[0]} (width x height)")
    
    processed_frame = detect_faces_and_bodies(frame.copy(), use_smoothing=False)
    
    max_display_width = 1920
    max_display_height = 1080
    
    if processed_frame.shape[1] > max_display_width or processed_frame.shape[0] > max_display_height:
        scale_w = max_display_width / processed_frame.shape[1]
        scale_h = max_display_height / processed_frame.shape[0]
        scale = min(scale_w, scale_h)
        
        new_width = int(processed_frame.shape[1] * scale)
        new_height = int(processed_frame.shape[0] * scale)
        
        print(f"Image is large, resizing for display: {new_width}x{new_height}")
        display_frame = cv2.resize(processed_frame, (new_width, new_height))
    else:
        display_frame = processed_frame
    
    cv2.imshow('Face & Body Detection - Image', display_frame)
    print("Image displayed! Press any key to close the image window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Image processing completed.")
    
else:
    if cap is None or not cap.isOpened():
        print("Error: Could not initialize video capture")
        exit(1)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if input_type == "video":
                print("End of video reached")
            else:
                print("Failed to grab frame")
            break

        frame_count += 1
        
        processed_frame = detect_faces_and_bodies(frame, use_smoothing=True)

        window_title = 'Face & Body Detection'
        if input_type == "camera":
            window_title += ' - Live Camera (Press Q to Quit)'
        else:
            window_title += ' - Video (Press Q to Quit)'
        
        cv2.imshow(window_title, processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    
    if input_type == "video":
        print(f"Processed {frame_count} frames")