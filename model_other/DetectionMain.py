import cv2
import numpy as np
import os

# Pre-trained DNN face detector model from OpenCV
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Gender classification model
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
genderNet = cv2.dnn.readNetFromCaffe(genderProto, genderModel)
genderList = ['Male', 'Female']

# Age classification model
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
ageNet = cv2.dnn.readNetFromCaffe(ageProto, ageModel)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Body detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Model mean values for preprocessing
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Webcam (0 is default camera)
cap = cv2.VideoCapture(0)

# Set confidence threshold (70%)
confidence_threshold = 0.7


def estimate_ethnicity(face_img):
    """
    Estimate ethnicity based on facial skin tone analysis.
    Returns ethnicity category and confidence score.
    """
    if face_img.size == 0:
        return "Unknown", 0.0

    # Convert to different color spaces for analysis
    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)

    # Get average values from the face region
    avg_value = np.mean(hsv[:, :, 2])
    avg_saturation = np.mean(hsv[:, :, 1])
    avg_cr = np.mean(ycrcb[:, :, 1])
    avg_cb = np.mean(ycrcb[:, :, 2])

    # Classification with confidence calculation
    confidence_scores = []
    
    if avg_value < 90:
        ethnicity = "African"
        # Calculate confidence based on how well the values fit the category
        value_confidence = min(1.0, (90 - avg_value) / 30.0 + 0.5)
        confidence_scores.append(value_confidence)
    elif avg_value < 130:
        if avg_cr > 140:
            ethnicity = "South Asian"
            value_confidence = min(1.0, (avg_value - 90) / 40.0 + 0.6)
            cr_confidence = min(1.0, (avg_cr - 140) / 20.0 + 0.6)
            confidence_scores.extend([value_confidence, cr_confidence])
        else:
            ethnicity = "Hispanic/Latino"
            value_confidence = min(1.0, (avg_value - 90) / 40.0 + 0.6)
            cr_confidence = min(1.0, (140 - avg_cr) / 20.0 + 0.6)
            confidence_scores.extend([value_confidence, cr_confidence])
    elif avg_value < 170:
        if avg_saturation > 40:
            ethnicity = "East Asian"
            value_confidence = min(1.0, (avg_value - 130) / 40.0 + 0.6)
            sat_confidence = min(1.0, (avg_saturation - 40) / 30.0 + 0.6)
            confidence_scores.extend([value_confidence, sat_confidence])
        else:
            ethnicity = "Middle Eastern"
            value_confidence = min(1.0, (avg_value - 130) / 40.0 + 0.6)
            sat_confidence = min(1.0, (40 - avg_saturation) / 30.0 + 0.6)
            confidence_scores.extend([value_confidence, sat_confidence])
    else:
        if avg_cb < 120:
            ethnicity = "Caucasian"
            value_confidence = min(1.0, (avg_value - 170) / 30.0 + 0.6)
            cb_confidence = min(1.0, (120 - avg_cb) / 20.0 + 0.6)
            confidence_scores.extend([value_confidence, cb_confidence])
        else:
            ethnicity = "East Asian"
            value_confidence = min(1.0, (avg_value - 170) / 30.0 + 0.6)
            cb_confidence = min(1.0, (avg_cb - 120) / 20.0 + 0.6)
            confidence_scores.extend([value_confidence, cb_confidence])
    
    # Calculate final confidence as average of individual confidence scores
    confidence = np.mean(confidence_scores) if confidence_scores else 0.5
    confidence = max(0.3, min(0.95, confidence))  # Clamp between 30% and 95%

    return ethnicity, confidence


def get_input_source():
    """
    Get input source from user - camera, image file, or video file
    """
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


def detect_faces_and_bodies(frame):
    """
    Process a single frame for face and body detection
    Returns the processed frame with annotations
    """
    # Get frame dimensions
    (h, w) = frame.shape[:2]

    # Prepare the frame for the DNN model (resize for processing but keep original for display)
    resized_frame = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(resized_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network for face detection
    net.setInput(blob)
    detections = net.forward()

    # Detect bodies using HOG on original frame
    bodies, weights = hog.detectMultiScale(frame, winStride=(8, 8),
                                           padding=(4, 4), scale=1.05)

    # Draw body detections
    for (x, y, w_body, h_body) in bodies:
        cv2.rectangle(frame, (x, y), (x + w_body, y + h_body),
                      (255, 0, 0), 3)
        cv2.putText(frame, "Body", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)

    # Loop through all face detections
    for i in range(detections.shape[2]):
        # Extract confidence for each detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > confidence_threshold:
            # Get bounding box coordinates from the 300x300 detection
            box = detections[0, 0, i, 3:7]
            (startX_300, startY_300, endX_300, endY_300) = box
            
            # Scale coordinates back to original image dimensions
            startX = int(startX_300 * w)
            startY = int(startY_300 * h)
            endX = int(endX_300 * w)
            endY = int(endY_300 * h)

            # Ensure coordinates are within frame bounds
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # Extract face region for analysis
            face_img = frame[startY:endY, startX:endX]

            if face_img.size > 0:
                # Predict gender
                blob_gender = cv2.dnn.blobFromImage(
                    face_img, 1.0, (227, 227),
                    MODEL_MEAN_VALUES, swapRB=False
                )
                genderNet.setInput(blob_gender)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]
                gender_confidence = genderPreds[0].max()

                # Predict age
                blob_age = cv2.dnn.blobFromImage(
                    face_img, 1.0, (227, 227),
                    MODEL_MEAN_VALUES, swapRB=False
                )
                ageNet.setInput(blob_age)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                age_confidence = agePreds[0].max()

                # Estimate ethnicity
                ethnicity, eth_confidence = estimate_ethnicity(face_img)
            else:
                gender = "Unknown"
                gender_confidence = 0.0
                age = "Unknown"
                age_confidence = 0.0
                ethnicity = "Unknown"
                eth_confidence = 0.0

            # Draw bounding box around the face
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 3)

            # Prepare text labels
            face_text = f"Face: {confidence * 100:.1f}%"
            gender_text = f"{gender}: {gender_confidence * 100:.1f}%"
            age_text = f"Age {age}: {age_confidence * 100:.1f}%"
            ethnicity_text = f"{ethnicity}: {eth_confidence * 100:.1f}%"

            # Calculate positions for text
            y_offset = startY - 10 if startY - 10 > 10 else startY + 10

            # Draw text with background for better visibility
            texts = [face_text, gender_text, age_text, ethnicity_text]
            colors = [(0, 255, 0), (255, 200, 0), (255, 100, 255), (0, 200, 255)]

            for idx, (text, color) in enumerate(zip(texts, colors)):
                y_pos = y_offset - (len(texts) - idx - 1) * 30

                # Get text size with larger font
                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 3
                )

                # Draw background rectangle
                cv2.rectangle(frame,
                              (startX, y_pos - text_height - 5),
                              (startX + text_width, y_pos + 5),
                              color, -1)

                # Put text with larger font
                cv2.putText(frame, text, (startX, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)

    return frame


# Get input source
input_type, input_source = get_input_source()

if input_type == "camera":
    print("Face Detection Started. Press 'q' to quit.")
    cap = cv2.VideoCapture(0)
elif input_type == "image":
    print(f"Processing image: {input_source}")
    cap = None
elif input_type == "video":
    print(f"Processing video: {input_source}")
    cap = cv2.VideoCapture(input_source)

# Process based on input type
if input_type == "image":
    print(f"Attempting to load image: {input_source}")
    # Process single image
    frame = cv2.imread(input_source)
    if frame is None:
        print(f"Error: Could not load image from {input_source}")
        print("Please check that the file exists and is a valid image format.")
        exit(1)
    
    print(f"Successfully loaded image. Dimensions: {frame.shape[1]}x{frame.shape[0]} (width x height)")
    
    # Process the image on the original frame
    processed_frame = detect_faces_and_bodies(frame.copy())
    
    # Check if image is too large for display and resize if necessary
    max_display_width = 1920
    max_display_height = 1080
    
    if processed_frame.shape[1] > max_display_width or processed_frame.shape[0] > max_display_height:
        # Calculate scaling factor to fit within display limits
        scale_w = max_display_width / processed_frame.shape[1]
        scale_h = max_display_height / processed_frame.shape[0]
        scale = min(scale_w, scale_h)
        
        new_width = int(processed_frame.shape[1] * scale)
        new_height = int(processed_frame.shape[0] * scale)
        
        print(f"Image is large, resizing for display: {new_width}x{new_height}")
        display_frame = cv2.resize(processed_frame, (new_width, new_height))
    else:
        display_frame = processed_frame
    
    # Display the result
    cv2.imshow('Face & Body Detection - Image', display_frame)
    print("Image displayed! Press any key to close the image window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Image processing completed.")
    
else:
    # Process video (camera or video file)
    if cap is None:
        print("Error: Could not initialize video capture")
        exit(1)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            if input_type == "video":
                print("End of video reached")
            else:
                print("Failed to grab frame")
            break

        # Process the frame
        processed_frame = detect_faces_and_bodies(frame)

        # Display the resulting frame
        window_title = 'Face & Body Detection'
        if input_type == "camera":
            window_title += ' - Live Camera (Press Q to Quit)'
        else:
            window_title += ' - Video (Press Q to Quit)'
        
        cv2.imshow(window_title, processed_frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()