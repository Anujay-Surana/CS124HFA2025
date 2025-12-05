# estimate_demographics.py  ← FINAL CLEAN VERSION (no warnings, 100% working)

import os
import cv2
import numpy as np
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from datetime import datetime
import warnings

# Suppress PyTorch deprecation warnings (clean output)
warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Labels
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
RACE_LIST = ["White", "Black", "Latino_Hispanic", "East Asian", "Southeast Asian", "Indian", "Middle Eastern"]

print("Loading models...")

# Age & Gender (OpenCV)
age_net = cv2.dnn.readNetFromCaffe(
    os.path.join(MODELS_DIR, "age_deploy.prototxt"),
    os.path.join(MODELS_DIR, "age_net.caffemodel")
)
gender_net = cv2.dnn.readNetFromCaffe(
    os.path.join(MODELS_DIR, "gender_deploy.prototxt"),
    os.path.join(MODELS_DIR, "gender_net.caffemodel")
)

# FairFace — clean way (no pretrained warning)
print("Loading FairFace model...")
model = models.resnet34(weights=None)  # ← this removes the warning
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 18)  # 7 race + 2 gender + 9 age

state_dict = torch.load(
    os.path.join(MODELS_DIR, "res34_fair_align_multi_7_20190809.pt"),
    map_location="cpu"
)
model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("All models loaded!\n")


def predict_age_gender(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                 (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    gender_net.setInput(blob)
    gender = GENDER_LIST[gender_net.forward()[0].argmax()]
    age_net.setInput(blob)
    age = AGE_LIST[age_net.forward()[0].argmax()]
    return age, gender


def predict_race(face_img):
    pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    x = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        output = model(x)
    probs = torch.softmax(output, dim=1)[0]
    race_idx = probs[:7].argmax().item()
    race = RACE_LIST[race_idx]
    conf = probs[:7].max().item()
    return race, round(conf, 3)


# Main loop
found = False
for person_folder in os.listdir(OUTPUT_DIR):
    if not person_folder.startswith("person_"):
        continue
    found = True
    person_dir = os.path.join(OUTPUT_DIR, person_folder)
    meta_path = os.path.join(person_dir, "metadata.json")

    if not os.path.exists(meta_path):
        continue

    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except:
        continue
    if not meta:
        continue

    best = max(meta, key=lambda x: x.get("quality_score", 0))
    img_path = os.path.join(person_dir, f"{best['frame']:06d}.jpg")
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    age, gender = predict_age_gender(img)
    race, conf = predict_race(img)

    result = {
        "age": age,
        "gender": gender,
        "race": race,
        "race_confidence": conf,
        "source_image": f"{best['frame']:06d}.jpg",
        "estimated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(os.path.join(person_dir, "demographics.json"), "w") as f:
        json.dump(result, f, indent=4)

    print(f"{person_folder} → {gender}, {age}, {race} (confidence: {conf:.3f})")

if not found:
    print("No persons found. Run main.py first!")
else:
    print("\nDemographics estimation complete!")