import csv
from datetime import datetime
from pathlib import Path

# Base directory for this script (.../opencv_setup/py_scripts)
BASE_DIR = Path(__file__).resolve().parent

# Folder where logs will be stored
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# CSV file that will store detection info
LOG_FILE = LOG_DIR / "detections.csv"


def init_log():
    """
    Create the CSV file with a header row if it doesn't exist yet.
    Call this once at the start of the program.
    """
    if not LOG_FILE.exists():
        with LOG_FILE.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time",
                "source",        # camera / image / video
                "frame",
                "person_id",
                "x", "y", "w", "h",
                "age",
                "age_conf",
                "gender",
                "gender_conf",
                "ethnicity",
                "eth_conf",
            ])


def log_detection(
    source,
    frame_idx,
    person_id,
    x,
    y,
    w,
    h,
    age,
    age_conf,
    gender,
    gender_conf,
    ethnicity,
    eth_conf,
):
    """
    Append one detection record to the CSV file.
    """
    with LOG_FILE.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            str(source),
            int(frame_idx),
            int(person_id),
            int(x),
            int(y),
            int(w),
            int(h),
            str(age),
            float(age_conf),
            str(gender),
            float(gender_conf),
            str(ethnicity),
            float(eth_conf),
        ])


if __name__ == "__main__":
    # Simple self-test
    init_log()
    print("Logging 3 fake detections...")

    for i in range(3):
        log_detection(
            source="test",
            frame_idx=i,
            person_id=1,
            x=10 + i * 5,
            y=20,
            w=60,
            h=60,
            age="(25-32)",
            age_conf=0.9,
            gender="Male",
            gender_conf=0.95,
            ethnicity="Caucasian",
            eth_conf=0.8,
        )

    print("Done. Check logs/detections.csv")