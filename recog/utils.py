# utils.py
import os, json, time
import numpy as np

def ensure_person_dir(base, pid):
    """Create a folder for each person."""
    pdir = os.path.join(base, f"person_{pid:03d}")
    os.makedirs(pdir, exist_ok=True)
    return pdir

def append_metadata(pdir, entry):
    """Append one record to metadata.json."""
    meta_path = os.path.join(pdir, "metadata.json")
    data = []
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
    data.append(entry)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_person_feature(pdir, feature):
    """Save the face feature vector for a person."""
    if feature is None:
        return
    feature_path = os.path.join(pdir, "feature.npy")
    np.save(feature_path, feature)

def load_existing_persons(output_dir):
    """
    Load existing persons from output directory.

    Returns:
        dict: {person_id: feature_vector} for all existing persons
    """
    persons = {}
    if not os.path.exists(output_dir):
        return persons

    for dirname in os.listdir(output_dir):
        if not dirname.startswith("person_"):
            continue

        try:
            # Extract person ID from folder name (e.g., "person_000" -> 0)
            pid = int(dirname.split("_")[1])

            # Load feature vector if it exists
            feature_path = os.path.join(output_dir, dirname, "feature.npy")
            if os.path.exists(feature_path):
                feature = np.load(feature_path)
                persons[pid] = feature
        except (ValueError, IndexError, IOError) as e:
            print(f"Warning: Could not load person from {dirname}: {e}")
            continue

    return persons

def now_ts():
    """Return current local time as string."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())