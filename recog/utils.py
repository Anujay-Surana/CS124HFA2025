# utils.py
import os, json, time

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

def now_ts():
    """Return current local time as string."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())