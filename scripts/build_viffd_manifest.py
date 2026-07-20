"""
build_viffd_manifest.py
=======================
Scans VIFFD Training + Testing Data directories, infers labels and tamper
types from filenames, derives a subject/scene grouping for GroupKFold CV,
and writes data/viffd/manifest.csv.
"""

import os
import re
import csv
import shutil

VIFFD_ROOT = r"d:\coherex\data\sources\VIFFD - A dataset for detecting video inter-frame forgeries"
OUT_DIR    = r"d:\coherex\data\viffd"
MANIFEST   = os.path.join(OUT_DIR, "manifest.csv")

AUTH_DIR   = os.path.join(OUT_DIR, "authentic")
TAMP_DIR   = os.path.join(OUT_DIR, "tampered")

os.makedirs(AUTH_DIR, exist_ok=True)
os.makedirs(TAMP_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".AVI", ".MP4"}

SUBJECT_PATTERNS = [
    "salman", "haiyong", "huabin", "nemo", "baby", "girl", "women",
    "anwar", "alimi", "hossin", "hau", "sami", "mustafa", "fuwen",
    "finger", "canon", "samsung", "badminton", "football", "dance",
    "table tennis", "men", "man", "children", "student", "teacher",
    "desk", "cup", "grass", "garden", "bic", "ball", "plane", "car",
]

def infer_subject(filename):
    fn = filename.lower()
    for s in SUBJECT_PATTERNS:
        if s in fn:
            return s.replace(" ", "_")
    return "unknown"

def infer_tamper(filename):
    fn = filename.lower()
    if fn.startswith("copy") or "copy " in fn:
        return "frame_duplication"
    if fn.startswith("delete") or "delete " in fn:
        return "frame_deletion"
    if fn.startswith("insert") or "insert " in fn:
        return "frame_insertion"
    return "none"

def is_video(filename):
    _, ext = os.path.splitext(filename)
    return ext in VIDEO_EXTS

# -------------------------------------------------------------------
# Scan both splits
# -------------------------------------------------------------------
rows = []
clip_id = 0

for split_name in ["Training Data", "Testing Data"]:
    split_dir = os.path.join(VIFFD_ROOT, split_name)
    for fname in sorted(os.listdir(split_dir)):
        if not is_video(fname):
            continue

        src_path   = os.path.join(split_dir, fname)
        tamper_type = infer_tamper(fname)
        label       = 0 if tamper_type == "none" else 1
        subject     = infer_subject(fname)

        # Safe clip name — strip special chars for filenames
        safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", os.path.splitext(fname)[0])
        safe_name = re.sub(r"_+", "_", safe_name).strip("_")
        ext        = os.path.splitext(fname)[1].lower()
        clip_name  = f"viffd_{clip_id:04d}_{safe_name[:50]}{ext}"

        # Copy into authentic/ or tampered/
        dest_dir  = AUTH_DIR if label == 0 else TAMP_DIR
        dest_path = os.path.join(dest_dir, clip_name)

        if not os.path.exists(dest_path):
            shutil.copy2(src_path, dest_path)

        rows.append({
            "clip_name":         clip_name,
            "label":             label,
            "tamper_type":       tamper_type,
            "scene_id":          subject,          # used for GroupKFold
            "source_id":         f"viffd_{subject}",
            "split":             split_name,
            "original_filename": fname,
        })

        clip_id += 1
        if clip_id % 20 == 0:
            print(f"  Processed {clip_id} files...")

# -------------------------------------------------------------------
# Write manifest
# -------------------------------------------------------------------
fieldnames = ["clip_name","label","tamper_type","scene_id","source_id","split","original_filename"]
with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
auth  = sum(1 for r in rows if r["label"] == 0)
tamp  = sum(1 for r in rows if r["label"] == 1)
scenes = len(set(r["scene_id"] for r in rows))
types  = {}
for r in rows:
    types[r["tamper_type"]] = types.get(r["tamper_type"], 0) + 1

print(f"\n{'='*60}")
print(f"VIFFD Manifest built: {MANIFEST}")
print(f"  Total clips : {len(rows)}")
print(f"  Authentic   : {auth}")
print(f"  Tampered    : {tamp}")
print(f"  Unique scenes/subjects : {scenes}")
print(f"  Tamper breakdown:")
for k, v in sorted(types.items()):
    print(f"    {k:<25} : {v}")
print(f"{'='*60}")
print(f"\nVideos copied to:")
print(f"  {AUTH_DIR}")
print(f"  {TAMP_DIR}")
