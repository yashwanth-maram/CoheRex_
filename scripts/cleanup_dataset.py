"""
Dataset Cleanup Script
======================
Removes orphan clips from data/evaluation that are NOT in the manifest.
These are old clips from a previous build run that were not overwritten.
"""
import os
import shutil
import pandas as pd

manifest = pd.read_csv('data/evaluation/manifest.csv')
valid_clips = set(manifest['clip_name'].tolist())

print(f"Manifest has {len(valid_clips)} clips\n")

removed = 0
for subdir in ['authentic', 'tampered']:
    folder = os.path.join('data', 'evaluation', subdir)
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.mp4'):
            continue
        if fname not in valid_clips:
            path = os.path.join(folder, fname)
            print(f"  [DELETE] {subdir}/{fname}")
            os.remove(path)
            removed += 1

print(f"\nRemoved {removed} orphan clips.")

# Verify
auth = len(os.listdir('data/evaluation/authentic'))
tamp = len(os.listdir('data/evaluation/tampered'))
print(f"\nFinal clean folder: {auth} authentic + {tamp} tampered = {auth+tamp} total")
print(f"Manifest:           {len(valid_clips)} clips")
if auth + tamp == len(valid_clips):
    print("SUCCESS: Clean folder matches manifest exactly!")
else:
    print("WARNING: Mismatch! Check above for issues.")
