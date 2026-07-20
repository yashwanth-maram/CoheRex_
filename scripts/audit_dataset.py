import os
import pandas as pd

clean_auth = os.listdir('data/evaluation/authentic')
clean_tamp = os.listdir('data/evaluation/tampered')
crf_auth = os.listdir('data/evaluation_recompressed/crf36/authentic')
crf_tamp = os.listdir('data/evaluation_recompressed/crf36/tampered')
temp_auth = os.listdir('data/evaluation_temporal/authentic')
temp_tamp = os.listdir('data/evaluation_temporal/tampered')

manifest = pd.read_csv('data/evaluation/manifest.csv')

print("=== DATASET AUDIT ===\n")
print(f"MANIFEST rows:              {len(manifest)}")
print(f"  Authentic in manifest:    {(manifest['label']=='authentic').sum()}")
print(f"  Tampered in manifest:     {(manifest['label']=='tampered').sum()}\n")

print(f"CLEAN FOLDER:               {len(clean_auth)} authentic + {len(clean_tamp)} tampered = {len(clean_auth)+len(clean_tamp)} total")
print(f"CRF36 FOLDER:               {len(crf_auth)} authentic + {len(crf_tamp)} tampered = {len(crf_auth)+len(crf_tamp)} total")
print(f"TEMPORAL FOLDER:            {len(temp_auth)} authentic + {len(temp_tamp)} tampered = {len(temp_auth)+len(temp_tamp)} total\n")

clean_all  = set(clean_auth + clean_tamp)
temp_all   = set(temp_auth + temp_tamp)
crf_all    = set(crf_auth + crf_tamp)

print(f"Clips in CLEAN but NOT in TEMPORAL:  {len(clean_all - temp_all)}")
print(f"Clips in TEMPORAL but NOT in CLEAN:  {len(temp_all - clean_all)}")
print(f"Clips in CLEAN but NOT in CRF36:     {len(clean_all - crf_all)}")
print(f"Clips in CRF36 but NOT in CLEAN:     {len(crf_all - clean_all)}")

manifest_clips = set(manifest['clip_name'].tolist())
print(f"\nClips in CLEAN but NOT in MANIFEST:  {len(clean_all - manifest_clips)}")
print(f"Clips in MANIFEST but NOT in CLEAN:  {len(manifest_clips - clean_all)}")
