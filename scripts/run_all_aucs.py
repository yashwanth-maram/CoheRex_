import os
import subprocess
import json

datasets = {
    "Pixel (Clean)": r"d:\coherex\results\features_pixel.csv",
    "Pixel (CRF36)": r"d:\coherex\results\features_pixel_crf36.csv",
    "Pixel (Temporal)": r"d:\coherex\results\features_pixel_temporal.csv",
    "Physics (Clean)": r"d:\coherex\results\features_physics.csv",
    "Physics (CRF36)": r"d:\coherex\results\features_physics_crf36.csv",
    "Physics (Temporal)": r"d:\coherex\results\features_physics_temporal.csv",
    "Hybrid (Clean)": r"d:\coherex\results\features_hybrid.csv",
    "Hybrid (CRF36)": r"d:\coherex\results\features_hybrid_crf36.csv",
    "Hybrid (Temporal)": r"d:\coherex\results\features_hybrid_temporal.csv"
}

results = []
metrics_file = r"d:\coherex\results\temp_all_metrics.json"

for name, path in datasets.items():
    if not os.path.exists(path):
        results.append((name, "MISSING", "-", "-", "-", "-", "-"))
        continue
        
    cmd = ["python", r"d:\coherex\scripts\train_meta_classifier.py", "--csv", path, "--metrics", metrics_file]
    
    # We pass --feature-set to ensure correct columns are selected
    if "Pixel" in name:
        cmd.extend(["--feature-set", "pixel"])
    elif "Physics" in name:
        cmd.extend(["--feature-set", "physics"])
    else:
        cmd.extend(["--feature-set", "hybrid"])
        
    print(f"Evaluating {name}...")
    subprocess.run(cmd, capture_output=True, text=True)
    
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            data = json.load(f)
            results.append((
                name,
                data.get("auc", 0),
                data.get("balanced_accuracy", 0),
                data.get("precision", 0),
                data.get("recall", 0),
                data.get("specificity", 0),
                data.get("f1", 0)
            ))
    else:
        results.append((name, "ERROR", "-", "-", "-", "-", "-"))

print("\n\n" + "="*110)
print(f"{'Dataset':<20} | {'AUC':<10} | {'Bal. Acc':<10} | {'Precision':<10} | {'Recall':<10} | {'Spec':<10} | {'F1':<10}")
print("-" * 110)
for r in results:
    if isinstance(r[1], float):
        print(f"{r[0]:<20} | {r[1]:<10.4f} | {r[2]:<10.4f} | {r[3]:<10.4f} | {r[4]:<10.4f} | {r[5]:<10.4f} | {r[6]:<10.4f}")
    else:
        print(f"{r[0]:<20} | {r[1]:<10} | {r[2]:<10} | {r[3]:<10} | {r[4]:<10} | {r[5]:<10} | {r[6]:<10}")
print("="*110)
