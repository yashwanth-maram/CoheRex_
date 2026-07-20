import os
import subprocess
import json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_gated_csv(dataset):
    print(f"Creating gated-hybrid CSV for {dataset}...")
    if dataset == "VIRAT":
        df = pd.read_csv(os.path.join(ROOT, "results", "features_hybrid.csv"))
        out_path = os.path.join(ROOT, "results", "features_gated_hybrid.csv")
    else:
        df = pd.read_csv(os.path.join(ROOT, "results", "viffd_features_hybrid.csv"))
        out_path = os.path.join(ROOT, "results", "viffd_features_gated_hybrid.csv")
        
    phys_features = [c for c in df.columns if c.startswith("phys_") or c.startswith("scene_")]
    phys_features = [c for c in phys_features if c != "scene_id"] # Keep metadata intact
    
    # Pre-committed gating threshold
    gate_mask = (df["scene_num_tracks"] >= 3) & (df["scene_mean_track_len"] >= 20)
    
    print(f"  {dataset} Gate ON (hybrid used): {gate_mask.sum()} / {len(df)} ({(gate_mask.sum()/len(df))*100:.1f}%)")
    
    # Zero out physics features where gate is OFF
    for c in phys_features:
        if c in df.columns:
            df.loc[~gate_mask, c] = 0.0
            
    df.to_csv(out_path, index=False)
    return out_path, (gate_mask.sum()/len(df))*100

def run_train(csv_path, feature_set, dataset, config_name, gate_percent=None):
    print(f"\n========================================================")
    print(f"Running Config: {config_name} ({dataset} - {feature_set})")
    print(f"========================================================")
    
    metrics_path = os.path.join(ROOT, "results", f"metrics_{config_name.replace(' ', '_')}.json")
    
    cmd = [
        "python", os.path.join(ROOT, "scripts", "train_meta_classifier.py"),
        "--csv", csv_path,
        "--feature-set", feature_set,
        "--metrics", metrics_path
    ]
    
    subprocess.run(cmd, check=True)
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    metrics["config_name"] = config_name
    metrics["dataset"] = dataset
    if gate_percent is not None:
        metrics["gate_on_percent"] = round(gate_percent, 1)
        
    return metrics

def main():
    results = []
    
    # Create gated CSVs
    virat_gated_path, virat_gate_pct = create_gated_csv("VIRAT")
    viffd_gated_path, viffd_gate_pct = create_gated_csv("VIFFD")
    
    configs = [
        # VIRAT
        (os.path.join(ROOT, "results", "features_pixel.csv"), "pixel", "VIRAT", "VIRAT_Pixel", None),
        (os.path.join(ROOT, "results", "features_physics.csv"), "physics", "VIRAT", "VIRAT_Physics", None),
        (os.path.join(ROOT, "results", "features_hybrid.csv"), "hybrid", "VIRAT", "VIRAT_Hybrid", None),
        (virat_gated_path, "hybrid", "VIRAT", "VIRAT_Gated_Hybrid", virat_gate_pct),
        
        # VIFFD
        (os.path.join(ROOT, "results", "viffd_features_pixel.csv"), "pixel", "VIFFD", "VIFFD_Pixel", None),
        (os.path.join(ROOT, "results", "viffd_features_physics.csv"), "physics", "VIFFD", "VIFFD_Physics", None),
        (os.path.join(ROOT, "results", "viffd_features_hybrid.csv"), "hybrid", "VIFFD", "VIFFD_Hybrid", None),
        (viffd_gated_path, "hybrid", "VIFFD", "VIFFD_Gated_Hybrid", viffd_gate_pct),
    ]
    
    for csv_path, feature_set, dataset, config_name, gate_pct in configs:
        res = run_train(csv_path, feature_set, dataset, config_name, gate_pct)
        results.append(res)
        
    out_json = os.path.join(ROOT, "results", "protocol_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n\n" + "="*80)
    print("PROTOCOL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Config':<20} | {'AUC (95% CI)':<25} | {'F1 (Youden)':<12} | {'F1 (0.5)':<10} | {'Gate ON %'}")
    print("-" * 80)
    for r in results:
        auc_str = f"{r['auc']:.4f} ({r['auc_ci_lo']:.4f}-{r['auc_ci_hi']:.4f})"
        gate_str = f"{r.get('gate_on_percent', 'N/A')}%"
        print(f"{r['config_name']:<20} | {auc_str:<25} | {r['f1']:.4f}       | {r.get('f1_05', 0.0):.4f}   | {gate_str}")
        
if __name__ == "__main__":
    main()
