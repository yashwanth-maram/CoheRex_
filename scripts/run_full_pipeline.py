import os
import subprocess
import sys

def run(cmd):
    print(f"\n{'='*70}")
    print(f"RUNNING: {' '.join(cmd)}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"COMMAND FAILED WITH EXIT CODE {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    # 1. Dataset Generation
    run(["python", "scripts/build_dataset.py", "--recompress", "--crf", "36"])
    
    # We just need temporal
    run(["python", "scripts/create_temporal_interpolated.py"])

    # 2. Extract Base Features
    run(["python", "scripts/extract_pixel.py"])
    run(["python", "scripts/extract_physics.py"])
    run(["python", "scripts/extract_semantic.py"])
    run(["python", "scripts/merge_features.py"])
    
    # 3. Extract Recompressed Features
    # Note: recompressed output dir from build_dataset is data/evaluation_recompressed/crf36
    run(["python", "scripts/extract_pixel.py", "--dataset", r"evaluation_recompressed\crf36"])
    run(["python", "scripts/extract_physics.py", "--dataset", r"evaluation_recompressed\crf36"])
    run(["python", "scripts/extract_semantic.py", "--dataset", r"evaluation_recompressed\crf36"])
    run(["python", "scripts/merge_features.py", "--dataset", r"evaluation_recompressed\crf36"])

    # 4. Extract Temporal Features
    run(["python", "scripts/extract_pixel.py", "--dataset", "evaluation_temporal"])
    run(["python", "scripts/extract_physics.py", "--dataset", "evaluation_temporal"])
    run(["python", "scripts/extract_semantic.py", "--dataset", "evaluation_temporal"])
    run(["python", "scripts/merge_features.py", "--dataset", "evaluation_temporal"])

    # 5. Build Final Aggregated CSVs
    run(["python", "scripts/build_features.py"])

    # 6. Evaluate
    # Reduce trials since the dataset is much larger now, and we want it to finish in a reasonable time.
    run(["python", "-X", "utf8", "scripts/enhanced_classifier.py", "--compare", "--n-trials", "20"])
