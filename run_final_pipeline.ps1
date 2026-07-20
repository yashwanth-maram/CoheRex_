# run_final_pipeline.ps1

Write-Host "[1/3] Extracting physics for CRF36 (8 parallel shards)..."

$jobs = @()
for ($i = 0; $i -lt 8; $i++) {
    $shard = $i
    $jobs += Start-Job -ScriptBlock {
        param($s)
        Set-Location d:\coherex
        $env:OMP_NUM_THREADS="1"
        d:\coherex\venv\Scripts\python.exe scripts/extract_physics.py --input data/evaluation_recompressed/crf36 --out cache/physics_crf36 --manifest data/evaluation/manifest.csv --no-pose --resume --num-shards 8 --shard $s
    } -ArgumentList $shard
}

Wait-Job $jobs
Receive-Job $jobs
Write-Host "[2/3] Building feature CSVs..."
d:\coherex\venv\Scripts\python.exe scripts/build_features.py --cache cache/physics_clean --out results/features_physics.csv
d:\coherex\venv\Scripts\python.exe scripts/merge_features.py --pixel results/features_pixel.csv --physics results/features_physics.csv --out results/features_hybrid.csv

d:\coherex\venv\Scripts\python.exe scripts/build_features.py --cache cache/physics_crf36 --out results/features_physics_crf36.csv
d:\coherex\venv\Scripts\python.exe scripts/build_features.py --cache cache/pixel_crf36 --out results/features_pixel_crf36.csv
d:\coherex\venv\Scripts\python.exe scripts/merge_features.py --pixel results/features_pixel_crf36.csv --physics results/features_physics_crf36.csv --out results/features_hybrid_crf28.csv
d:\coherex\venv\Scripts\python.exe scripts/merge_features.py --pixel results/features_pixel_crf36.csv --physics results/features_physics_crf36.csv --out results/features_hybrid_crf32.csv
d:\coherex\venv\Scripts\python.exe scripts/merge_features.py --pixel results/features_pixel_crf36.csv --physics results/features_physics_crf36.csv --out results/features_hybrid_crf36.csv

Write-Host "[3/3] Generating Robustness Table..."
d:\coherex\venv\Scripts\python.exe scripts/robustness_eval.py

Write-Host "[DONE] Pipeline complete!"
