"""
extract_semantic.py - Semantic Temporal Consistency Feature Extractor
====================================================================

Extracts per-frame semantic embeddings from a frozen CLIP ViT-B/32 model,
then computes temporal statistics that capture tampering signatures in
the semantic trajectory space.

Key insight: Foundation model embeddings encode WHAT is in a frame (objects,
layout, scene semantics), not raw pixel values. Temporal statistics on these
embeddings are inherently robust to both spatial recompression and temporal
interpolation because the semantic content survives these transformations.

Features extracted:
  - Cosine similarity statistics (mean, std, min, max, range)
  - Gradient of cosine similarity (max jump detection)
  - Embedding velocity/acceleration statistics  
  - Stasis ratio (fraction of near-identical consecutive embeddings)
  - Trajectory reversibility score
  - Spectral energy ratio (FFT of cosine similarity sequence)
  - PCA temporal variance ratios

Usage:
    python scripts/extract_semantic.py --input data/evaluation --out cache/semantic_clean
    python scripts/extract_semantic.py --input data/evaluation_temporal --out cache/semantic_temporal
"""

import os
import sys
import cv2
import argparse
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "data", "evaluation", "manifest.csv")


def parse_args():
    p = argparse.ArgumentParser(description="Semantic Temporal Consistency Extractor")
    p.add_argument("--input", type=str, required=True,
                   help="Input directory (must contain authentic/ and tampered/ subdirs)")
    p.add_argument("--out", type=str, required=True,
                   help="Output cache directory")
    p.add_argument("--manifest", type=str, default=MANIFEST)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--frame-skip", type=int, default=1,
                   help="Process every Nth frame (1=all, 2=every other, 3=every 3rd)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size for CLIP inference")
    p.add_argument("--model", type=str, default="ViT-B-32",
                   help="OpenCLIP model name")
    p.add_argument("--pretrained", type=str, default="openai",
                   help="Pretrained weights")
    return p.parse_args()


def load_clip_model(model_name, pretrained, device):
    """Load frozen CLIP model for feature extraction."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    return model, preprocess


def extract_frame_embeddings(video_path, model, preprocess, device,
                              frame_skip=1, batch_size=32):
    """Extract per-frame CLIP embeddings from a video."""
    from PIL import Image
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    frames_pil = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            # Convert BGR to RGB, then to PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            frames_pil.append(pil_img)
        frame_idx += 1
    cap.release()
    
    if len(frames_pil) < 5:
        return None
    
    # Batch inference
    all_embeddings = []
    for i in range(0, len(frames_pil), batch_size):
        batch_pil = frames_pil[i:i+batch_size]
        batch_tensor = torch.stack([preprocess(img) for img in batch_pil]).to(device)
        
        with torch.no_grad():
            features = model.encode_image(batch_tensor)
            # L2 normalize
            features = features / features.norm(dim=-1, keepdim=True)
        
        all_embeddings.append(features.cpu().numpy())
    
    embeddings = np.concatenate(all_embeddings, axis=0)  # (T, D)
    return embeddings


def compute_temporal_features(embeddings):
    """
    Compute temporal statistics on a sequence of semantic embeddings.
    
    Args:
        embeddings: np.ndarray of shape (T, D) - per-frame embeddings
    
    Returns:
        dict of feature_name -> value
    """
    T, D = embeddings.shape
    features = {}
    
    # === 1. Cosine Similarity Sequence ===
    # Adjacent-frame cosine similarity
    cos_sims = np.array([
        np.dot(embeddings[i], embeddings[i+1])
        for i in range(T - 1)
    ])
    
    features["sem_cos_mean"] = np.mean(cos_sims)
    features["sem_cos_std"] = np.std(cos_sims)
    features["sem_cos_min"] = np.min(cos_sims)
    features["sem_cos_max"] = np.max(cos_sims)
    features["sem_cos_range"] = np.max(cos_sims) - np.min(cos_sims)
    features["sem_cos_median"] = np.median(cos_sims)
    
    # Percentiles
    features["sem_cos_p5"] = np.percentile(cos_sims, 5)
    features["sem_cos_p10"] = np.percentile(cos_sims, 10)
    features["sem_cos_p25"] = np.percentile(cos_sims, 25)
    
    # === 2. Cosine Similarity Gradient (Jump Detection) ===
    cos_gradient = np.diff(cos_sims)  # first derivative
    features["sem_grad_max"] = np.max(np.abs(cos_gradient))
    features["sem_grad_std"] = np.std(cos_gradient)
    features["sem_grad_mean_abs"] = np.mean(np.abs(cos_gradient))
    
    # Second derivative (acceleration of similarity changes)
    if len(cos_gradient) > 1:
        cos_accel = np.diff(cos_gradient)
        features["sem_accel_max"] = np.max(np.abs(cos_accel))
        features["sem_accel_std"] = np.std(cos_accel)
    else:
        features["sem_accel_max"] = 0.0
        features["sem_accel_std"] = 0.0
    
    # === 3. Embedding Velocity ===
    # L2 distance between consecutive embeddings
    velocities = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
    features["sem_vel_mean"] = np.mean(velocities)
    features["sem_vel_std"] = np.std(velocities)
    features["sem_vel_max"] = np.max(velocities)
    features["sem_vel_range"] = np.max(velocities) - np.min(velocities)
    features["sem_vel_cv"] = np.std(velocities) / (np.mean(velocities) + 1e-8)  # coefficient of variation
    
    # === 4. Stasis Detection ===
    # Fraction of consecutive frame pairs with cosine sim > 0.999 (near-identical)
    stasis_threshold = 0.999
    features["sem_stasis_ratio"] = np.mean(cos_sims > stasis_threshold)
    features["sem_stasis_max_run"] = max_consecutive_true(cos_sims > stasis_threshold)
    
    # Also at a looser threshold
    features["sem_near_stasis_ratio"] = np.mean(cos_sims > 0.995)
    
    # === 5. Trajectory Reversibility ===
    # Compare forward and backward temporal patterns
    # If a segment is reversed, the forward and backward patterns differ
    mid = T // 2
    if mid > 2:
        first_half_sims = cos_sims[:mid]
        second_half_sims = cos_sims[mid:]
        # Asymmetry between halves
        min_len = min(len(first_half_sims), len(second_half_sims))
        if min_len > 0:
            features["sem_asymmetry"] = np.mean(np.abs(
                first_half_sims[:min_len] - second_half_sims[:min_len]
            ))
        else:
            features["sem_asymmetry"] = 0.0
        
        # Autocorrelation-based reversibility
        if len(cos_sims) > 3:
            fwd = np.correlate(cos_sims, cos_sims[::-1], mode='valid')[0]
            bwd = np.correlate(cos_sims, cos_sims, mode='valid')[0]
            features["sem_reversibility"] = fwd / (bwd + 1e-8)
        else:
            features["sem_reversibility"] = 1.0
    else:
        features["sem_asymmetry"] = 0.0
        features["sem_reversibility"] = 1.0
    
    # === 6. Spectral Analysis ===
    # FFT of cosine similarity sequence
    if len(cos_sims) >= 8:
        fft_vals = np.abs(np.fft.rfft(cos_sims - np.mean(cos_sims)))
        n_freqs = len(fft_vals)
        mid_freq = n_freqs // 2
        
        low_energy = np.sum(fft_vals[:mid_freq] ** 2)
        high_energy = np.sum(fft_vals[mid_freq:] ** 2)
        total_energy = low_energy + high_energy + 1e-8
        
        features["sem_spectral_high_ratio"] = high_energy / total_energy
        features["sem_spectral_centroid"] = np.sum(np.arange(n_freqs) * fft_vals) / (np.sum(fft_vals) + 1e-8)
        features["sem_spectral_entropy"] = -np.sum(
            (fft_vals / (np.sum(fft_vals) + 1e-8) + 1e-10) *
            np.log(fft_vals / (np.sum(fft_vals) + 1e-8) + 1e-10)
        )
    else:
        features["sem_spectral_high_ratio"] = 0.0
        features["sem_spectral_centroid"] = 0.0
        features["sem_spectral_entropy"] = 0.0
    
    # === 7. PCA Temporal Variance ===
    # How many principal components explain the temporal variation?
    if T >= 5:
        from sklearn.decomposition import PCA
        n_components = min(5, T, D)
        pca = PCA(n_components=n_components)
        pca.fit(embeddings)
        features["sem_pca_var_1"] = pca.explained_variance_ratio_[0]
        features["sem_pca_var_cumul_3"] = np.sum(pca.explained_variance_ratio_[:min(3, n_components)])
        features["sem_pca_n_components_90"] = np.searchsorted(
            np.cumsum(pca.explained_variance_ratio_), 0.90
        ) + 1
    else:
        features["sem_pca_var_1"] = 1.0
        features["sem_pca_var_cumul_3"] = 1.0
        features["sem_pca_n_components_90"] = 1
    
    # === 8. Outlier Frame Detection ===
    # How many frames are "outliers" in embedding space?
    mean_emb = np.mean(embeddings, axis=0)
    distances = np.linalg.norm(embeddings - mean_emb, axis=1)
    threshold = np.mean(distances) + 2 * np.std(distances)
    features["sem_outlier_ratio"] = np.mean(distances > threshold)
    features["sem_distance_std"] = np.std(distances)
    features["sem_distance_skew"] = float(
        np.mean(((distances - np.mean(distances)) / (np.std(distances) + 1e-8)) ** 3)
    )
    
    return features


def max_consecutive_true(arr):
    """Find the longest run of True values in a boolean array."""
    max_run = 0
    current_run = 0
    for val in arr:
        if val:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def find_videos(input_dir):
    """Find all mp4 files in authentic/ and tampered/ subdirectories."""
    videos = []
    for subdir in ["authentic", "tampered"]:
        dirpath = os.path.join(input_dir, subdir)
        if not os.path.isdir(dirpath):
            continue
        for fname in sorted(os.listdir(dirpath)):
            if fname.lower().endswith(".mp4"):
                label = 0 if subdir == "authentic" else 1
                videos.append((subdir, fname, os.path.join(dirpath, fname), label))
    return videos


def main():
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print(" Semantic Temporal Consistency Extractor")
    print(f" Model: {args.model} ({args.pretrained})")
    print(f" Device: {device}")
    print(f" Frame skip: {args.frame_skip}")
    print(f" Batch size: {args.batch_size}")
    print("=" * 60)
    
    # Load model
    print("\n[STC] Loading CLIP model...")
    model, preprocess = load_clip_model(args.model, args.pretrained, device)
    print(f"[STC] Model loaded on {device}")
    
    # Find videos
    videos = find_videos(args.input)
    
    # Filter to manifest
    if os.path.exists(args.manifest):
        manifest = pd.read_csv(args.manifest)
        manifest_names = set(manifest['clip_name'])
        scene_map = dict(zip(manifest['clip_name'], manifest['scene_id']))
        videos = [v for v in videos if v[1] in manifest_names]
    else:
        scene_map = {}
    
    print(f"[STC] Found {len(videos)} videos\n")
    
    os.makedirs(args.out, exist_ok=True)
    
    rows = []
    skipped = 0
    
    for subdir, fname, video_path, label in tqdm(videos, desc="Extracting"):
        cache_path = os.path.join(args.out, fname.replace(".mp4", ".npz"))
        
        if args.resume and os.path.exists(cache_path):
            # Load cached
            data = np.load(cache_path, allow_pickle=True)
            feat_dict = data["features"].item()
            feat_dict["video_name"] = fname
            feat_dict["label"] = label
            feat_dict["scene_id"] = scene_map.get(fname, "unknown")
            rows.append(feat_dict)
            skipped += 1
            continue
        
        # Extract embeddings
        embeddings = extract_frame_embeddings(
            video_path, model, preprocess, device,
            frame_skip=args.frame_skip, batch_size=args.batch_size
        )
        
        if embeddings is None:
            print(f"  [SKIP] {fname} (no frames)")
            continue
        
        # Compute temporal features
        feat_dict = compute_temporal_features(embeddings)
        
        # Save cache
        np.savez_compressed(cache_path, features=feat_dict, embeddings=embeddings)
        
        feat_dict["video_name"] = fname
        feat_dict["label"] = label
        feat_dict["scene_id"] = scene_map.get(fname, "unknown")
        rows.append(feat_dict)
    
    # Save CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(ROOT, "results", f"features_semantic_{os.path.basename(args.out)}.csv")
    df.to_csv(csv_path, index=False)
    
    sem_cols = [c for c in df.columns if c.startswith("sem_")]
    print(f"\n[DONE] Extracted {len(rows)} videos ({skipped} cached)")
    print(f"       Semantic features: {len(sem_cols)}")
    print(f"       CSV: {csv_path}")


if __name__ == "__main__":
    main()
