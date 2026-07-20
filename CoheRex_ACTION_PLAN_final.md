# CoheRex — Action Plan v2 (measured diagnosis → fastest publishable path)

Based on your uploaded `coherex.zip` and its actual results. Supersedes earlier optimism with what
the data actually shows.

---

## 1 — What the data says (measured, not guessed)

**Single-feature separability on your `features_clean_v6.csv` (177 clips, 6 scenes):**
| Feature | \|AUC\| |
|---|---|
| nis_peak / nis_relative_spike | 0.66 |
| glr_peak / nis_mad_zscore | 0.65 |
| _legacy_motion_min | 0.69 (best) |
| whiteness_lag1 | 0.60 |
| spec_entropy / dom_freq_stab (pose) | 0.54 / 0.55 |
| tc_max_jump / tc_jump_frac (splice) | **0.51 / 0.50 (dead)** |
| _baseline_ssim_mean / _baseline_flow_var | **0.50 / 0.50 (dead)** |

**Conclusions:**
1. **Pure physics tops out ~0.70 AUC here.** Best feature 0.69, all others 0.6–0.66, mutually
   correlated → no honest classifier reaches 0.85 from these. This is a ceiling, not a tuning gap.
2. **The splice channel (`tc_*`) is dead** → that's why `clip_splice` per-type AUC ≈ 0.46 (chance).
3. **The pixel baselines are dead — because they're mean-aggregated.** `evaluate_dataset.py` stores
   `ssim_mean` and `flow_var` (means over all frames). A deleted/duplicated frame is a **spike**, not
   a mean shift; averaging destroys it. The strong signal exists but is being thrown away.
4. **Speed cause found:** `evaluate_dataset.py` re-runs YOLO + pose + Farnebäck flow + SSIM per frame
   on every video every run. That's the 3–4 h loop. None of it changes between feature experiments.
5. **Class imbalance 52 authentic : 125 tampered** → classifiers collapse to "predict tampered"
   (specificity 0.0–0.35 in your runs).
6. **Only 6 scenes**, 25 source_ids → GroupKFold on `source_id` still shares scenes across folds
   (mild leakage inflating the number). Group by `scene_id`.

---

## 2 — The decision (pick the path)

### ✅ RECOMMENDED — Path H: Hybrid detector, robustness headline
Make the **pixel-residual signal** the accuracy engine (fix the extreme-value aggregation → expected
high clean AUC), keep **physics** as the robustness engine, and make the **contribution the
robustness gap**:
> "Pixel temporal forensics is accurate but collapses under recompression; motion-physics is robust;
> the hybrid stays accurate *and* robust." + temporal localization + training-free physics detector.

- **Clean AUC:** high (pixel signal; SOTA-level for deletion/duplication).
- **Novelty:** the robustness gap, physics-as-robustness-engine, localization — genuinely publishable.
- **Uses everything you built.** Physics work becomes the robustness half.

### ⚠️ FALLBACK — Path P: Pure physics, robustness/localization paper
Keep physics-only (ceiling ~0.70–0.78 after the SNR fixes below), and make the paper a **robustness +
localization** study where a modest-but-stable AUC beats collapsing baselines. Publishable at a
workshop, but the headline number stays modest. Lower reward, less rewriting.

**Honest statement:** no one can guarantee AUC ≥ 0.85 before running it, and your data shows pure
physics won't reach it. Path H is your realistic route to a *high headline number* **and** a
defensible paper. Recommend Path H.

---

## 3 — FIX #1 (do first): kill the 3–4 h loop with a cache

Split the pipeline into **extract-once** and **build-features-fast**.

### NEW `scripts/extract_raw.py` — run the heavy stuff ONCE per video
Runs YOLO + pose + tracking + Kalman innovation + the pixel-residual *sequences*, saves one `.npz`:
```python
import numpy as np, cv2
# ... run detector/tracker/pose over frames ONCE ...
np.savez_compressed(f"cache/{name}.npz",
    # --- pixel-residual SEQUENCES (keep the whole sequence, NOT the mean) ---
    frame_diff = np.asarray(frame_diff_seq, np.float32),   # mean|Δpixel| per adjacent pair
    ssim_drop  = np.asarray(ssim_drop_seq,  np.float32),   # 1 - SSIM per adjacent pair
    flow_spike = np.asarray(flow_var_seq,   np.float32),   # Farnebäck flow variance per frame
    # --- physics: per-track innovation sequences (object arrays or padded) ---
    nis_cv = nis_cv_arr, nis_ca = nis_ca_arr, innov = innov_arr,
    track_len = track_len_arr, track_bbox = bbox_arr,
    # --- pose keypoints per track (for gait) ---
    pose = pose_arr,
    # --- metadata ---
    label=label, tamper_type=ttype, tamper_start=t0, tamper_end=t1,
    source_id=sid, scene_id=scid, fps=fps)
```
Run it once for clean **and** once for each recompression level → `cache/clean/`, `cache/crf28/`, …

### PIVOT `scripts/evaluate_dataset.py` → `scripts/build_features.py`
Reads the `.npz` cache and computes the feature CSV in **seconds** (no YOLO, no flow). This is where
you iterate. Every feature experiment now costs seconds, not hours.

> Net effect: one-time 3–4 h extraction, then unlimited fast feature/model iteration.

---

## 4 — FIX #2 (accuracy): extreme-value PIXEL features (the high-AUC win)

In `build_features.py`, apply extreme-value aggregation to the pixel sequences (same idea you used for
NIS, never applied to pixels):
```python
def extreme(seq, thr):
    a = np.asarray(seq, float); a = a[np.isfinite(a)]
    if a.size == 0: return dict(peak=0., topk=0., frac=0., run=0)
    k = max(1, a.size//10)
    over = a > thr
    run = mx = 0
    for v in over: run = run+1 if v else 0; mx = max(mx, run)
    return dict(peak=float(a.max()),
                topk=float(np.sort(a)[::-1][:k].mean()),
                frac=float(over.mean()), run=int(mx))

# strong accuracy features:
fd = extreme(npz['frame_diff'], thr=np.percentile(npz['frame_diff'], 95))
sd = extreme(npz['ssim_drop'],  thr=np.percentile(npz['ssim_drop'], 95))
# fd['peak'], fd['topk'], sd['peak'], ... -> expect 0.9+ AUC on deletion/duplication
```
Also add the **first-difference of the frame-diff sequence** (`np.diff`) and take its peak — the
sharpest deletion/insertion detector.

> These pixel-extreme features are the SOTA signal for inter-frame forgery. Verify they hit ~0.9 on
> deletion/duplication before anything else — that's your clean-video headline.

---

## 5 — FIX #3 (physics SNR): squeeze physics as high as it honestly goes

In the extractor/tracker:
1. **Use the feet-point** (bbox bottom-center) as the Kalman measurement, not the centroid — far more
   stable for standing/walking people; centroid jitters with pose/aspect changes. Edit
   `coherex/tracking/track.py` / wherever the measurement is set.
2. **Track-quality gating:** only tracks with `length ≥ 15` frames and `bbox_height ≥ 60 px`
   contribute to physics features. Short/tiny tracks are pure detection noise. (Filter in
   `build_features.py`.)
3. **Multi-step prediction residual (new, stronger physics signal):** from a state a few frames before
   frame t, predict K frames ahead and measure the position error. A deletion of D frames yields an
   error ≈ speed × D — a much larger, recompression-robust signal than 1-step NIS. Add as
   `predK_peak`. This is the physics feature most likely to break past 0.70.

---

## 6 — FIX #4 (evaluation honesty): imbalance, grouping, splice

- **Balance classes:** subsample tampered to 52 (matching authentic), or `class_weight="balanced"`
  and report **balanced accuracy** + per-type AUC. Don't report raw accuracy on 52:125.
- **Group by `scene_id` (6 groups), not `source_id`,** to remove the residual scene leakage. With 6
  scenes use 6-fold leave-one-scene-out. (Better still: get to ~10–12 scenes from VIRAT.)
- **Splice:** the `tc_*` track-count channel is dead. Replace with an **appearance-discontinuity**
  signal — per-frame color-histogram distance (Bhattacharyya) between adjacent frames; a splice causes
  a spike. Add `appear_peak` via `extreme()`. Or, if it stays weak, report splice honestly as the hard
  case (it often is for physics).

---

## 7 — File-by-file change list

| File | Action | Change |
|---|---|---|
| `scripts/extract_raw.py` | **NEW** | Heavy extraction once → `.npz` cache (detections, tracks, NIS_cv/ca, pose, **pixel-residual sequences**). |
| `scripts/evaluate_dataset.py` | **PIVOT → `build_features.py`** | Read cache, compute features in seconds. Remove all YOLO/flow/SSIM per-frame calls from the iteration path. |
| `coherex/meta/feature_extractor.py` | **EDIT** | Add pixel-extreme features (`fd_peak, fd_topk, sd_peak, fd_diff_peak, appear_peak`) and `predK_peak`. Tag channels: `pix_*` (pixel), `phys_*` (physics) so you can train pixel-only / physics-only / hybrid. |
| `coherex/tracking/track.py` | **EDIT** | Measurement = bbox **feet-point**, not centroid. |
| `scripts/build_features.py` | **EDIT** | Track-quality gating (len≥15, h≥60px); multi-step residual; appearance-discontinuity for splice. |
| `scripts/train_meta_classifier.py` | **EDIT** | Add `--feature-set {pixel,physics,hybrid}`; group by `scene_id`; `class_weight="balanced"`; report per-type AUC + balanced acc + bootstrap CI. |
| `scripts/robustness_eval.py` | **EDIT** | Run pixel-only / physics-only / hybrid across clean + CRF28/32/36 → **the headline table** (ΔAUC). |
| `scripts/build_dataset.py` | **EDIT** | Emit recompressed copies per clip; add ≥4–6 more VIRAT scenes if possible; keep manifest with `scene_id`. |
| `scripts/localize_eval.py` | **KEEP** | Already present — feed it the GLR/multi-step change-point frame. |
| `coherex/integrity/*`, `coherence/mcv.py`, legacy | **QUARANTINE** | Stop feeding `_legacy_*`, `birth/death_burst`, dead `tc_*` into the model unless they earn their place (they mostly don't). Keep code, drop from feature set. |

---

## 8 — Run order & realistic timeline

```bash
# Day 1 (one-time heavy run — do overnight)
python scripts/extract_raw.py --input data/evaluation           --out cache/clean
python scripts/extract_raw.py --input data/evaluation_recompressed --out cache/crf32   # etc.

# Day 1–2 (fast, iterate freely — seconds each)
python scripts/build_features.py --cache cache/clean --out results/feat_clean.csv
python scripts/train_meta_classifier.py --csv results/feat_clean.csv \
       --feature-set pixel   --group-col scene_id     # expect HIGH clean AUC (verify ~0.9)
python scripts/train_meta_classifier.py --csv results/feat_clean.csv \
       --feature-set physics --group-col scene_id     # ~0.70–0.78
python scripts/train_meta_classifier.py --csv results/feat_clean.csv \
       --feature-set hybrid  --group-col scene_id     # ≥ pixel

# Day 2 (headline)
python scripts/robustness_eval.py --clean cache/clean --recompressed cache/crf32
# -> table: pixel high→low, physics stable, hybrid high→high  == the paper's money figure
```

**Timeline to a submittable draft:** ~4–6 focused days (1 heavy extraction + 2 feature/model + 2
robustness/localization + 1 writing), assuming the pixel-extreme features hit ~0.9 as expected.

**Realistic venue:** a solid workshop or mid-tier journal (e.g., ICASSP/WIFS/IWDW/ICIP workshops;
*Multimedia Tools & Applications*, *Forensic Science Int.: Digital Investigation*). Not CVPR-main —
but a real, honest, peer-reviewed publication, which is the right target for this.

---

## 9 — What I can and can't promise
- **Can't:** a specific AUC (≥0.85) before it's run — your data shows pure physics won't reach it,
  and I won't inflate a number for you.
- **Can:** a path to a **high clean AUC** (via the pixel-extreme fix), a **genuine novel contribution**
  (robustness gap + localization + training-free physics), **fast iteration** (caching), and a
  **rigorous, publishable** paper. That is the best *real* outcome, and it's very achievable.
