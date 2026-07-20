# CoheRex 2.0 — Step-by-Step Design & Build Guide

Companion to `CoheRex_Audit_and_Remediation_Plan.md`. This is the *constructive* half:
the sharpened idea, the novel method, the dataset, the exact code changes (file by file),
and the evaluation protocol that produces a credible, publishable result.

> **Set expectations honestly.** Classic inter-frame forgery detection (frame
> delete/duplicate/insert) is near-saturated — pixel/optical-flow/compression methods already
> report AUC 0.99+. CoheRex will **not** win that race and should not try. Its contribution is a
> **recompression-robust, codec-agnostic, object-motion-physics** detector with **temporal
> localization** that catches what pixel/compression forensics miss. The headline result is a
> **robustness gap under re-encoding**, not a bigger clean-video AUC.

---

## STEP 1 — Define the idea precisely (the thesis)

**One-sentence thesis.**
Temporal video manipulation violates the *physical inertia* of tracked objects; therefore tampering
can be detected — and localized in time — as a statistically significant departure of object-motion
residuals from the distribution predicted by a physics (constant-velocity/acceleration) dynamics
model, without any manipulation-specific training, any spatial-artifact cue, or any codec assumption.

**Four novelty claims (state exactly these, and no more):**
1. **Physics-consistency hypothesis testing.** Reframe temporal forensics as a filter-consistency
   test: the Kalman *innovation* sequence is χ²-distributed and white under authentic motion; edits
   break this. This yields a **training-free, interpretable, threshold-principled** detector.
2. **Temporal localization.** A change-point test (CUSUM) on motion residuals localizes the tamper
   frame, not just a whole-video yes/no.
3. **Pose-rhythm channel.** A gait-periodicity signal derived from pose keypoints — an independent
   physics cue inaccessible to motion-vector/optical-flow forensics.
4. **Recompression robustness.** Because the signal is trajectory-level, it survives re-encoding that
   destroys pixel/compression footprints — demonstrated as the headline experiment.

**Be honest about prior art.** NIS/whiteness (target tracking), CUSUM (change-point detection), and
gait analysis are individually known. The contribution is the **synthesis + forensic application +
localization + robustness study**. Say so; reviewers respect it.

---

## STEP 2 — The method (CoheRex 2.0), stage by stage

Keep Stages 1–2 (detection + tracking) — they already work. Replace Stages 3+ with principled signals.

### Stage 1 — Detection & tracking (KEEP)
YOLOv8 person detection → Kalman (constant-velocity) → Hungarian association. No change except
**exposing the Kalman innovation** (see code map).

### Stage 2 — Physics residual per track (NEW core)
For each track, at each frame after `update(z)`:
- innovation `yₜ = zₜ − H x̂ₜ` (filterpy: `kf.y`)
- innovation covariance `Sₜ` (filterpy: `kf.S`)
- **Normalized Innovation Squared** `NISₜ = yₜᵀ Sₜ⁻¹ yₜ`

Under authentic constant-velocity motion, `NISₜ ~ χ²(2)`. Critical value at 95%: `χ²₀.₉₅(2) ≈ 5.99`.

### Stage 3 — Three principled test statistics per track
- **(a) NIS consistency:** `nis_exceed_frac = mean(NISₜ > 5.99)`, `nis_max = max(NISₜ)`.
- **(b) Whiteness:** lag-1 autocorrelation of innovation magnitude (≈0 authentic; >0 under tamper).
- **(c) CUSUM change-point:** Page's test on standardized `‖yₜ‖` → `(cusum_peak, cusum_frame)`.

### Stage 4 — Pose-rhythm channel (NEW; uses existing pose)
Per track: build a gait signal from keypoints (ankle–hip vertical oscillation, torso-normalized) →
estimate local period via autocorrelation → **rhythm_residual** = variability of the local period
(spikes at duplication/deletion/speed edits).

### Stage 5 — Scene aggregation + two detectors
Aggregate per-track statistics to **video level** (max and mean over tracks), plus a scene-level
**track-count discontinuity** (for splice). Then provide:
- **Unsupervised detector (training-free):** flag a video if `max_cusum_peak` or `nis_exceed_frac`
  exceeds a χ²-derived / percentile threshold. Report AUC + localization with **zero training** —
  this is what makes the 38→small-N problem irrelevant.
- **Learned fusion (optional):** `LogisticRegression` (preferred over RF for small-N stability &
  interpretability) over the principled statistics, evaluated with **GroupKFold** + bootstrap CI.

### Stage 6 — Localization output
Because CUSUM returns a frame index, emit a predicted tamper window and score it against ground truth
(temporal IoU / hit-rate). New evaluation axis; strong for publication.

---

## STEP 3 — Build the dataset (the blocker fix)

### 3.1 Get multi-scene source video
**Primary: VIRAT Ground 2.0** — 329 stationary-camera surveillance videos across **11 scenes**,
25–30 FPS, 1080p; matches your static-camera assumption. Access via click-through agreement:
- Landing: `https://viratdata.org`
- Data: `data.kitware.com` (VIRAT Release 2.0 collection).

**Alternative/supplement: SULFA** (standard forgery source clips) or your own multi-scene captures.
Whatever you pick, the rule is **many independent scenes**, not 2.

### 3.2 New script `scripts/download_dataset.py`
Downloads/organizes N source videos into `data/sources/<scene_id>/*.mp4`. (VIRAT requires the
agreement, so this script can just *verify/organize* files you place there and record a `sources.csv`
with `source_id, scene_id, path`.)

### 3.3 Rewrite `scripts/create_tampered_videos.py` → `scripts/build_dataset.py`
- Iterate over **each** source independently (`source_id` = group key).
- For each source, cut authentic clips **and** produce tampered clips *from that same source*, but
  record `source_id` on every clip so CV can keep a source out of both train and test.
- Log a **manifest** `data/evaluation/manifest.csv` with:
  `clip_name, source_id, scene_id, label, tamper_type, tamper_start_frame, tamper_end_frame`.
- Fix segment lengths to match what the paper claims, per tamper type.
- Target **≥150–300 clips** across ≥10 scenes, balanced by class and tamper type.

### 3.4 Recompression variant (for the headline experiment)
For every tampered/authentic clip, also emit a **re-encoded** copy (e.g., `ffmpeg -c:v libx264
-crf 32`) into `data/evaluation_recompressed/`. This is what breaks the baselines.

---

## STEP 4 — Code correction map (exactly where to change what)

Legend: **[EDIT]** change existing file · **[NEW]** create file · **[DELETE]** remove code.

| File | Action | What to do |
|---|---|---|
| `coherex/tracking/kalman.py` | **[EDIT]** | After `create_kalman_filter`, ensure innovation is retrievable. filterpy exposes `kf.y` (residual) and `kf.S` (innovation cov) **after `kf.update(z)`**. No math change; just don't discard them. |
| `coherex/tracking/track.py` | **[EDIT]** | In `update(meas_x, meas_y)`, after `self.kf.update(...)`, store `self.last_innovation = self.kf.y.copy()`, `self.last_S = self.kf.S.copy()`, and compute `nis = float(self.kf.y.T @ inv(self.kf.S) @ self.kf.y)`. Append to `self.nis_history` / `self.innov_history` (plain lists, **not** capped deques). Add these lists in `__init__`. |
| `coherex/trajectory/state.py` | **[EDIT]** | Add unbounded (or large-window) `self.nis_history = []` and `self.innov_history = []`; append in `update(...)`. The existing `maxlen=10` deques are fine for MCV but **too short** for whiteness/CUSUM — keep NIS/innovation as full sequences. |
| `coherex/coherence/innovation.py` | **[NEW]** | NIS stats + whiteness (code below). |
| `coherex/coherence/changepoint.py` | **[NEW]** | CUSUM/Page's test (code below). |
| `coherex/coherence/pose_rhythm.py` | **[NEW]** | Gait-signal extraction + rhythm residual (code below). Uses the pose keypoints already produced in `evaluate_dataset.py` / `tracking/pose.py`. |
| `coherex/coherence/mcv.py` | **[EDIT]** | Demote MCV to an *auxiliary* channel (no longer the flagship). Either keep as one more feature, or delete. If kept: **document** the leave-one-out z-score and the `noise_floor`; make the engine default equal the config (currently engine=3 vs config=8). |
| `coherex/integrity/motion_agent.py` | **[EDIT] + [DELETE]** | **Delete** the undocumented "Motion-Triggered Latch" block (forces integrity to 0 for 10 frames) — it is unreported and distorts every downstream feature. Base the motion score on `1 − min(NIS/χ²₀.₉₅, 1)` instead of ad-hoc `abs(mcv)/τ`. |
| `coherex/integrity/fusion_engine.py` | **[EDIT]** | Decide 2-agent vs 3-agent+reliability and make it match the paper. If keeping reliability weighting, **document the exact formula** and report effective weights. Default `fusion_weights=(0.7,0.3,0.0)` → either use it and state α=0.7/β=0.3, or re-tune with the new signals. |
| `coherex/integrity/crowd_agent.py`, `reliability.py` | **[DECIDE]** | Keep + document, or remove. Do not leave them running-but-unmentioned. |
| `coherex/meta/feature_extractor.py` | **[EDIT]** | Replace the ad-hoc set with the **principled feature set** (one canonical list, matching the paper exactly): `nis_max, nis_mean, nis_exceed_frac, whiteness_lag1, cusum_peak, pose_rhythm_residual, track_count_discontinuity` (+ optionally `motion_min` as legacy). |
| `scripts/evaluate_dataset.py` | **[EDIT] + [DELETE]** | (1) Compute the new per-frame stats and per-track aggregates; (2) **read `manifest.csv`** so each row carries `source_id` and ground-truth tamper window; (3) **delete** the `all_crowd_scores = []` duplicate line; (4) **delete** the "late-birth" `m_score *= 0.5` heuristic (undocumented); (5) decide whether to keep SSIM/`flow_var` — if pure-physics, move them to a separate `baseline_features.csv`, not the CoheRex feature set. |
| `scripts/train_meta_classifier.py` | **[EDIT]** | Switch `StratifiedKFold` → **`GroupKFold(groups=source_id)`**. Report **pooled OOF AUC + bootstrap 95% CI** as primary (not mean-of-tiny-folds). Prefer `LogisticRegression(class_weight="balanced")`. Move correlation pruning **inside** the CV loop (or drop it). Wire the *actual* run's numbers into a single `results/metrics.json` that the paper reads from. |
| `scripts/unsupervised_detector.py` | **[NEW]** | Training-free detector: threshold `max_cusum_peak`/`nis_exceed_frac`; output AUC + confusion matrix. This is the small-N-proof result. |
| `scripts/localize_eval.py` | **[NEW]** | Compare predicted CUSUM frame vs `manifest` tamper window → localization hit-rate & temporal IoU. |
| `scripts/download_dataset.py`, `scripts/build_dataset.py` | **[NEW]** | See Step 3. |
| repo root | **[EDIT]** | Proper `.gitignore` (`__pycache__/`, `*.pyc`, `data/**/*.mp4`, `*.pt`, `data/models/*`); pin versions in `requirements.txt`; add a `results/` dir the paper reads from. |

---

## STEP 5 — Drop-in code for the new modules

> These are correct, standard implementations to adapt. Tune constants against your data.

### `coherex/coherence/innovation.py`
```python
import numpy as np
from scipy.stats import chi2

CHI2_95_DOF2 = float(chi2.ppf(0.95, df=2))  # ~5.991

def nis(y, S):
    y = np.asarray(y, float).reshape(-1)
    S = np.asarray(S, float)
    return float(y @ np.linalg.inv(S) @ y)

def nis_stats(nis_values):
    a = np.asarray(nis_values, float)
    if a.size == 0:
        return {"nis_mean": 0.0, "nis_max": 0.0, "nis_exceed_frac": 0.0}
    return {
        "nis_mean": float(a.mean()),
        "nis_max": float(a.max()),
        "nis_exceed_frac": float((a > CHI2_95_DOF2).mean()),
    }

def whiteness_lag1(innovations):
    """innovations: list of 2D residual vectors. Returns |lag-1 autocorr| of magnitude."""
    m = np.array([np.linalg.norm(np.asarray(v, float)) for v in innovations], float)
    if m.size < 3:
        return 0.0
    m = m - m.mean()
    denom = float(np.sum(m * m)) + 1e-9
    r1 = float(np.sum(m[:-1] * m[1:]) / denom)
    return abs(r1)
```

### `coherex/coherence/changepoint.py`
```python
import numpy as np

def cusum_peak(signal, drift=0.5):
    """Two-sided CUSUM (Page's test) on a signal. Returns (peak_statistic, peak_index)."""
    x = np.asarray(signal, float)
    if x.size < 3:
        return 0.0, -1
    mu, sd = float(x.mean()), float(x.std()) + 1e-9
    z = (x - mu) / sd
    s_pos = np.zeros_like(z)
    s_neg = np.zeros_like(z)
    for t in range(1, len(z)):
        s_pos[t] = max(0.0, s_pos[t - 1] + z[t] - drift)
        s_neg[t] = max(0.0, s_neg[t - 1] - z[t] - drift)
    s = np.maximum(s_pos, s_neg)
    idx = int(np.argmax(s))
    return float(s[idx]), idx
```

### `coherex/coherence/pose_rhythm.py`
```python
import numpy as np

# COCO-17 keypoint indices used by yolov8-pose
L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = 5, 6, 11, 12, 15, 16

def gait_signal(keypoint_series):
    """keypoint_series: list of (17,2) arrays (or None) for one track over frames."""
    sig = []
    for kp in keypoint_series:
        if kp is None or len(kp) <= R_ANK:
            sig.append(np.nan); continue
        hips = 0.5 * (kp[L_HIP] + kp[R_HIP])
        torso = np.linalg.norm(kp[L_SH] - kp[L_HIP]) + 1e-6
        val = (abs(kp[L_ANK][1] - hips[1]) + abs(kp[R_ANK][1] - hips[1])) / torso
        sig.append(val)
    return np.array(sig, float)

def rhythm_residual(sig, win=15):
    """Variability of local gait period; spikes at stutter/phase/frequency breaks."""
    g = np.asarray(sig, float)
    g = g[~np.isnan(g)]
    if g.size < 2 * win:
        return 0.0
    periods = []
    for s in range(0, len(g) - win, max(1, win // 2)):
        w = g[s:s + win] - np.mean(g[s:s + win])
        ac = np.correlate(w, w, mode="full")[win - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        for l in range(2, len(ac) - 1):
            if ac[l] > ac[l - 1] and ac[l] > ac[l + 1]:
                periods.append(l); break
    if len(periods) < 2:
        return 0.0
    return float(np.std(periods) / (np.mean(periods) + 1e-6))
```

### `scripts/train_meta_classifier.py` (core of the fix)
```python
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

def grouped_oof(X, y, groups, n_splits=5, seed=42):
    gkf = GroupKFold(n_splits=n_splits)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    proba = cross_val_predict(clf, X, y, groups=groups, cv=gkf,
                              method="predict_proba")[:, 1]
    auc = roc_auc_score(y, proba)
    rng = np.random.default_rng(seed); boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], proba[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, (float(lo), float(hi)), proba
```
> `GroupKFold` needs `n_splits ≤ number of groups`. With ≥10 scenes this is fine; it guarantees no
> source appears in both train and test — killing the leakage that produced your 1.00-AUC folds.

---

## STEP 6 — Run order (parse videos → results)

```bash
# 0. Environment
pip install -r requirements.txt          # pin: ultralytics, scikit-learn, opencv-python,
                                          # filterpy, scikit-image, scipy, numpy, pandas

# 1. Dataset (place VIRAT sources under data/sources/<scene>/, then:)
python scripts/build_dataset.py           # writes clips + data/evaluation/manifest.csv
python scripts/build_dataset.py --recompress   # writes data/evaluation_recompressed/

# 2. Feature extraction (parse every video through the pipeline)
python scripts/evaluate_dataset.py --input data/evaluation \
       --output results/features_clean.csv
python scripts/evaluate_dataset.py --input data/evaluation_recompressed \
       --output results/features_recompressed.csv

# 3a. Training-free result (no small-N problem)
python scripts/unsupervised_detector.py --csv results/features_clean.csv

# 3b. Learned, leak-free result
python scripts/train_meta_classifier.py --csv results/features_clean.csv \
       --group-col source_id --model logreg

# 4. Localization
python scripts/localize_eval.py --csv results/features_clean.csv \
       --manifest data/evaluation/manifest.csv

# 5. Headline robustness experiment (clean vs recompressed, CoheRex vs baselines)
python scripts/robustness_eval.py --clean results/features_clean.csv \
       --recompressed results/features_recompressed.csv
```

---

## STEP 7 — Evaluation protocol for a research-grade result

Report **all** of these; the *combination* is what makes it publishable, not any single number.

1. **Leak-free detection.** Pooled OOF AUC + bootstrap 95% CI under **GroupKFold**. Also report
   the **training-free** detector AUC (headline honesty: works with no training data).
2. **Baselines on the same data.** Implement 2–3 real competitors you already have the pieces for:
   optical-flow-variation detector (Farnebäck `flow_var` you already compute), SSIM-drift detector
   (`ssim_mean`/frame-difference), and a majority/chance baseline. CoheRex must be reported *relative*
   to these.
3. **Ablation.** motion-NIS only → +whiteness → +CUSUM → +pose-rhythm → +learned fusion. Use the
   `--agents` mechanism you already built.
4. **Per-tamper-type breakdown.** Detection rate for deletion / duplication / speed / splice /
   reverse. Be honest that interior-of-reversal is hard (only seams are detectable).
5. **Temporal localization.** Hit-rate (predicted change-point ∈ ground-truth window) + mean temporal
   IoU. New axis, strong.
6. **Headline: recompression robustness.** Table of ΔAUC (clean → recompressed) for each method.
   Expectation: baselines drop sharply; CoheRex barely moves. *This is the paper's money figure.*

---

## STEP 8 — Rewrite the paper around the real result

- **Title/abstract:** foreground recompression-robust, codec-agnostic, physics-based temporal
  forensics with localization. Drop the unreproducible 0.7507.
- **Method:** derive NIS/whiteness/CUSUM/pose-rhythm from first principles; one feature list;
  every hyperparameter and heuristic documented; correct all equations.
- **Experiments:** the 6-part protocol above, all numbers from one saved `results/metrics.json`.
- **Related work:** position against optical-flow / histogram / compression-footprint inter-frame
  forensics; state the recompression weakness those share and that CoheRex resolves.
- **Limitations:** static camera, person-centric, reversal interior, short-window sensitivity.
- **References:** verify every entry; remove fabricated ones (e.g., the future-dated arXiv id).

---

## Realistic outcome
On clean video, expect CoheRex to be **competitive-but-not-best** (maybe 0.80–0.92 AUC on a good
dataset — real, not inflated). The publishable win is: **(i)** it needs no training, **(ii)** it
localizes the edit, **(iii)** it uses a signal others don't (gait rhythm), and **(iv)** it barely
degrades under recompression while pixel/compression baselines fall apart. That is a genuine,
defensible contribution — far stronger than a lone AUC number, and much harder for a reviewer to
reject.
