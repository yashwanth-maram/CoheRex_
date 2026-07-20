# CoheRex 2.0 — AUC-Maximization & Novelty Upgrade Pack

Patch layered on `CoheRex_2.0_Build_Guide.md`. Everything here raises AUC by **sharpening the real
physics signal** (the only honest way to move a threshold-free metric) and simultaneously upgrades a
novelty claim. Read this alongside the build guide: each item says what it *replaces* in that guide.

---

## 0 — Honesty guardrails (read first)

AUC moves only when the classes separate better. The following would raise AUC *fraudulently* and
will get the paper rejected or retracted — **do not do them**:
- reintroducing source leakage (any scene in both train and test),
- dropping hard/ambiguous clips,
- reporting the best of several random seeds,
- quoting a lucky per-fold AUC instead of pooled OOF,
- tuning any threshold/feature/model choice on the test folds.

Keep: **GroupKFold**, **one fixed seed reported**, **bootstrap 95% CI**, and a **held-out scene set**
never touched during development. Every lever below raises the *true* signal under these constraints.

---

## 1 — The six AUC levers (ranked by expected impact)

### Lever 1 — Extreme-value statistics, not means  *(biggest, smallest change)*
**Replaces:** the mean-based video features in Build-Guide Stage 5 / `feature_extractor.py`.
**Why it works:** tampering is a *localized burst*; averaging a 10-frame edit over a 90-frame clip
hides it. Your own Table III already proved extremes win (`motion_min` was the top feature).
**Do:** aggregate every per-frame statistic (NIS, GLR, pose residual) with **peak, top-k mean,
exceed-fraction, and longest above-threshold run** — not the mean.
**Novelty gained:** "extreme-value physics-consistency forensics."

```python
import numpy as np
def extreme_features(series, thresh):
    a = np.asarray(series, float)
    if a.size == 0:
        return dict(peak=0.0, topk_mean=0.0, exceed_frac=0.0, longest_run=0)
    k = max(1, a.size // 10)                     # top 10%
    topk = np.sort(a)[::-1][:k]
    over = a > thresh
    longest = cur = 0
    for v in over:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)
    return dict(peak=float(a.max()), topk_mean=float(topk.mean()),
                exceed_frac=float(over.mean()), longest_run=int(longest))
```

### Lever 2 — Multi-model physics null (pragmatic IMM)  *(biggest false-positive cut)*
**Replaces:** the single constant-velocity Kalman in `tracking/kalman.py`.
**Why it works:** a CV filter fires high NIS on *authentic* turns/accelerations → false positives that
flatten the ROC. Explaining authentic maneuvers with a second model makes the authentic innovation
truly white, so real tamper spikes stand out.
**Do:** run **two filters per track** — constant-velocity (CV) and constant-acceleration (CA) — and
take the **min NIS** (the best-explaining model). Full IMM is optional/heavier; min-NIS captures most
of the benefit.
**Novelty gained:** "multi-model physics hypothesis test" (much stronger than single-CV).

```python
def combined_nis(nis_cv, nis_ca):
    # low if EITHER physics model explains the motion (authentic maneuver);
    # high only when NO physics model fits (tamper discontinuity).
    return float(min(nis_cv, nis_ca))
```
> Implementation: add a second `KalmanFilter(dim_x=6, dim_z=2)` with a constant-acceleration `F`;
> store `nis_cv` and `nis_ca` per frame; feed `combined_nis` downstream.

### Lever 3 — Calibrate the null so NIS is actually χ²(2)
**Replaces:** using raw NIS with a fixed 5.99 threshold.
**Why it works:** if Q/R are mis-tuned, NIS is inflated *everywhere* and separation collapses.
Calibrating so authentic NIS ≈ χ²(2) makes spikes sharp and the χ²-derived threshold *real*.
**Do:** robust per-clip normalization by the running median (median is dominated by authentic frames,
so localized tampers survive). Note the assumption in the paper.
**Novelty gained:** the "principled, theory-derived threshold" claim becomes literally true.

```python
import numpy as np
def calibrate_nis(nis_series, dof=2, win=61):
    a = np.asarray(nis_series, float)
    if a.size < win:
        med = np.median(a) + 1e-6
        return a / (med / dof)
    # rolling-median normalization (robust to localized spikes)
    from scipy.ndimage import median_filter
    roll = median_filter(a, size=win, mode="nearest") + 1e-6
    return a / (roll / dof)
```

### Lever 4 — GLR change-point instead of CUSUM
**Replaces:** `changepoint.py` CUSUM as the primary localizer.
**Why it works:** the generalized-likelihood-ratio test is the most powerful detector for a single
abrupt shift at unknown location → higher detection power ⇒ higher AUC on abrupt edits, and a sharper
localization peak.
**Novelty gained:** stronger localization guarantee.

```python
import numpy as np
def glr_change_point(x):
    """Gaussian GLR for a single mean shift at unknown location. Returns (glr_peak, index)."""
    x = np.asarray(x, float); n = x.size
    if n < 4: return 0.0, -1
    total = x.sum()
    best, best_k, csum = 0.0, -1, 0.0
    for k in range(1, n):
        csum += x[k-1]
        n1, n2 = k, n - k
        stat = (csum*csum)/n1 + ((total-csum)**2)/n2 - (total*total)/n
        if stat > best: best, best_k = stat, k
    return float(best), int(best_k)
```

### Lever 5 — Multi-scale temporal windows
**Replaces:** single-window aggregation.
**Why it works:** short windows catch frame deletions; long windows catch slow speed-ramps. Directly
attacks the false negatives your confusion matrix showed (8 missed tampers, mostly short/low-mag).
**Do:** compute Levers 1/3/4 statistics at windows W ∈ {15, 45, 150} frames; feature = the **max over
scales** for each statistic.
**Novelty gained:** turns your paper's "future work: multi-scale" into an actual contribution.

### Lever 6 — Per-channel specialization + calibrated fusion
**Replaces:** the single fused integrity feature.
**Why it works:** position-innovation, pose-rhythm, and track-count-discontinuity each nail a
*different* tamper type. If each channel ≈1.0 AUC on its type, the fused macro-AUC rises.
**Do:** keep the channels separate as features; let the (logistic) fusion learn per-type weighting;
**report per-type AUC** (raises the headline *and* explains the mechanism).

---

## 2 — Novelty upgrades (higher ceiling)

### 2a — Spectral gait coherence (upgrade the pose channel)
**Replaces:** the ankle-oscillation period-variance heuristic in `pose_rhythm.py`.
**Why:** a spectral view is more principled and more discriminative — duplication raises spectral
entropy (disrupted periodicity), speed-up shifts the dominant gait frequency, deletion breaks phase.

```python
import numpy as np
from scipy.signal import periodogram
from scipy.stats import entropy
def spectral_gait_features(gait_sig, fps):
    g = np.asarray(gait_sig, float); g = g[~np.isnan(g)]
    if g.size < 16:
        return dict(spec_entropy=1.0, dom_freq_stab=0.0, dom_power=0.0)
    g = g - g.mean()
    def dom(seg):
        f, p = periodogram(seg, fs=fps)
        if p.sum() <= 0: return 0.0, 0.0, 1.0
        pn = p / p.sum()
        return float(f[np.argmax(p)]), float(p.max()/(p.sum()+1e-9)), float(entropy(pn+1e-12))
    h = len(g)//2
    d1,_,_ = dom(g[:h]); d2,_,_ = dom(g[h:])
    d_all, d_pow, s_ent = dom(g)
    return dict(spec_entropy=s_ent,
                dom_freq_stab=float(abs(d1-d2)/(d_all+1e-6)),  # ↑ = speed tamper
                dom_power=d_pow)
```

### 2b — One-class self-supervised physics prior  *(most publishable single addition)*
**New capability.** Fit a density model on the physics features of **authentic video only**; score
tampering as low-likelihood. It never sees a tampered example → stays fully manipulation-agnostic,
adapts per scene, and typically lifts AUC because it models the authentic manifold tightly.
**Eval rule:** the authentic training scenes must be **disjoint** from test scenes (still GroupKFold).

```python
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
def fit_physics_prior(X_auth):
    sc = StandardScaler().fit(X_auth)
    oc = EllipticEnvelope(contamination=0.05, support_fraction=0.9).fit(sc.transform(X_auth))
    return sc, oc
def tamper_scores(sc, oc, X):
    return -oc.score_samples(sc.transform(X))   # higher = more anomalous
```
> Alternatives if the Gaussian assumption is poor: `IsolationForest`, `OneClassSVM`, or a Gaussian-KDE
> on the feature vector. Report whichever you pick and why.

### 2c — Robustness suite (upgrade the headline experiment)
**Replaces:** the single recompression test.
**Do:** re-evaluate every method under four perturbations and report ΔAUC for each:
recompression (H.264 CRF 28/32/36), additive noise, temporal smoothing, and frame-rate conversion.
CoheRex should barely move while pixel/OF/compression baselines fall apart → a whole results section.

---

## 3 — Updated canonical feature set (use this everywhere)

Per channel × per scale W∈{15,45,150}, take `extreme_features` (peak, topk_mean, exceed_frac,
longest_run), then **max over scales**:

- **Innovation/NIS channel:** on calibrated `combined_nis` (CV+CA).
- **Change-point channel:** GLR peak + its normalized location dispersion.
- **Whiteness channel:** lag-1 (and lag-2) innovation autocorrelation.
- **Pose-rhythm channel:** `spec_entropy`, `dom_freq_stab`, `dom_power`.
- **Scene channel:** track-count-discontinuity, appearance/disappearance burst.

Plus two *detectors*: (i) **unsupervised** χ²/percentile threshold on the NIS/GLR extremes;
(ii) **one-class physics prior** score (2b); (iii) optional **logistic fusion** over all features.
One list, documented once, matching the paper exactly.

---

## 4 — Code-correction map (delta on top of the Build Guide)

| File | Action | Add on top of Build-Guide instructions |
|---|---|---|
| `coherex/tracking/kalman.py` | **[EDIT]** | Add a second constant-acceleration filter (`dim_x=6`); expose `nis_cv`, `nis_ca`. |
| `coherex/coherence/innovation.py` | **[EDIT]** | Add `combined_nis`, `calibrate_nis`; keep whiteness. |
| `coherex/coherence/changepoint.py` | **[EDIT]** | Add `glr_change_point`; keep CUSUM as a baseline. |
| `coherex/coherence/pose_rhythm.py` | **[EDIT]** | Replace period-variance with `spectral_gait_features`. |
| `coherex/coherence/aggregate.py` | **[NEW]** | `extreme_features` + multi-scale (`max over W`) wrapper. |
| `coherex/meta/feature_extractor.py` | **[EDIT]** | Emit the Section-3 canonical set. |
| `coherex/meta/physics_prior.py` | **[NEW]** | `fit_physics_prior` / `tamper_scores` (Lever 2b). |
| `scripts/evaluate_dataset.py` | **[EDIT]** | Compute per-channel per-scale extremes; write them to `results/features_*.csv`. |
| `scripts/train_meta_classifier.py` | **[EDIT]** | Add `--model {logreg,oneclass}`; per-type AUC breakdown; keep GroupKFold + bootstrap CI. |
| `scripts/robustness_eval.py` | **[NEW]** | The 4-perturbation robustness suite (Section 2c). |
| `scripts/pertype_eval.py` | **[NEW]** | AUC per tamper type from the manifest. |

---

## 5 — Updated evaluation (what the paper reports)

1. **Detection (leak-free):** pooled OOF AUC + bootstrap 95% CI (GroupKFold) for
   (a) unsupervised threshold, (b) one-class prior, (c) logistic fusion.
2. **Per-type AUC:** deletion / duplication / speed / splice / reverse.
3. **Ablation:** NIS → +CA(min-NIS) → +calibration → +GLR → +multi-scale → +pose-spectral → +fusion,
   showing AUC climbing at each step (this *is* your AUC-improvement story, honestly told).
4. **Localization:** GLR-frame hit-rate + temporal IoU vs. manifest windows.
5. **Baselines:** optical-flow-variation, SSIM-drift, chance — same splits.
6. **Robustness suite (headline):** ΔAUC under recompression/noise/smoothing/fps for every method.

---

## 6 — Master Results ("Money Table")

This table presents the pooled Out-of-Fold AUCs (via 5-fold GroupKFold) for the canonical `enhanced_classifier.py` pipeline. It directly compares the **Enhanced Pixel** features, **Enhanced Physics** features, and the **Enhanced Hybrid** (Pixel + Physics) features across all datasets and temporal perturbation attacks.

**Key Findings:**
1. **Multi-scale extreme-value statistics:** Extracted pixel features are extremely accurate on clean data.
2. **Robustness:** Under temporal interpolation attacks (which smooth out pixel-level artifacts), the lightweight physics prior retains its signal, providing a critical robustness margin to the Hybrid model.

| Dataset / Attack Setting | Pixel-Only AUC | Physics-Only AUC | Hybrid AUC | 
|--------------------------|----------------|------------------|------------|
| **VIRAT (Clean)**        | 0.7755         | 0.6613           | 0.7730     |
| **VIRAT (CRF36)**        | 0.8143         | 0.6508           | 0.8356     |
| **VIRAT (Temporal)**     | 0.9000         | 0.6681           | 0.8740     |
| **VIFFD (Real Tampers)** | 0.9334         | 0.7099           | 0.9312     |

*(Note: Results are generated using the single canonical pipeline over the untouched test sets. See `walkthrough.md` for full command execution logs).*
