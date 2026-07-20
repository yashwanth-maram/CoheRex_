# CoheRex — Audit & Remediation Plan

**Goal:** bring the *implementation* and the *paper* to research-grade quality, with the
central idea (temporal-coherence / motion-physics as a manipulation-agnostic tampering signal)
correctly derived and honestly evidenced.

This document has three parts:

- **Part A — Findings.** Every concrete mistake/error found, with evidence, grouped by severity.
- **Part B — Root causes.** Why the current results can't support the current claims.
- **Part C — Remediation plan.** A phased, ordered plan: fix errors first, then re-implement,
  then re-test and regenerate results, then rewrite the paper honestly.

Nothing below requires throwing the project away. The core idea is sound and publishable *if*
the experiment is redesigned and the paper is made to describe what the code actually does.

---

## PART A — FINDINGS

### A0. Severity legend
- **[BLOCKER]** — invalidates a central claim; must be fixed before any submission.
- **[MAJOR]** — materially misleads a reader; must be fixed.
- **[MINOR]** — correctness/clarity issue; fix during the rewrite.

---

### A1. Reproducibility of the headline number — [BLOCKER]

- Paper abstract & Table I report **AUC = 0.7507**.
- The committed training log (`train_output.txt`) shows the actual run:
  - Fold AUCs: **0.7500, 1.0000, 1.0000, 0.4167, 0.5000**
  - **MEAN AUC = 0.7333 ± 0.2438**
- The string `0.7507` does **not** appear as an AUC anywhere in the artifacts (only as a stray
  per-frame integrity value in `input_A2_..._report.json`).
- **Conclusion:** the paper's headline metric is not reproducible from the code as committed.
  Either it came from an unsaved run, or it was mis-transcribed. Every reported metric must be
  regenerated from a single, saved, seed-fixed run and must match the text exactly.

### A2. Statistical validity of the AUC claim — [BLOCKER]

- Per-fold AUCs range from **0.42 to 1.00** on folds of ~7–8 samples each (3–4 authentic,
  4 tampered per fold). AUC on ~7 points is extremely high-variance; two folds hitting 1.00 and
  two at/below chance is what noise looks like, not stable discrimination.
- The paper reports `0.7507` (a point estimate from one computation) together with `±0.24`
  (the fold-wise std of a *different* computation, the 0.7333 run). **These two numbers describe
  different quantities and should not be paired.**
- A 5-fold std of 0.24 gives a rough 95% interval that comfortably includes 0.50 (chance).
  The honest statement is **"inconclusive on this dataset size,"** not "statistically meaningful
  discrimination."

### A3. Dataset design — source leakage — [BLOCKER]

From `scripts/create_tampered_videos.py`:
- **Both** authentic and tampered clips are cut from the **same two source videos**
  (`SRC_A = sample.mp4`, `SRC_B = sample1.mp4`).
- **Every** tampered clip is a segment of `SRC_A` with an edit spliced in
  (`base_seg = extract_segment(frames_a, ...)`).
- Consequences:
  1. There are effectively **~2 scenes**, not 38 independent videos. Scene diversity ≈ 0.
  2. Authentic and tampered clips share identical backgrounds, actors, camera, and often
     overlapping/adjacent time windows.
  3. Standard k-fold CV places clips from the same source video in **both** train and test →
     **source/scene leakage**. This is the most likely cause of the 1.00-AUC folds.
- The paper's claims of a **"generalizable, manipulation-agnostic"** detector cannot be supported
  by an experiment on two source videos with derived clips.

### A4. Paper vs. code — architecture mismatch — [MAJOR]

| Aspect | Paper says | Code actually does |
|---|---|---|
| Number of agents | 2 (Motion, Continuity) | **3** (Motion, Continuity, **Crowd**) + a **ReliabilityEstimator** |
| Frame fusion (Eq. 17) | `I_t = α·I_motion + β·I_continuity`, `α+β=1` | 3-way fusion with per-frame **reliability re-weighting** (`fusion_engine.py`); default `fusion_weights=(0.7,0.3,0.0)` |
| Undocumented mechanisms | none mentioned | 10-frame **"tamper latch"** blackout (`motion_agent.py`), **noise floor = 1.5** in z-score (`config.py`), **"late-birth" ×0.5** penalty (`evaluate_dataset.py`), **MCV>2.0** anomaly rule |

The default crowd weight is 0.0, so the two-agent story is *partially* defensible — but the crowd
agent and the reliability re-weighting still run, and the paper never states `α=0.7, β=0.3` nor any
of the four hand-tuned mechanisms that materially change the scores.

### A5. Paper vs. code — feature set is described three different ways — [MAJOR]

- **Eq. 22:** `X = [motion_mean, motion_min, volatility, anomaly_density]` → **4 features**
  (includes `volatility`, which is actually *dropped*; omits `perc_below_06` and `max_mcv_log`,
  which are actually *used*).
- **§IV-C "Feature Configurations":** lists **6** (motion-mean, motion-min, volatility,
  anomaly-density, perc-below-06, max-mcv-log) — omits `min_integrity`, which the code computes.
- **Table III:** lists the **5** actually-used features (motion_min, motion_mean, max_mcv_log,
  perc_below_06, anomaly_density). ✔ this one is correct.
- **Code (`feature_extractor.py` / `train_meta_classifier.py`):** **7 in**
  (motion_min, motion_mean, **min_integrity**, volatility, anomaly_density, perc_below_06,
  max_mcv_log), then `min_integrity` and `volatility` are auto-pruned by the |corr|>0.9 rule,
  leaving **5 used**.
- **Fix:** one consistent description everywhere: "7 candidate features → correlation pruning
  removes min_integrity and volatility → 5 features train the model." Report the pruning rule and
  the correlation matrix.

### A6. Paper vs. code — feature *equations* are wrong — [MAJOR]

- **Eq. 18–19** define `motion_mean`/`motion_min` as the mean/min of the **fused frame integrity**
  `I_t`. The code computes them as the mean/min of the **motion-agent score** only
  (`all_motion_scores`), which is a different quantity (the fused `I_t` is stored separately as
  `integrity_score`/`min_integrity`).
- **Eq. 21** defines `anomaly_density = #{t : I_t < γ}/T`. That formula actually matches the code's
  **`perc_below_06`** (fraction of frames with `I_t < 0.6`). The code's real `anomaly_density` is
  the fraction of frames where **any track has |MCV| > 2.0** — a different definition entirely.
- **Fix:** either rename the features to match the equations or (better) rewrite the equations to
  match the code, and define every symbol (`γ`, the MCV anomaly threshold, etc.) once.

### A7. "Physics-only / not spatial" thesis vs. hybrid code — [MAJOR]

- The paper's framing repeatedly contrasts CoheRex against spatial-artifact methods and calls it
  physics/motion-only.
- `evaluate_dataset.py` computes **SSIM** and **dense optical-flow variance** (Farnebäck) — both
  pixel/spatial features — and stores them in the CSV (`ssim_mean`, `flow_var`, plus
  `accel_kurt`, `burst_length`).
- They are not in the final 5-feature model, so the "physics-only" claim is *technically* true for
  the reported model — but the codebase is a hybrid, and a reviewer reading the repo will notice.
- **Fix:** decide the story. Either (a) keep the pure-physics model and delete/clearly quarantine
  the spatial features, or (b) embrace a hybrid model and report it honestly (this may actually
  perform better and is a legitimate contribution).

### A8. Config / hyperparameter inconsistencies — [MAJOR]

- **MCV window:** engine default `= 3`; `config.py` `mcv_window_size = 8`; paper says
  "rolling history window of **10** frames." Three different numbers. (`max_trajectory_history=10`
  is the trajectory buffer, a different thing from the z-score window.) State each explicitly.
- **Noise floor = 1.5** in the z-score denominator is undocumented and materially changes MCV.
- **Latch:** `config.tamper_latch_frames = 30` but `motion_agent.py` hard-codes a 10-frame latch —
  inconsistent even within the code.

### A9. Single-feature sanity check contradicts the framing — [MAJOR]

- `final_results_summary.txt` and `output_roc.txt` report per-feature AUCs from two runs:
  - `integrity_score` (the flagship fused signal) alone: **AUC ≈ 0.34–0.38** — i.e., *worse than
    chance / anti-correlated* with tampering.
  - `motion_mean` alone: **0.80 in one run, 0.49 in another** — wildly unstable between runs.
- This says the fused "integrity score" is not a good tampering signal on this data, and the
  per-feature signal is dominated by run-to-run noise. This must be understood before claiming the
  pipeline "verifies integrity."

### A10. Dataset description errors in the paper — [MINOR→MAJOR]

- Paper: "38 videos: 17 authentic + 21 tampered." Generator header comment says "20 authentic +
  20 tampered." Actual: some authentic segments were too short and skipped (→17), and a manually
  added `extreme_teleport.mp4` makes tampered 21. Describe the real provenance.
- Paper: tampered segment lengths "**5 to 30 frames**." Code: deletion 5–20 frames, duplication
  ~1.5 s (~37–45 frames), speed window ~2 s, splice 2–3 s (~50–90 frames), reverse ~1.5 s. The
  "5–30 frames" claim is inaccurate for most tamper types.
- Paper: "The only variable ... was the camera's fixed position at 30 FPS" — garbled and does not
  reflect that all clips derive from 2 sources.

### A11. Continuity-agent description is self-contradictory — [MINOR]

- Paper §III-F.2: *"Continuity agent rewards track improvement by penalizing track **stability**."*
  The code docstring correctly says it **penalises instability** (misses, dormancy, re-attachments).
  The sentence in the paper is backwards.

### A12. Reference integrity — [MAJOR, verify]

- Ref [20]: `arXiv:2602.04939, 2026` — a February-2026 arXiv ID; needs verification (looks
  implausible/possibly fabricated).
- Several references carry very specific venue/page details that read as auto-generated. **Every
  reference must be verified against a real DOI/arXiv/DBLP entry**; fabricated citations are a
  desk-reject and an integrity violation.

### A13. Prose / grammar quality — [MINOR]

- The abstract and several sections are heavily "and…and…and" run-ons and contain LLM-artifact
  phrasing ("charges the performance," "rewards track improvement by penalizing track stability").
  A full language pass is needed for a research-grade read.

### A14. Minor code-quality issues — [MINOR]

- Duplicated line `all_crowd_scores = []` in `evaluate_dataset.py`.
- Correlation-based feature pruning is computed on the **full dataset** before CV (mild
  information use across the split; acceptable for unsupervised feature-feature correlation but
  should be stated, or moved inside the CV loop for strictness).
- `.pyc` caches, model weights, and ~450 MB of videos are committed to the repo; add a proper
  `.gitignore` and a small reproducible data-pull instead.

---

## PART B — ROOT CAUSES (why the claims aren't supported yet)

1. **The experiment measures the wrong thing.** With 2 source videos and derived clips, CV
   measures "can the model recognize edited segments of *these two specific videos*," not "can
   motion-coherence detect tampering in general." Leakage inflates some folds to 1.00 and the tiny
   sample makes the rest pure noise.
2. **The paper documents an idealized system, not the running one.** Extra agents, reliability
   weighting, latches, noise floors, and heuristics all exist in code but not on paper, so the
   method section is not a faithful derivation.
3. **Numbers were transcribed, not regenerated from one canonical run.** Hence 0.7507 vs 0.7333,
   mixed variance, and three different feature lists.

Fixing these three things is the whole job.

---

## PART C — REMEDIATION PLAN (ordered)

> Sequencing matches your request: **first correct the errors, then the implementation, then
> re-test and get results, then rewrite the paper.**

### Phase 0 — Freeze & reproduce (0.5 day)
- [ ] Create a clean branch; add `.gitignore` (exclude `*.pyc`, `data/**/*.mp4`, `*.pt`, models).
- [ ] Pin the environment: exact versions of `ultralytics`, `scikit-learn`, `opencv`, `numpy`,
      `filterpy`, `scikit-image`, `scipy` in `requirements.txt`; record Python version.
- [ ] Reproduce the *current* numbers end-to-end from the committed CSV to confirm the 0.7333 run,
      and save the full console output (including the OOF metrics block) to a file. This becomes
      the "before" baseline.

### Phase 1 — Fix the dataset (the BLOCKER) (2–4 days)
This is the single most important change.
- [ ] **Get scene diversity.** Replace the 2-source design with a real corpus of **independent**
      surveillance/authentic clips from *different* scenes. Options, in order of rigor:
  - Public datasets: **VIRAT**, **UCF-Crime**, **CDnet/ChangeDetection**, **DAVIS**,
      or a video-forensics benchmark. (You already have a VIRAT clip in the repo.)
  - If you must self-generate, use **≥30–50 distinct source scenes**, and derive each clip from a
      different source.
- [ ] **Eliminate leakage.** Make authentic and tampered clips come from **disjoint** sources, or
      if a tampered clip is derived from a source, ensure **no authentic clip from that same source
      is in a different fold**. Implement **grouped cross-validation** (`GroupKFold`, group =
      source video ID) so no source appears in both train and test.
- [ ] **Fix and document the tamper generator.** Make segment lengths match what you claim; log the
      exact (type, start, length) per clip to a manifest CSV. Keep the 5 tamper types but record
      them as ground-truth metadata.
- [ ] **Target size.** Aim for **≥150–300 clips** balanced across classes and tamper types. Even
      100+ independent clips changes the statistics dramatically vs. 38.

### Phase 2 — Reconcile code with the intended method (2–3 days)
Decide the *canonical* system and make code = paper.
- [ ] **Agents:** either (a) keep all three agents + reliability and fully document them, or
      (b) drop Crowd/reliability for a clean two-agent method. Pick one and make the code match.
- [ ] **Remove or document heuristics:** the tamper latch, noise floor, late-birth penalty, and the
      MCV>2.0 rule must each be either (i) removed, or (ii) written into the method with a stated
      value and justification. No silent magic numbers.
- [ ] **Fix feature definitions:** make the code's feature names and the paper's Eq. 18–21 agree.
      Recommended: rename `motion_mean/motion_min` → `motionagent_mean/min` and introduce
      `integrity_mean/min` for the fused signal, OR change the equations to reference the
      motion-agent score. Define `anomaly_density` once (choose the MCV-threshold version *or* the
      integrity-threshold version, not both).
- [ ] **Decide physics-only vs hybrid:** quarantine SSIM/flow features if pure-physics is the
      thesis; otherwise promote them and re-frame as a hybrid method.
- [ ] **Unify hyperparameters:** single source of truth for MCV window, noise floor, τ, latch, and
      fusion weights; make the engine default equal the config.

### Phase 3 — Re-run evaluation with correct protocol (1–2 days)
- [ ] Re-extract features on the new dataset (`evaluate_dataset.py`), writing a fresh `results.csv`.
- [ ] Replace k-fold with **GroupKFold** (or leave-one-source-out) in
      `train_meta_classifier.py`. Report **pooled OOF AUC** as the primary metric (more stable than
      mean-of-tiny-folds), plus a **bootstrap 95% CI** on AUC.
- [ ] Report a **stratified train/test holdout** in addition to CV if the dataset is large enough.
- [ ] Add **baselines** so the number means something: (1) random/majority, (2) a single-feature
      threshold (e.g., best individual feature), (3) an off-the-shelf spatial or optical-flow
      tamper detector if feasible. AUC in isolation is not a contribution; AUC *vs. baselines* is.
- [ ] Add an honest **ablation** (motion-only vs +continuity vs +crowd vs +reliability) using the
      `--agents` flag you already have, on the leak-free split.
- [ ] Regenerate the confusion matrix, per-feature AUCs, feature importances, and ROC — **all from
      one saved run** — and wire the exact numbers into the paper via a single results file.

### Phase 4 — Rewrite the paper to research grade (3–5 days)
- [ ] **Abstract & results:** replace every metric with the regenerated numbers; state the CI;
      describe discrimination honestly relative to baselines.
- [ ] **Method:** derive the *actual* system (agents, fusion, reliability, every heuristic, every
      hyperparameter). One consistent feature list. Correct Eq. 17–22. Define all symbols.
- [ ] **Dataset section:** describe real provenance, scene count, leakage-free splitting
      (GroupKFold), tamper types with correct lengths, and the manifest.
- [ ] **Limitations:** keep them, but they should now match the fixed design (e.g., static camera,
      person-centric detection, short-window sensitivity) rather than masking leakage.
- [ ] **References:** verify every entry against a real source; remove/replace fabricated ones.
- [ ] **Language pass:** de-run-on the prose; fix the continuity-agent sentence; consistent notation.
- [ ] **Reproducibility:** include a "how to reproduce" appendix (commands, seeds, dataset source).

### Phase 5 — Optional strengthening (if time allows)
- [ ] Add per-tamper-type detection rates (which manipulations are detectable, which aren't).
- [ ] Add a temporal-localization evaluation (you generate ground-truth tamper windows — use them
      to score *where*, not just *whether*).
- [ ] Multi-scale temporal modeling (you already list this as future work; a small version would
      directly address the "short-window" limitation).

---

## Suggested order of work (dependency-aware)
1. Phase 0 (freeze/repro) → 2. **Phase 1 (dataset — the blocker)** → 3. Phase 2 (code=method) →
4. Phase 3 (re-run + baselines + GroupKFold) → 5. Phase 4 (rewrite) → 6. Phase 5 (extras).

The dataset fix (Phase 1) gates everything: there is no point re-running or rewriting until the
experiment measures the general phenomenon rather than two memorized videos.

---

## One-line honest summary
The *idea* — motion-coherence as a manipulation-agnostic, physics-based tampering signal — is
worth publishing. The current paper reports an unreproducible number on a leaky 2-source dataset and
describes a simpler system than the code runs. Fix the dataset (independent scenes + GroupKFold),
make the paper describe the real system, regenerate every metric from one run against baselines, and
this becomes a credible contribution.
