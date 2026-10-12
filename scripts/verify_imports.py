"""Quick import verification for all CoheRex 2.0 modules."""
import sys
sys.path.insert(0, ".")

errors = []

# 1. Config
try:
    from coherex.config import CONFIG
    assert CONFIG.version == "CoheRex-Integrity v2.0"
    assert CONFIG.physics.chi2_threshold > 5.9
    print(f"[OK] config.py (v={CONFIG.version}, chi2={CONFIG.physics.chi2_threshold})")
except Exception as e:
    errors.append(f"config.py: {e}")
    print(f"[FAIL] config.py: {e}")

# 2. Kalman
try:
    from coherex.tracking.kalman import create_kalman_filter, create_ca_kalman_filter
    kf_cv = create_kalman_filter(100, 200)
    kf_ca = create_ca_kalman_filter(100, 200)
    assert kf_cv.dim_x == 4
    assert kf_ca.dim_x == 6
    print(f"[OK] kalman.py (CV dim={kf_cv.dim_x}, CA dim={kf_ca.dim_x})")
except Exception as e:
    errors.append(f"kalman.py: {e}")
    print(f"[FAIL] kalman.py: {e}")

# 3. Track
try:
    from coherex.tracking.track import Track
    from coherex.config import CONFIG
    t = Track(0, (100, 200), 0)
    assert hasattr(t, 'kf_cv')
    assert hasattr(t, 'kf_ca')
    assert hasattr(t, 'nis_combined_history')
    assert hasattr(t, 'keypoint_history')
    import numpy as np
    # Advance past warmup frames so NIS values are captured
    for f in range(CONFIG.physics.nis_warmup_frames + 1):
        t.mark_hit(f)
        t.predict()
        t.update(105 + f, 205 + f)
    assert len(t.nis_cv_history) >= 1, "nis_cv_history is empty"
    assert len(t.nis_ca_history) >= 1, "nis_ca_history is empty"
    assert len(t.nis_combined_history) >= 1, "nis_combined_history is empty"
    print(f"[OK] track.py (NIS after update: cv={t.nis_cv_history[0]:.2f}, ca={t.nis_ca_history[0]:.2f}, combined={t.nis_combined_history[0]:.2f})")
except Exception as e:
    errors.append(f"track.py: {e}")
    print(f"[FAIL] track.py: {e}")

# 4. Innovation
try:
    from coherex.coherence.innovation import (
        nis, combined_nis, calibrate_nis, nis_stats,
        whiteness_lag1, whiteness_lag2, CHI2_95_DOF2
    )
    assert abs(CHI2_95_DOF2 - 5.991) < 0.01
    
    # Test extreme values
    import numpy as np
    test_nis = [1.0, 2.0, 3.0, 10.0, 1.5]
    stats = nis_stats(test_nis)
    assert stats["nis_max"] == 10.0
    
    # Test calibration
    cal = calibrate_nis([1,2,3,4,5,6,7,8,9,10], dof=2, win=5)
    assert len(cal) == 10
    
    # Test whiteness
    innovations = [[1,0], [0,1], [1,0], [0,1], [1,0]]
    w1 = whiteness_lag1(innovations)
    assert 0 <= w1 <= 1
    
    print(f"[OK] innovation.py (CHI2={CHI2_95_DOF2:.3f}, whiteness={w1:.3f})")
except Exception as e:
    errors.append(f"innovation.py: {e}")
    print(f"[FAIL] innovation.py: {e}")

# 5. Changepoint
try:
    from coherex.coherence.changepoint import cusum_peak, glr_change_point, glr_normalized_dispersion
    import numpy as np
    
    # Test with a step signal
    signal = np.concatenate([np.zeros(50), np.ones(50) * 5])
    peak, idx = cusum_peak(signal)
    assert peak > 0
    # CUSUM accumulates — peak is at the end of the sustained shift
    assert idx > 50
    
    glr_peak, glr_idx = glr_change_point(signal)
    assert glr_peak > 0
    assert 45 <= glr_idx <= 55  # GLR precisely locates the step
    
    print(f"[OK] changepoint.py (CUSUM peak={peak:.2f}@{idx}, GLR peak={glr_peak:.2f}@{glr_idx})")
except Exception as e:
    errors.append(f"changepoint.py: {e}")
    print(f"[FAIL] changepoint.py: {e}")

# 6. Pose rhythm
try:
    from coherex.coherence.pose_rhythm import gait_signal, rhythm_residual, spectral_gait_features
    import numpy as np
    
    # Test with synthetic keypoints
    kp_series = []
    for i in range(60):
        kp = np.zeros((17, 2))
        kp[5] = [100, 50]   # L_SH
        kp[6] = [120, 50]   # R_SH
        kp[11] = [105, 100]  # L_HIP
        kp[12] = [115, 100]  # R_HIP
        kp[15] = [100, 150 + 10*np.sin(i*0.5)]  # L_ANK (oscillating)
        kp[16] = [120, 150 - 10*np.sin(i*0.5)]  # R_ANK (oscillating)
        kp_series.append(kp)
    
    gsig = gait_signal(kp_series)
    assert len(gsig) == 60
    assert not np.all(np.isnan(gsig))
    
    rr = rhythm_residual(gsig)
    
    spec = spectral_gait_features(gsig, fps=30.0)
    assert "spec_entropy" in spec
    assert "dom_freq_stab" in spec
    assert "dom_power" in spec
    
    print(f"[OK] pose_rhythm.py (rhythm_residual={rr:.3f}, spec_entropy={spec['spec_entropy']:.3f})")
except Exception as e:
    errors.append(f"pose_rhythm.py: {e}")
    print(f"[FAIL] pose_rhythm.py: {e}")

# 7. Aggregate
try:
    from coherex.coherence.aggregate import (
        extreme_features, multi_scale_features,
        track_count_discontinuity, appearance_burst
    )
    import numpy as np
    
    series = np.array([1, 2, 3, 10, 2, 1, 8, 1, 1, 1])
    ef = extreme_features(series, thresh=5.0)
    assert ef["peak"] == 10.0
    assert ef["longest_run"] == 1  # only single frames above 5
    
    ms = multi_scale_features(series, thresh=5.0, windows=(3, 5))
    assert ms["ms_peak"] == 10.0
    
    tc = track_count_discontinuity([3, 3, 5, 5, 2])
    assert tc["tc_max_jump"] == 3
    
    print(f"[OK] aggregate.py (peak={ef['peak']}, longest_run={ef['longest_run']}, tc_jump={tc['tc_max_jump']})")
except Exception as e:
    errors.append(f"aggregate.py: {e}")
    print(f"[FAIL] aggregate.py: {e}")

# 8. Physics prior
try:
    from coherex.meta.physics_prior import fit_physics_prior, tamper_scores
    import numpy as np
    
    # Synthetic authentic data
    rng = np.random.default_rng(42)
    X_auth = rng.normal(0, 1, size=(50, 3))
    sc, model = fit_physics_prior(X_auth, method="elliptic")
    
    # Score some test data
    X_test = rng.normal(0, 1, size=(10, 3))
    scores = tamper_scores(sc, model, X_test)
    assert len(scores) == 10
    
    print(f"[OK] physics_prior.py (scores range: {scores.min():.2f} to {scores.max():.2f})")
except Exception as e:
    errors.append(f"physics_prior.py: {e}")
    print(f"[FAIL] physics_prior.py: {e}")

# 9. Feature extractor
try:
    from coherex.meta.feature_extractor import (
        FEATURE_NAMES, build_feature_vector, build_feature_dict,
        LEGACY_FEATURE_NAMES, build_legacy_feature_vector
    )
    
    assert len(FEATURE_NAMES) == 13
    vec = build_feature_vector()
    assert len(vec) == 13
    
    fd = build_feature_dict(nis_peak=10.0, glr_peak=5.0)
    assert fd["nis_peak"] == 10.0
    assert fd["glr_peak"] == 5.0
    
    print(f"[OK] feature_extractor.py ({len(FEATURE_NAMES)} features: {FEATURE_NAMES[:3]}...)")
except Exception as e:
    errors.append(f"feature_extractor.py: {e}")
    print(f"[FAIL] feature_extractor.py: {e}")

# 10. Motion agent
try:
    from coherex.integrity.motion_agent import MotionAgent
    agent = MotionAgent()
    # Test NIS-based scoring
    score = agent.evaluate_nis(3.0)  # Below chi2 threshold
    assert score > 0.4
    score_high = agent.evaluate_nis(12.0)  # Above chi2 threshold
    assert score_high < 0.1
    
    print(f"[OK] motion_agent.py (NIS=3.0 -> {score:.2f}, NIS=12.0 -> {score_high:.2f})")
except Exception as e:
    errors.append(f"motion_agent.py: {e}")
    print(f"[FAIL] motion_agent.py: {e}")

# 11. Trajectory store
try:
    from coherex.trajectory.store import TrajectoryStore
    store = TrajectoryStore()
    print("[OK] trajectory/store.py")
except Exception as e:
    errors.append(f"store.py: {e}")
    print(f"[FAIL] store.py: {e}")

# 12. Config loader
try:
    from coherex.config_loader import load_config_from_yaml
    print("[OK] config_loader.py")
except Exception as e:
    errors.append(f"config_loader.py: {e}")
    print(f"[FAIL] config_loader.py: {e}")

# Summary
print()
print("=" * 60)
if errors:
    print(f"FAILED: {len(errors)} modules had errors:")
    for e in errors:
        print(f"  - {e}")
else:
    print("ALL 12 MODULE IMPORTS PASSED SUCCESSFULLY")
print("=" * 60)
