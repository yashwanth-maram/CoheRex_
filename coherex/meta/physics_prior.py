# coherex/meta/physics_prior.py
"""
One-Class Self-Supervised Physics Prior (Upgrade 2b)
=====================================================

Fits a density model on the physics features of AUTHENTIC video only;
scores tampering as low-likelihood.

Key properties:
    - Never sees a tampered example → stays fully manipulation-agnostic
    - Adapts per scene via the authentic training manifold
    - Typically lifts AUC because it models the authentic manifold tightly

Evaluation rule:
    The authentic training scenes must be DISJOINT from test scenes
    (still GroupKFold). This is enforced by the training script.

Alternatives if Gaussian assumption is poor:
    - IsolationForest
    - OneClassSVM
    - Gaussian KDE on the feature vector
"""

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def fit_physics_prior(X_auth, method="elliptic", contamination=0.05):
    """
    Fit a one-class density model on authentic-only physics features.

    Args:
        X_auth: (n_samples, n_features) array of physics features from
                authentic videos only
        method: "elliptic" (EllipticEnvelope, Gaussian assumption) or
                "isolation" (IsolationForest, non-parametric)
        contamination: Expected fraction of outliers in the authentic data

    Returns:
        tuple: (scaler, model) — both fitted
    """
    X_auth = np.asarray(X_auth, float)
    if X_auth.ndim == 1:
        X_auth = X_auth.reshape(-1, 1)

    sc = StandardScaler().fit(X_auth)
    X_scaled = sc.transform(X_auth)

    if method == "elliptic":
        model = EllipticEnvelope(
            contamination=contamination,
            support_fraction=0.9
        ).fit(X_scaled)
    elif method == "isolation":
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200
        ).fit(X_scaled)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'elliptic' or 'isolation'.")

    return sc, model


def tamper_scores(sc, model, X):
    """
    Score samples for anomaly (higher = more anomalous = more likely tampered).

    Args:
        sc: Fitted StandardScaler
        model: Fitted one-class model (EllipticEnvelope or IsolationForest)
        X: (n_samples, n_features) array to score

    Returns:
        np.ndarray: Anomaly scores (higher = more anomalous)
    """
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return -model.score_samples(sc.transform(X))  # higher = more anomalous


def fit_and_score_grouped(X, y, groups, method="elliptic", contamination=0.05):
    """
    Fit physics prior per group (leave-one-group-out) and score all samples.

    This implements the GroupKFold-compatible one-class evaluation:
    for each test group, fit on authentic samples from all other groups,
    then score the test group.

    Args:
        X: (n_samples, n_features) feature matrix
        y: (n_samples,) labels (0=authentic, 1=tampered)
        groups: (n_samples,) group identifiers (source_id)
        method: "elliptic" or "isolation"
        contamination: Expected outlier fraction

    Returns:
        np.ndarray: Anomaly scores for all samples (higher = more anomalous)
    """
    X = np.asarray(X, float)
    y = np.asarray(y)
    groups = np.asarray(groups)

    scores = np.zeros(len(X))
    unique_groups = np.unique(groups)

    for test_group in unique_groups:
        test_mask = groups == test_group
        train_mask = ~test_mask

        # Train only on authentic samples from other groups
        train_auth_mask = train_mask & (y == 0)
        X_train_auth = X[train_auth_mask]

        if len(X_train_auth) < 3:
            # Not enough authentic training data; assign neutral score
            scores[test_mask] = 0.0
            continue

        sc, model = fit_physics_prior(X_train_auth, method=method,
                                       contamination=contamination)
        scores[test_mask] = tamper_scores(sc, model, X[test_mask])

    return scores
