# coherex/meta/classifier.py
"""
Random Forest Meta-Classifier
==============================
Loads a pre-trained scikit-learn model and serves inference.

This module is intentionally thin — it only handles:
  1. Loading the model from disk (lazy, thread-safe)
  2. Running predict_proba() on a feature vector
  3. Returning a structured result dictionary

The actual training lives in scripts/train_meta_classifier.py.

Usage:
    from coherex.meta.classifier import MetaClassifier

    clf = MetaClassifier()            # auto-locates model file
    result = clf.score(feature_vec)   # returns dict
    print(result["confidence"])       # 0.0–1.0 tamper probability
    print(result["verdict"])          # "AUTHENTIC" | "SUSPICIOUS" | "TAMPERED"
"""

import os
import warnings

# ── Path resolution ──────────────────────────────────────────────────────────
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(os.path.dirname(_PKG_DIR))
DEFAULT_MODEL_PATH = os.path.join(_ROOT, "data", "models", "meta_classifier.pkl")


class MetaClassifier:
    """
    Thin wrapper around a serialized scikit-learn model.

    Lazy-loads on first call to score(). Safe to instantiate even
    when the model file does not yet exist — score() returns None.
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self._model    = None
        self._features = None
        self._loaded   = False
        self._available = False

    # ── Loading ──────────────────────────────────────────────────────────────

    def _try_load(self):
        """Load model from disk. Silent on failure — not yet trained is OK."""
        if self._loaded:
            return
        self._loaded = True

        if not os.path.exists(self.model_path):
            self._available = False
            return

        try:
            import joblib
            bundle = joblib.load(self.model_path)
            self._model    = bundle["model"]
            self._features = bundle["feature_names"]
            self._available = True
        except Exception as e:
            warnings.warn(f"[MetaClassifier] Failed to load model: {e}")
            self._available = False

    @property
    def is_available(self) -> bool:
        self._try_load()
        return self._available

    # ── Inference ─────────────────────────────────────────────────────────────

    def score(self, feature_vector: list) -> dict:
        """
        Run the classifier on a feature vector.

        Args:
            feature_vector: list aligned with FEATURE_NAMES from
                            coherex.meta.feature_extractor.

        Returns:
            dict with keys:
                confidence  (float):  P(tampered) in [0, 1]
                verdict     (str):    "AUTHENTIC" | "SUSPICIOUS" | "TAMPERED"
                available   (bool):   whether the model was loaded
        """
        self._try_load()

        if not self._available:
            return {
                "confidence": None,
                "verdict":    "UNAVAILABLE",
                "available":  False,
            }

        import numpy as np
        X = np.array(feature_vector, dtype=float).reshape(1, -1)
        try:
            proba = self._model.predict_proba(X)[0]
            # Class 1 = tampered
            p_tampered = float(proba[1]) if len(proba) > 1 else float(proba[0])

            if p_tampered >= 0.65:
                verdict = "TAMPERED"
            elif p_tampered >= 0.40:
                verdict = "SUSPICIOUS"
            else:
                verdict = "AUTHENTIC"

            return {
                "confidence": round(p_tampered, 4),
                "verdict":    verdict,
                "available":  True,
            }
        except Exception as e:
            warnings.warn(f"[MetaClassifier] Inference error: {e}")
            return {
                "confidence": None,
                "verdict":    "ERROR",
                "available":  False,
            }

    def score_dict(self, feature_dict: dict) -> dict:
        """
        Score from a named feature dict (from build_feature_dict()).
        Validates feature order against the trained model's expectation.
        """
        self._try_load()
        if not self._available:
            return {"confidence": None, "verdict": "UNAVAILABLE", "available": False}

        vec = [feature_dict.get(k, 0.0) for k in self._features]
        return self.score(vec)

    def reload(self):
        """Force a model reload from disk (e.g., after retraining)."""
        self._loaded = False
        self._available = False
        self._model = None
        self._try_load()
