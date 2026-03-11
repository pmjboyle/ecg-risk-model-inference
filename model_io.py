import os
import json
from datetime import datetime
import joblib
import pandas as pd

def save_bundle(bundle_dir, model, scaler, feature_names, thresholds, config=None):
    os.makedirs(bundle_dir, exist_ok=True)

    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": list(feature_names),
        "thresholds": thresholds,
        "config": config or {},
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }

    bundle_path = os.path.join(bundle_dir, "lasso_logreg_bundle.joblib")
    joblib.dump(bundle, bundle_path)

    # optional human-readable helpers
    with open(os.path.join(bundle_dir, "feature_names.json"), "w") as f:
        json.dump(list(feature_names), f, indent=2)

    with open(os.path.join(bundle_dir, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    with open(os.path.join(bundle_dir, "config.json"), "w") as f:
        json.dump(bundle["config"], f, indent=2)

    print(f"[OK] Saved bundle to: {bundle_path}")
    return bundle_path


def load_bundle(bundle_path):
    bundle = joblib.load(bundle_path)
    for k in ["model", "scaler", "feature_names", "thresholds"]:
        if k not in bundle:
            raise ValueError(f"Bundle missing key: {k}")
    return bundle


def align_features(X: pd.DataFrame, feature_names):
    """
    Make X match training-time columns exactly:
      - add missing cols as 0
      - drop extra cols
      - reorder to feature_names
    """
    X = X.copy()

    missing = [c for c in feature_names if c not in X.columns]
    for c in missing:
        X[c] = 0

    extra = [c for c in X.columns if c not in feature_names]
    if extra:
        X = X.drop(columns=extra)

    return X[feature_names]
