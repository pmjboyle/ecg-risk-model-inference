import argparse
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score, accuracy_score,
    average_precision_score
)

from model_io import load_bundle


def evaluate_thresholds(y_true, y_prob, thresholds):
    results = {}
    auc_val = roc_auc_score(y_true, y_prob)
    auprc_val = average_precision_score(y_true, y_prob)

    for name, thr in thresholds.items():
        if thr is None:
            results[name] = {k: np.nan for k in ["Accuracy","Precision","Sensitivity","F1","NPV","Specificity","AUROC","AUPRC"]}
            continue

        y_pred = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        sens = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Sensitivity": sens,
            "F1": f1,
            "NPV": npv,
            "Specificity": spec,
            "AUROC": auc_val,
            "AUPRC": auprc_val,
        }

    return pd.DataFrame(results).round(2)


def plot_roc(y_true, y_prob, out_png):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC={auc_val:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="Path to lasso_logreg_bundle.joblib")
    ap.add_argument("--input", required=True, help="CSV/Parquet containing model-ready features")
    ap.add_argument("--out_dir", default="inference_outputs")
    ap.add_argument("--prefix", default="run")
    ap.add_argument("--id_col", default="sample_id")
    ap.add_argument("--label_col", default="y_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    bundle = load_bundle(args.bundle)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]
    thresholds = bundle["thresholds"]

    # Load input
    if args.input.lower().endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    # Separate optional columns
    sample_ids = df[args.id_col].values if args.id_col in df.columns else None
    y_true = df[args.label_col].values if args.label_col in df.columns else None

    # Extract X candidates = everything except optional id/label columns
    drop_cols = [c for c in [args.id_col, args.label_col] if c in df.columns]
    X = df.drop(columns=drop_cols)

    # Column checks
    X_cols = list(X.columns)
    missing = [c for c in feature_names if c not in X_cols]
    extra = [c for c in X_cols if c not in feature_names]

    if missing:
        print("\n[ERROR] Input is missing required feature columns:")
        for c in missing:
            print("  -", c)
        print("\nFix: add these columns to your input file (names must match exactly).")
        print("No inference was run.")
        raise SystemExit(2)

    if extra:
        print("\n[WARN] Input has extra columns not used by the model (will be ignored):")
        for c in extra[:50]:
            print("  -", c)
        if len(extra) > 50:
            print(f"  ... plus {len(extra) - 50} more")

    # Reorder exactly as training
    X = X[feature_names]

    # Scale + predict
    X_scaled = scaler.transform(X)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # Save predictions
    out_pred = pd.DataFrame({"y_prob": y_prob})
    if sample_ids is not None:
        out_pred[args.id_col] = sample_ids
    if y_true is not None:
        out_pred[args.label_col] = y_true

    pred_path = os.path.join(args.out_dir, f"{args.prefix}_predictions.csv")
    out_pred.to_csv(pred_path, index=False)
    print(f"[OK] Saved predictions: {pred_path}")

    # If labels exist, compute metrics + ROC
    if y_true is not None:
        y_true = np.asarray(y_true).astype(int)

        metrics_df = evaluate_thresholds(y_true, y_prob, thresholds)
        metrics_path = os.path.join(args.out_dir, f"{args.prefix}_metrics.csv")
        metrics_df.to_csv(metrics_path)
        print(f"[OK] Saved metrics: {metrics_path}")
        print(metrics_df)

        roc_path = os.path.join(args.out_dir, f"{args.prefix}_roc.png")
        plot_roc(y_true, y_prob, roc_path)
        print(f"[OK] Saved ROC: {roc_path}")


if __name__ == "__main__":
    main()
