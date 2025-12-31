# ECG / EMR Risk Model — Inference & Evaluation

This repository allows users to run **inference and evaluation** using a **pre-trained logistic regression model**.
Users only provide model-ready features.


## What is included

- `lasso_logreg_bundle.joblib`  
  → Trained model + scaler + feature list + thresholds

- `infer_and_eval.py`  
  → Script to run inference and (optionally) evaluation

## What users need to provide

A **single CSV file** containing **model-ready features**. Required columns are listed in feature_names.json.

Column names must match **exactly**.

Each row corresponds to **one sample / ECG**.

### Optional columns
You may also include:
- `sample_id` — identifier for each row## What users need to provide
- `y_true` — ground-truth label (0 or 1)

If `y_true` is provided, the script will compute metrics and ROC curves.

python infer_and_eval.py \
  --bundle bundle_ECG/lasso_logreg_bundle.joblib \
  --input your_model_ready_data.csv \
  --out_dir results \
  --prefix my_run

## Outputs

All outputs are written to the specified output directory.Creates two files:

1. Predictions (always created) and contains: y_prob, saomple_id (if provided), y_true (if provided)
2. Metrics (only if y_true exists) and includes AUROC, AUPRC, Sensitivity, Specificity, Precision, NPV, F1-score (all at preset threshold)


Summary

If you provide the same feature columns the model was trained on, this script will handle scaling, prediction, and evaluation automatically.
