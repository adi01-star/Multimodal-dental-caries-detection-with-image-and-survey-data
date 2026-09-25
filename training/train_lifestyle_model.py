"""
train_lifestyle_model.py
-------------------------
Trains the lifestyle/behavioral caries-risk pipeline on structured survey
data (diet, brushing habits, dental visits, fluoride exposure, smoking,
demographics, etc.) and compares three model families:

    - Logistic Regression (baseline, interpretable)
    - Random Forest
    - XGBoost

Each model is evaluated with stratified train/test split + cross-validation
on ROC-AUC, accuracy, precision, recall and F1. The best model (by mean CV
ROC-AUC) is refit on the full training set, wrapped together with its
preprocessing pipeline, and saved to `data/lifestyle_model.pkl` for use by
`app/lifestyle_predict.py`.

Usage:
    # 1) If you don't have real survey data yet, generate a synthetic set:
    python training/generate_synthetic_lifestyle_data.py

    # 2) Train and compare models:
    python training/train_lifestyle_model.py \
        --data data/lifestyle_survey_data.csv \
        --out data/lifestyle_model.pkl
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "feature_schema.json")


def load_schema():
    with open(SCHEMA_PATH, "r") as f:
        return json.load(f)


def build_preprocessor(schema):
    numeric_cols = [f["name"] for f in schema["numeric_features"]]
    categorical_cols = [f["name"] for f in schema["categorical_features"]]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def get_candidate_models(seed: int):
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }
    if HAS_XGBOOST:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
        )
    else:
        print("xgboost not installed - skipping XGBoost candidate (pip install xgboost to enable).")
    return models


def evaluate_model(pipeline, X_train, y_train, X_test, y_test, cv_folds=5, seed=42):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "cv_auc_mean": float(np.mean(cv_auc)),
        "cv_auc_std": float(np.std(cv_auc)),
        "test_auc": float(roc_auc_score(y_test, y_proba)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    report = classification_report(y_test, y_pred, zero_division=0)
    return metrics, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "lifestyle_survey_data.csv"),
        help="Path to CSV with lifestyle/survey features + target column.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "lifestyle_model.pkl"),
        help="Where to save the trained pipeline (preprocessor + best model).",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    schema = load_schema()
    target_col = schema["target_column"]

    df = pd.read_csv(args.data)
    feature_cols = [f["name"] for f in schema["numeric_features"]] + [
        f["name"] for f in schema["categorical_features"]
    ]
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing expected columns {missing}. "
            f"Either regenerate synthetic data or update data/feature_schema.json to match your real dataset."
        )

    X = df[feature_cols]
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    results = {}
    fitted_pipelines = {}

    for name, model in get_candidate_models(args.seed).items():
        preprocessor, _, _ = build_preprocessor(schema)
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

        print(f"\n=== {name} ===")
        metrics, report = evaluate_model(
            pipeline, X_train, y_train, X_test, y_test, cv_folds=args.cv_folds, seed=args.seed
        )
        print(f"CV ROC-AUC: {metrics['cv_auc_mean']:.4f} +/- {metrics['cv_auc_std']:.4f}")
        print(f"Test ROC-AUC: {metrics['test_auc']:.4f} | Accuracy: {metrics['test_accuracy']:.4f} "
              f"| Precision: {metrics['test_precision']:.4f} | Recall: {metrics['test_recall']:.4f} "
              f"| F1: {metrics['test_f1']:.4f}")
        print(report)

        results[name] = metrics
        fitted_pipelines[name] = pipeline

    best_name = max(results, key=lambda n: results[n]["cv_auc_mean"])
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nBest model by CV ROC-AUC: {best_name} ({results[best_name]['cv_auc_mean']:.4f})")

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "model_name": best_name,
            "feature_cols": feature_cols,
            "metrics": results[best_name],
            "all_model_metrics": results,
        },
        out_path,
    )
    print(f"Saved best lifestyle model ('{best_name}') to {out_path}")

    # Also dump a small comparison table for the README / reports
    comparison_path = os.path.join(os.path.dirname(out_path), "lifestyle_model_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved model comparison metrics to {comparison_path}")


if __name__ == "__main__":
    main()
