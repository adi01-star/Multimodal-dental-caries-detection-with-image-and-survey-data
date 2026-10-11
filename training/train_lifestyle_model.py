"""
train_lifestyle_model.py
-------------------------
Trains the lifestyle/behavioral caries-risk prediction model on official CDC NHANES
data (or custom survey data) and compares candidate model families:
    1. Logistic Regression (baseline, interpretable)
    2. Random Forest Classifier
    3. XGBoost Classifier

Protocol:
    - Stratified train/test split (80% train, 20% held-out test).
    - 5-fold Stratified Cross-Validation strictly on training split for model comparison.
    - Class imbalance handling via balanced class weights / scale_pos_weight.
    - Preprocessing (StandardScaler for numerics, OneHotEncoder for categoricals) fitted
      strictly on training folds without test-set leakage.
    - Model selection based on mean CV ROC-AUC.
    - Final unbiased evaluation of the selected model on the held-out test set.
    - Extraction and export of feature importances, confusion matrix, and classification reports.

Outputs:
    - data/lifestyle_model.pkl: Model bundle (fitted pipeline, feature_cols, test metrics)
    - data/lifestyle_preprocessor.pkl: Fitted preprocessing object
    - data/lifestyle_model_comparison.json: Cross-validation and test metrics across all models
    - data/lifestyle_model_evaluation.json: Test metrics, confusion matrix, and classification report
    - data/feature_importance.csv & .json: Ranked feature importance
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
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


def get_candidate_models(y_train, seed: int):
    # Calculate positive weight ratio for XGBoost
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = float(n_neg / max(n_pos, 1))

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed
        ),
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
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=seed,
        )
    else:
        print("[WARN] xgboost not installed - skipping XGBoost candidate.")

    return models


def extract_feature_importance(pipeline, preprocessor, numeric_cols, categorical_cols, model_name):
    """Extracts and maps feature importance or model coefficients to feature names."""
    try:
        ohe = preprocessor.named_transformers_["cat"]
        cat_features = ohe.get_feature_names_out(categorical_cols).tolist()
        all_features = numeric_cols + cat_features

        clf = pipeline.named_steps["model"]
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])
        else:
            return []

        fi_df = pd.DataFrame({
            "feature": all_features,
            "importance": importances,
        }).sort_values(by="importance", ascending=False)
        return fi_df.to_dict(orient="records")
    except Exception as e:
        print(f"[WARN] Could not extract feature importance: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "nhanes_lifestyle_features.csv"),
        help="Path to CSV with lifestyle features and target column.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "lifestyle_model.pkl"),
        help="Where to save the trained pipeline (preprocessor + best model).",
    )
    parser.add_argument("--test-size", type=float, default=0.20, help="Held-out test set fraction.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed for reproducibility.")
    args = parser.parse_args()

    schema = load_schema()
    target_col = schema["target_column"]

    if not os.path.exists(args.data):
        # Fall back to synthetic data if nhanes data not yet prepared
        alt_data = os.path.join(os.path.dirname(__file__), "..", "data", "lifestyle_survey_data.csv")
        if os.path.exists(alt_data):
            print(f"[WARN] '{args.data}' not found. Falling back to '{alt_data}'.")
            args.data = alt_data
        else:
            raise FileNotFoundError(f"Dataset not found at {args.data}. Run training/preprocess_nhanes_data.py first.")

    df = pd.read_csv(args.data)
    print("=" * 60)
    print(f"Loaded dataset: {os.path.abspath(args.data)}")
    print(f"Total records: {len(df):,}, Columns: {len(df.columns)}")
    print("=" * 60)

    numeric_cols = [f["name"] for f in schema["numeric_features"]]
    categorical_cols = [f["name"] for f in schema["categorical_features"]]
    feature_cols = numeric_cols + categorical_cols

    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing expected columns: {missing}. "
            f"Check data/feature_schema.json and training/preprocess_nhanes_data.py."
        )

    X = df[feature_cols]
    y = df[target_col].astype(int)

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    print(f"Target '{target_col}' distribution: {n_pos:,} positive ({n_pos/len(y)*100:.2f}%), {n_neg:,} negative ({n_neg/len(y)*100:.2f}%)")

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    print(f"Split: Train = {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%), Test = {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

    candidate_models = get_candidate_models(y_train, args.seed)
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    comparison_results = {}
    fitted_pipelines = {}

    scoring = {
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    print("\n" + "=" * 60)
    print(f"Cross-Validation Model Comparison ({args.cv_folds}-Fold Stratified CV on Train Split)")
    print("=" * 60)

    for name, model in candidate_models.items():
        preprocessor, num_c, cat_c = build_preprocessor(schema)
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

        # Cross-validation on train split only
        cv_scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

        cv_metrics = {
            "cv_auc_mean": float(np.mean(cv_scores["test_roc_auc"])),
            "cv_auc_std": float(np.std(cv_scores["test_roc_auc"])),
            "cv_accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
            "cv_precision_mean": float(np.mean(cv_scores["test_precision"])),
            "cv_recall_mean": float(np.mean(cv_scores["test_recall"])),
            "cv_f1_mean": float(np.mean(cv_scores["test_f1"])),
        }

        # Fit on full training split
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Evaluate on held-out test split
        y_test_pred = pipeline.predict(X_test)
        y_test_proba = pipeline.predict_proba(X_test)[:, 1]

        test_metrics = {
            "test_auc": float(roc_auc_score(y_test, y_test_proba)),
            "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
            "test_precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
            "test_f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
        }

        combined_metrics = {**cv_metrics, **test_metrics}
        comparison_results[name] = combined_metrics

        print(f"\nModel: {name.upper()}")
        print(f"  CV ROC-AUC:   {cv_metrics['cv_auc_mean']:.4f} +/- {cv_metrics['cv_auc_std']:.4f}")
        print(f"  CV Recall:    {cv_metrics['cv_recall_mean']:.4f} | CV Precision: {cv_metrics['cv_precision_mean']:.4f} | CV F1: {cv_metrics['cv_f1_mean']:.4f}")
        print(f"  Test ROC-AUC: {test_metrics['test_auc']:.4f} | Test Accuracy: {test_metrics['test_accuracy']:.4f}")
        print(f"  Test Recall:  {test_metrics['test_recall']:.4f} | Test Precision: {test_metrics['test_precision']:.4f} | Test F1: {test_metrics['test_f1']:.4f}")

    # Select best model by mean CV ROC-AUC
    best_name = max(comparison_results, key=lambda n: comparison_results[n]["cv_auc_mean"])
    best_pipeline = fitted_pipelines[best_name]
    print("\n" + "=" * 60)
    print(f"SELECTED BEST MODEL: {best_name.upper()} (CV ROC-AUC = {comparison_results[best_name]['cv_auc_mean']:.4f})")
    print("=" * 60)

    # Detailed evaluation of selected best model on held-out test set
    y_test_pred = best_pipeline.predict(X_test)
    y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    cm_dict = {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "matrix": cm.tolist(),
        "normalized": (cm / cm.sum(axis=1, keepdims=True)).tolist(),
    }

    report_dict = classification_report(y_test, y_test_pred, zero_division=0, output_dict=True)
    report_text = classification_report(y_test, y_test_pred, zero_division=0)

    print("\nHeld-out Test Classification Report:")
    print(report_text)
    print("Confusion Matrix:")
    print(f"  TN: {tn:,} | FP: {fp:,}")
    print(f"  FN: {fn:,}  | TP: {tp:,}")

    # Extract feature importance
    preprocessor = best_pipeline.named_steps["preprocess"]
    fi_list = extract_feature_importance(best_pipeline, preprocessor, numeric_cols, categorical_cols, best_name)

    # Save outputs
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    # 1. Main model bundle
    bundle = {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "metrics": comparison_results[best_name],
        "all_model_metrics": comparison_results,
        "test_confusion_matrix": cm_dict,
        "classification_report": report_dict,
        "dataset_name": "CDC NHANES 2017-March 2020 Pre-pandemic",
        "target_definition": "Untreated dental caries (OHX##CTC in ['Z', 'K', 'J'])",
    }
    joblib.dump(bundle, args.out)
    print(f"\nSaved trained model bundle to {args.out}")

    # 2. Standalone preprocessor object
    preproc_path = os.path.join(out_dir, "lifestyle_preprocessor.pkl")
    joblib.dump(preprocessor, preproc_path)
    print(f"Saved standalone fitted preprocessor to {preproc_path}")

    # 3. Model comparison metrics JSON
    comp_path = os.path.join(out_dir, "lifestyle_model_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"Saved model comparison table to {comp_path}")

    # 4. Detailed evaluation JSON
    eval_dict = {
        "selected_model": best_name,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": comparison_results[best_name],
        "confusion_matrix": cm_dict,
        "classification_report": report_dict,
    }
    eval_path = os.path.join(out_dir, "lifestyle_model_evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(eval_dict, f, indent=2)
    print(f"Saved evaluation metrics to {eval_path}")

    # 5. Feature importance files
    if fi_list:
        fi_json_path = os.path.join(out_dir, "feature_importance.json")
        with open(fi_json_path, "w") as f:
            json.dump(fi_list, f, indent=2)

        fi_df = pd.DataFrame(fi_list)
        fi_csv_path = os.path.join(out_dir, "feature_importance.csv")
        fi_df.to_csv(fi_csv_path, index=False)
        print(f"Saved feature importance report to {fi_csv_path} & {fi_json_path}")
        print("\nTop 5 Most Predictive Features:")
        for row in fi_list[:5]:
            print(f"  - {row['feature']}: {row['importance']:.4f}")

    print("\nTraining and evaluation pipeline finished successfully!")


if __name__ == "__main__":
    main()
