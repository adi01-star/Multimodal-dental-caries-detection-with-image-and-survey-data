"""
evaluate_pipelines.py
----------------------
Evaluates the system at the three levels described in the project summary:

    1. Image-only        (YOLO detection confidence -> caries probability)
    2. Lifestyle-only     (XGBoost/RF/LogReg -> risk probability)
    3. Combined            (late fusion of the two)

Requires a *paired* evaluation dataset: for each subject/case you need an
image path, the matching survey answers, and a ground-truth label (whether
caries was clinically confirmed). This is intentionally a CSV-driven script
so you can point it at whatever paired dataset you assemble - the two
training pipelines (image detector, lifestyle classifier) are trained
independently elsewhere; this script only evaluates and fuses their outputs.

Expected CSV columns:
    image_path, <one column per feature in data/feature_schema.json>, label

Usage:
    python training/evaluate_pipelines.py --data data/paired_eval_data.csv \
        --image-weight 0.5
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from predict import predict  # noqa: E402
from lifestyle_predict import predict_lifestyle_risk  # noqa: E402
from fusion import weighted_average_fusion  # noqa: E402
from lifestyle_model import load_schema  # noqa: E402


def score_predictions(y_true, y_prob, threshold=0.5, label=""):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    metrics = {
        "auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    print(f"\n--- {label} ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="CSV with image_path, survey feature columns, and a 'label' column (1 = caries confirmed).",
    )
    parser.add_argument("--image-weight", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "pipeline_evaluation.json"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if "label" not in df.columns or "image_path" not in df.columns:
        raise ValueError("Dataset must contain 'image_path' and 'label' columns.")

    schema = load_schema()
    feature_cols = [f["name"] for f in schema["numeric_features"]] + [
        f["name"] for f in schema["categorical_features"]
    ]

    image_probs, lifestyle_probs, combined_probs, labels = [], [], [], []

    for _, row in df.iterrows():
        image_path = row["image_path"]
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {image_path}: could not open image ({e})")
            continue

        _img_with_boxes, _status, image_confidence = predict(img)

        survey_answers = {col: row[col] for col in feature_cols if col in df.columns}
        lifestyle_prob, _status, _details = predict_lifestyle_risk(survey_answers)

        combined_prob, _band = weighted_average_fusion(
            image_confidence, lifestyle_prob, image_weight=args.image_weight
        )

        image_probs.append(image_confidence)
        lifestyle_probs.append(lifestyle_prob)
        combined_probs.append(combined_prob)
        labels.append(int(row["label"]))

    if not labels:
        raise RuntimeError("No rows were successfully evaluated - check image paths and columns.")

    results = {
        "n_samples": len(labels),
        "image_only": score_predictions(labels, image_probs, args.threshold, "Image Only"),
        "lifestyle_only": score_predictions(labels, lifestyle_probs, args.threshold, "Lifestyle Only"),
        "combined": score_predictions(labels, combined_probs, args.threshold, "Combined (Late Fusion)"),
        "image_weight": args.image_weight,
    }

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation results to {out_path}")

    print("\n=== Summary ===")
    for level in ["image_only", "lifestyle_only", "combined"]:
        print(f"{level}: AUC={results[level]['auc']:.4f}  F1={results[level]['f1']:.4f}")


if __name__ == "__main__":
    main()
