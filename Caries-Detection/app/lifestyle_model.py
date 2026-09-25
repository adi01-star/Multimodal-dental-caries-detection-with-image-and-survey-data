"""
lifestyle_model.py
-------------------
Loads the trained lifestyle/behavioral caries-risk model (produced by
`training/train_lifestyle_model.py`) plus the feature schema that describes
which survey fields it expects. Mirrors the style of `app/model.py`, which
loads the YOLO image model.
"""

import json
import os

import joblib

# Prefer the in-container path used by Docker, fall back to a path relative
# to this file so the app also runs directly from a local clone.
_CONTAINER_MODEL_PATH = "/app/data/lifestyle_model.pkl"
_LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "lifestyle_model.pkl")

_CONTAINER_SCHEMA_PATH = "/app/data/feature_schema.json"
_LOCAL_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "feature_schema.json")


def _first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[-1]  # let the caller raise a clear FileNotFoundError


def load_schema() -> dict:
    schema_path = _first_existing(_CONTAINER_SCHEMA_PATH, _LOCAL_SCHEMA_PATH)
    with open(schema_path, "r") as f:
        return json.load(f)


def load_lifestyle_model():
    """
    Returns the saved dict with keys:
        pipeline       -> fitted sklearn Pipeline (preprocessing + classifier)
        model_name     -> str, which algorithm won ("logistic_regression" |
                           "random_forest" | "xgboost")
        feature_cols   -> list[str], expected input columns in order
        metrics        -> dict of held-out test metrics for this model
    Raises FileNotFoundError with a helpful message if training hasn't been
    run yet.
    """
    model_path = _first_existing(_CONTAINER_MODEL_PATH, _LOCAL_MODEL_PATH)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No lifestyle model found at '{model_path}'. "
            "Train one first, e.g.:\n"
            "    python training/generate_synthetic_lifestyle_data.py\n"
            "    python training/train_lifestyle_model.py"
        )
    return joblib.load(model_path)
