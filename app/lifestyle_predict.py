"""
lifestyle_predict.py
---------------------
Runs the lifestyle/behavioral pipeline: takes structured survey answers
(diet, brushing habits, dental visits, fluoride exposure, smoking, etc.),
validates/orders them against the schema, and returns a caries-risk
probability plus a human-readable risk band.

This mirrors `app/predict.py` (the image pipeline) so the two can be called
independently or combined via `app/fusion.py`.
"""

from typing import Dict, Tuple

import pandas as pd

try:
    from app.lifestyle_model import load_lifestyle_model, load_schema
except ImportError:
    from lifestyle_model import load_lifestyle_model, load_schema

_bundle = None
_schema = None


def _get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = load_lifestyle_model()
    return _bundle


def _get_schema():
    global _schema
    if _schema is None:
        _schema = load_schema()
    return _schema


def risk_band(probability: float) -> str:
    """Buckets a probability into a coarse, non-diagnostic risk label."""
    if probability < 0.33:
        return "Low risk"
    elif probability < 0.66:
        return "Moderate risk"
    return "High risk"


def build_feature_row(survey_answers: Dict) -> pd.DataFrame:
    """
    Validates a dict of survey answers against the schema and returns a
    single-row DataFrame in the exact column order the trained pipeline
    expects. Missing fields fall back to the schema's documented default
    rather than failing, since a screening form may leave some questions
    blank.
    """
    schema = _get_schema()
    bundle = _get_bundle()
    feature_cols = bundle["feature_cols"]

    defaults = {f["name"]: f["default"] for f in schema["numeric_features"]}
    defaults.update({f["name"]: f["default"] for f in schema["categorical_features"]})

    row = {col: survey_answers.get(col, defaults[col]) for col in feature_cols}
    return pd.DataFrame([row], columns=feature_cols)


def predict_lifestyle_risk(survey_answers: Dict) -> Tuple[float, str, Dict]:
    """
    Parameters
    ----------
    survey_answers : dict
        Keys should match the "name" fields in data/feature_schema.json
        (e.g. {"age": 34, "brushing_freq_per_day": 2, "sugary_intake_per_day": 4, ...}).
        Any missing keys are filled with schema defaults.

    Returns
    -------
    probability : float
        Predicted probability of elevated caries risk, in [0, 1].
    status : str
        Human-readable risk band ("Low risk" / "Moderate risk" / "High risk").
    details : dict
        Extra info: which model produced the score and the exact feature
        row used, useful for debugging / display in the UI.
    """
    bundle = _get_bundle()
    pipeline = bundle["pipeline"]

    X = build_feature_row(survey_answers)
    probability = float(pipeline.predict_proba(X)[0, 1])
    status = risk_band(probability)

    details = {
        "model_name": bundle["model_name"],
        "feature_row": X.iloc[0].to_dict(),
    }
    return probability, status, details
