"""
fusion.py
---------
Late-fusion combination of the two independent pipelines:

    1. Image pipeline (YOLO)         -> P(caries | image)
    2. Lifestyle pipeline (XGBoost/  -> P(elevated risk | survey answers)
       Random Forest/LogReg)

Two fusion strategies are provided:

  - weighted_average_fusion: simple, interpretable weighted average of the
    two probabilities. Good default, easy to explain to a non-technical
    audience, and lets you tune how much to trust each modality.

  - meta_model_fusion: optional stacking approach - trains a small logistic
    regression "meta-model" on [image_prob, lifestyle_prob] -> true label,
    using a paired dataset with both modalities. Use this once you have
    enough paired image+survey+outcome data to learn the right combination
    weights rather than hand-picking them.

Both return a probability in [0, 1] plus a risk band, exactly like the two
component pipelines, so the same `risk_band()` helper is reused.
"""

from typing import Dict, Optional, Tuple

import joblib
import numpy as np

try:
    from app.lifestyle_predict import risk_band
except ImportError:
    from lifestyle_predict import risk_band


def weighted_average_fusion(
    image_prob: float,
    lifestyle_prob: float,
    image_weight: float = 0.5,
) -> Tuple[float, str]:
    """
    Combines the two pipeline outputs as a simple weighted average.

    Parameters
    ----------
    image_prob : float
        Confidence that the image pipeline assigns to caries being present
        (e.g. the top detection confidence from app/predict.py; 0.0 if no
        caries was detected).
    lifestyle_prob : float
        Probability of elevated risk from app/lifestyle_predict.py.
    image_weight : float
        Weight given to the image pipeline, in [0, 1]. The lifestyle
        pipeline gets (1 - image_weight). Defaults to an even 50/50 split;
        tune this once you have paired validation data showing which
        modality is more reliable for your population.
    """
    image_weight = float(np.clip(image_weight, 0.0, 1.0))
    combined = image_weight * image_prob + (1 - image_weight) * lifestyle_prob
    return combined, risk_band(combined)


def train_meta_model_fusion(image_probs, lifestyle_probs, labels, seed: int = 42):
    """
    Fits a small logistic-regression meta-model on top of the two pipelines'
    outputs. Requires a paired dataset (same subjects have both an image
    prediction, a lifestyle prediction, and a ground-truth label).

    Parameters
    ----------
    image_probs, lifestyle_probs, labels : array-likes of equal length.

    Returns
    -------
    A fitted sklearn LogisticRegression you can save with joblib and load
    back with `load_meta_model_fusion`.
    """
    from sklearn.linear_model import LogisticRegression

    X = np.column_stack([image_probs, lifestyle_probs])
    y = np.asarray(labels)
    meta_model = LogisticRegression(class_weight="balanced", random_state=seed)
    meta_model.fit(X, y)
    return meta_model


def save_meta_model(meta_model, path: str):
    joblib.dump(meta_model, path)


def load_meta_model(path: str):
    return joblib.load(path)


def meta_model_fusion(image_prob: float, lifestyle_prob: float, meta_model) -> Tuple[float, str]:
    X = np.array([[image_prob, lifestyle_prob]])
    combined = float(meta_model.predict_proba(X)[0, 1])
    return combined, risk_band(combined)


def combine(
    image_prob: float,
    lifestyle_prob: float,
    strategy: str = "weighted_average",
    image_weight: float = 0.5,
    meta_model=None,
) -> Dict:
    """
    Convenience wrapper used by the Gradio app. Returns a dict so the UI
    layer doesn't need to know about tuples.
    """
    if strategy == "weighted_average":
        prob, band = weighted_average_fusion(image_prob, lifestyle_prob, image_weight)
    elif strategy == "meta_model":
        if meta_model is None:
            raise ValueError("meta_model_fusion strategy requires a fitted meta_model.")
        prob, band = meta_model_fusion(image_prob, lifestyle_prob, meta_model)
    else:
        raise ValueError(f"Unknown fusion strategy: {strategy}")

    return {
        "combined_probability": prob,
        "combined_risk_band": band,
        "image_probability": image_prob,
        "lifestyle_probability": lifestyle_prob,
        "strategy": strategy,
    }
