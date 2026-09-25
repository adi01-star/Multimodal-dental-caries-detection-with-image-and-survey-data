"""
generate_synthetic_lifestyle_data.py
-------------------------------------
Creates a *synthetic* lifestyle/behavioral dataset shaped like the schema in
`data/feature_schema.json`, so the lifestyle-risk pipeline can be trained and
tested end-to-end before real survey data is available.

IMPORTANT: This data is NOT real. It is generated from a hand-written
"ground-truth" risk function with noise added, purely so the training and
evaluation code has something to run on. Before using this project for real
risk assessment, replace this with an actual population-based health /
behavioral dataset, e.g.:
  - NHANES (National Health and Nutrition Examination Survey) Oral Health module
  - A national dental epidemiology survey with lifestyle questionnaires
  - Your own de-identified clinic intake + exam data

As long as the replacement CSV has the same column names as
`data/feature_schema.json` (or you update the schema to match), the rest of
the pipeline (train_lifestyle_model.py, app/lifestyle_predict.py) works
unchanged.

Usage:
    python training/generate_synthetic_lifestyle_data.py \
        --n 3000 --out data/lifestyle_survey_data.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "feature_schema.json")


def load_schema():
    with open(SCHEMA_PATH, "r") as f:
        return json.load(f)


def sample_numeric(rng, spec, n):
    lo, hi = spec["min"], spec["max"]
    mean = spec["default"]
    std = max((hi - lo) / 5.0, 1e-6)
    vals = rng.normal(loc=mean, scale=std, size=n)
    return np.clip(vals, lo, hi)


def sample_categorical(rng, spec, n, weights=None):
    choices = spec["choices"]
    return rng.choice(choices, size=n, p=weights)


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    schema = load_schema()
    rng = np.random.default_rng(seed)

    data = {}

    for spec in schema["numeric_features"]:
        data[spec["name"]] = sample_numeric(rng, spec, n)

    for spec in schema["categorical_features"]:
        # Slightly skew a few fields so the synthetic population isn't uniform
        weights = None
        if spec["name"] == "smoking_status":
            weights = [0.7, 0.15, 0.15]
        elif spec["name"] == "snacking_between_meals":
            weights = [0.3, 0.45, 0.25]
        data[spec["name"]] = sample_categorical(rng, spec, n, weights)

    df = pd.DataFrame(data)

    # Round some numeric columns to sensible integer-like values
    df["age"] = df["age"].round().astype(int)
    df["brushing_freq_per_day"] = df["brushing_freq_per_day"].round().astype(int)
    df["flossing_freq_per_week"] = df["flossing_freq_per_week"].round().astype(int)
    df["sugary_intake_per_day"] = df["sugary_intake_per_day"].round().astype(int)
    df["dental_visits_per_year"] = df["dental_visits_per_year"].round().astype(int)
    df["years_since_last_visit"] = df["years_since_last_visit"].round(1)

    # --- Hand-written "ground truth" risk score used only to LABEL the
    # synthetic data. This encodes plausible clinical relationships
    # (more sugar/less brushing/no fluoride -> higher risk) purely so the
    # demo labels are learnable; it has no clinical validity of its own. ---
    risk = np.zeros(n)
    risk += (df["sugary_intake_per_day"] / 15.0) * 2.5
    risk -= (df["brushing_freq_per_day"] / 5.0) * 1.5
    risk -= (df["flossing_freq_per_week"] / 14.0) * 0.8
    risk -= (df["dental_visits_per_year"] / 6.0) * 1.0
    risk += (df["years_since_last_visit"] / 30.0) * 1.2
    risk += df["fluoride_exposure"].map({"none": 1.2, "low": 0.6, "moderate": 0.0, "high": -0.6})
    risk += df["smoking_status"].map({"never": 0.0, "former": 0.4, "current": 1.0})
    risk += df["snacking_between_meals"].map({"rarely": -0.4, "sometimes": 0.2, "often": 0.9})
    risk += df["dry_mouth_or_low_saliva"].map({"no": 0.0, "yes": 0.7})
    risk += df["family_history_caries"].map({"no": 0.0, "yes": 0.5})
    risk += df["diabetes_status"].map({"no": 0.0, "yes": 0.4})
    risk += df["orthodontic_appliance"].map({"no": 0.0, "yes": 0.5})
    risk += df["socioeconomic_access"].map({"limited": 0.6, "moderate": 0.1, "good": -0.3})
    risk += (df["age"] < 12).astype(float) * 0.3  # young children, primary teeth
    risk += rng.normal(0, 0.6, size=n)  # noise

    prob = 1 / (1 + np.exp(-(risk - risk.mean())))
    df["caries_risk"] = (rng.random(n) < prob).astype(int)

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3000, help="Number of synthetic records")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "lifestyle_survey_data.csv"),
    )
    args = parser.parse_args()

    df = generate(args.n, args.seed)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    positive_rate = df["caries_risk"].mean()
    print(f"Wrote {len(df)} synthetic records to {out_path}")
    print(f"Positive (caries risk) rate: {positive_rate:.2%}")
    print("Replace this file with a real lifestyle/health survey dataset before deployment.")


if __name__ == "__main__":
    main()
