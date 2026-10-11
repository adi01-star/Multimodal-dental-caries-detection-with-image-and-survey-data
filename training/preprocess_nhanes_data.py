"""
preprocess_nhanes_data.py
-------------------------
Reads the official CDC NHANES 2017-March 2020 pre-pandemic SAS transport (.xpt) files,
merges records on participant sequence number (SEQN), constructs a clinically grounded
untreated dental caries target from dentition examination codes, extracts validated
lifestyle/behavioral features, and saves the processed dataset and feature schema.

Files used:
  - P_OHXDEN.xpt : Dentition examination (caries target from tooth counts OHX02CTC..OHX31CTC)
  - P_OHQ.xpt    : Oral health questionnaire (dental visits, unmet need, oral health rating)
  - P_DR1TOT.xpt : Dietary interview (total sugars and carbohydrates intake in grams)
  - P_DEMO.xpt   : Demographics (age, sex)
  - P_SMQ.xpt    : Smoking questionnaire (lifetime and current smoking status)
  - P_DIQ.xpt    : Diabetes questionnaire (diagnosed diabetes)

Outputs:
  - data/nhanes_processed.csv: Full merged audit dataset
  - data/nhanes_lifestyle_features.csv: Selected feature matrix + target for ML training
  - data/feature_schema.json: Schema defining features, choices, and defaults
  - data/nhanes_preprocessing_report.json: Merge counts, missing values, and target stats
"""

import argparse
import json
import os
import sys
import pandas as pd
import numpy as np

try:
    import pyreadstat
    HAS_PYREADSTAT = True
except ImportError:
    HAS_PYREADSTAT = False


def load_xpt(path: str) -> pd.DataFrame:
    """Loads an XPT file using pyreadstat with pandas.read_sas fallback."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing XPT file: {path}. Run training/download_nhanes_data.py first.")
    if HAS_PYREADSTAT:
        df, _ = pyreadstat.read_xport(path)
        return df
    else:
        return pd.read_sas(path, format="xport", encoding="utf-8")


def preprocess(raw_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    report = {}

    print("=" * 60)
    print("STEP 1: Loading raw CDC NHANES XPT files...")
    print("=" * 60)

    files = {
        "den": os.path.join(raw_dir, "P_OHXDEN.xpt"),
        "ohq": os.path.join(raw_dir, "P_OHQ.xpt"),
        "diet": os.path.join(raw_dir, "P_DR1TOT.xpt"),
        "demo": os.path.join(raw_dir, "P_DEMO.xpt"),
        "smq": os.path.join(raw_dir, "P_SMQ.xpt"),
        "diq": os.path.join(raw_dir, "P_DIQ.xpt"),
    }

    dfs = {}
    for key, path in files.items():
        dfs[key] = load_xpt(path)
        print(f"  Loaded {os.path.basename(path)}: {len(dfs[key]):,} participants, {dfs[key].shape[1]} columns")
        report[f"raw_count_{key}"] = int(len(dfs[key]))
        # Verify SEQN uniqueness
        assert dfs[key]["SEQN"].nunique() == len(dfs[key]), f"Duplicate SEQN detected in {path}!"

    # ---------------------------------------------------------
    # STEP 2: Filter dentition examination and construct target
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Constructing target variable from clinical examination...")
    print("=" * 60)

    df_den = dfs["den"]
    # OHDDESTS: 1 = Complete, 2 = Partial, 3 = Not Done
    # Only complete dentition exams have reliable tooth-level calls
    complete_den = df_den[df_den["OHDDESTS"] == 1].copy()
    report["dentition_total"] = int(len(df_den))
    report["dentition_complete_count"] = int(len(complete_den))
    print(f"  Total dentition exams: {len(df_den):,}")
    print(f"  Complete dentition exams (OHDDESTS == 1): {len(complete_den):,} (filtered out {len(df_den) - len(complete_den)} incomplete/missing)")

    # Identify coronal caries tooth codes: OHX02CTC through OHX31CTC (28 teeth assessed)
    # NHANES Coronal Caries Tooth Codes:
    #   'Z' = Permanent tooth with dental carious surface condition (untreated active decay)
    #   'K' = Primary tooth with dental carious surface condition (untreated active decay)
    #   'J' = Permanent root tip present without restoration (severe untreated carious destruction)
    #   'F', 'A' = Restored/filled surface condition without untreated decay
    #   'S', 'D' = Sound tooth (no decay, no restoration)
    #   'U' = Unerupted
    #   'E', 'P', 'R' = Missing due to disease / replaced
    #   'M', 'Q', 'X' = Missing due to non-disease causes
    #   'Y' = Cannot be assessed
    ctc_cols = [c for c in df_den.columns if c.endswith("CTC")]
    report["examined_teeth_count"] = len(ctc_cols)
    print(f"  Examined tooth count columns: {len(ctc_cols)} ({ctc_cols[0]} to {ctc_cols[-1]})")

    def count_untreated_caries_teeth(row):
        count = 0
        for c in ctc_cols:
            val = row[c]
            if isinstance(val, str) and val in ("Z", "K", "J"):
                count += 1
        return count

    complete_den["untreated_caries_teeth_count"] = complete_den.apply(count_untreated_caries_teeth, axis=1)
    complete_den["caries_risk"] = (complete_den["untreated_caries_teeth_count"] >= 1).astype(int)

    num_pos = int(complete_den["caries_risk"].sum())
    num_neg = int(len(complete_den) - num_pos)
    prevalence = float(complete_den["caries_risk"].mean())
    report["target_positive_count"] = num_pos
    report["target_negative_count"] = num_neg
    report["target_prevalence"] = prevalence

    print(f"  Target variable definition: 1 if >=1 tooth has untreated caries ('Z', 'K', 'J'), else 0")
    print(f"  Target distribution: Positive = {num_pos:,} ({prevalence*100:.2f}%), Negative = {num_neg:,} ({(1-prevalence)*100:.2f}%)")

    # ---------------------------------------------------------
    # STEP 3: Merging on SEQN
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Merging datasets on participant identifier (SEQN)...")
    print("=" * 60)

    # Base: Complete dentition cohort
    merged = complete_den[["SEQN", "untreated_caries_teeth_count", "caries_risk"]].copy()
    print(f"  Base cohort (complete dentition): {len(merged):,} rows")

    # Merge Demographics
    df_demo = dfs["demo"][["SEQN", "RIDAGEYR", "RIAGENDR", "INDFMPIR"]]
    merged = merged.merge(df_demo, on="SEQN", how="inner")
    print(f"  After inner merge with DEMO (Age, Sex, Poverty): {len(merged):,} rows")
    report["merge_with_demo_count"] = len(merged)

    # Merge Oral Health Questionnaire
    ohq_cols = ["SEQN", "OHQ030", "OHQ033", "OHQ770", "OHQ845"]
    merged = merged.merge(dfs["ohq"][ohq_cols], on="SEQN", how="left")
    print(f"  After left merge with OHQ (Dental visits, unmet need, oral health): {len(merged):,} rows")

    # Merge Dietary Interview (Day 1 Total Nutrients)
    diet_cols = ["SEQN", "DR1TSUGR", "DR1TCARB", "DR1TKCAL"]
    merged = merged.merge(dfs["diet"][diet_cols], on="SEQN", how="left")
    print(f"  After left merge with DR1TOT (Dietary sugars, carbohydrates, energy): {len(merged):,} rows")

    # Merge Smoking Questionnaire
    smq_cols = ["SEQN", "SMQ020", "SMQ040"]
    merged = merged.merge(dfs["smq"][smq_cols], on="SEQN", how="left")
    print(f"  After left merge with SMQ (Smoking history): {len(merged):,} rows")

    # Merge Diabetes Questionnaire
    diq_cols = ["SEQN", "DIQ010"]
    merged = merged.merge(dfs["diq"][diq_cols], on="SEQN", how="left")
    print(f"  After left merge with DIQ (Diabetes diagnosis): {len(merged):,} rows")

    # ---------------------------------------------------------
    # STEP 4: Feature Cleaning and Representation Mapping
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Cleaning variables and handling missing/special codes...")
    print("=" * 60)

    # 1. Demographics: Age and Sex
    clean_df = pd.DataFrame()
    clean_df["SEQN"] = merged["SEQN"]
    clean_df["caries_risk"] = merged["caries_risk"]
    clean_df["age"] = merged["RIDAGEYR"].astype(float)
    clean_df["sex"] = merged["RIAGENDR"].map({1.0: "male", 2.0: "female"}).fillna("female")

    # 2. Dental Visit Frequency / Recency (OHQ030: When did you last visit a dentist)
    # NHANES codes:
    #   1: <= 6 months, 2: >6 mo to 1 yr, 3: >1 yr to 2 yr, 4: >2 yr to 3 yr,
    #   5: >3 yr to 5 yr, 6: >5 yr, 7: never, 77: refused, 99: don't know
    visit_map = {
        1.0: "within_6_months",
        2.0: "6_to_12_months",
        3.0: "1_to_2_years",
        4.0: "2_to_3_years",
        5.0: "3_to_5_years",
        6.0: "more_than_5_years",
        7.0: "never",
    }
    clean_df["last_dental_visit"] = merged["OHQ030"].map(visit_map).fillna("1_to_2_years")

    # 3. Reason for last dental visit (OHQ033)
    # NHANES codes: 1: Routine checkup/cleaning, 2: Called in/appointment, 3: Pain/problem, 4: Other, 5: Never been
    reason_map = {
        1.0: "routine_checkup_or_cleaning",
        2.0: "called_in_or_appointment",
        3.0: "pain_or_problem",
        4.0: "other_reason",
    }
    clean_df["reason_last_dental_visit"] = merged["OHQ033"].map(reason_map)
    # If participant never visited a dentist, set reason to 'never_visited'
    clean_df.loc[merged["OHQ030"] == 7.0, "reason_last_dental_visit"] = "never_visited"
    clean_df["reason_last_dental_visit"] = clean_df["reason_last_dental_visit"].fillna("routine_checkup_or_cleaning")

    # 4. Unmet Dental Need Due to Cost (OHQ770: In past yr needed dental care but could not get it)
    # NHANES codes: 1: Yes, 2: No, 7: Refused, 9: Don't know
    clean_df["unmet_dental_need_cost"] = merged["OHQ770"].map({1.0: "yes", 2.0: "no"}).fillna("no")

    # 5. Self-Rated Oral Health (OHQ845: Rate health of teeth and gums)
    # NHANES codes: 1: Excellent, 2: Very good, 3: Good, 4: Fair, 5: Poor, 7: Refused, 9: Don't know
    health_map = {
        1.0: "excellent",
        2.0: "very_good",
        3.0: "good",
        4.0: "fair",
        5.0: "poor",
    }
    clean_df["self_rated_oral_health"] = merged["OHQ845"].map(health_map).fillna("good")

    # 6. Smoking Status (SMQ020: >=100 cigarettes in life; SMQ040: do you now smoke)
    # For participants <18 years (unasked in SMQ), coded as 'never'
    def derive_smoking(row):
        if row["RIDAGEYR"] < 18:
            return "never"
        s20 = row["SMQ020"]
        s40 = row["SMQ040"]
        if s20 == 2.0:
            return "never"
        elif s20 == 1.0:
            if s40 in (1.0, 2.0):
                return "current"
            elif s40 == 3.0:
                return "former"
            return "former"
        return "never"

    clean_df["smoking_status"] = merged.apply(derive_smoking, axis=1)

    # 7. Diabetes Status (DIQ010: Doctor diagnosed diabetes)
    # NHANES codes: 1: Yes, 2: No, 3: Borderline
    clean_df["diabetes_status"] = merged["DIQ010"].map({1.0: "yes", 2.0: "no", 3: "yes"}).fillna("no")

    # 8. Dietary Intake: Sugars (DR1TSUGR) and Carbohydrates (DR1TCARB) in grams
    # For respondents missing dietary recall (approx 11%), impute median
    sugars_median = float(merged["DR1TSUGR"].dropna().median())
    carbs_median = float(merged["DR1TCARB"].dropna().median())
    clean_df["sugars_intake_g"] = merged["DR1TSUGR"].fillna(sugars_median).clip(lower=0, upper=500).round(1)
    clean_df["carbs_intake_g"] = merged["DR1TCARB"].fillna(carbs_median).clip(lower=0, upper=800).round(1)

    report["sugars_median_imputed"] = sugars_median
    report["carbs_median_imputed"] = carbs_median
    print(f"  Dietary intake median: Sugars = {sugars_median:.1f} g/day, Carbs = {carbs_median:.1f} g/day")

    # ---------------------------------------------------------
    # STEP 5: Save Datasets and Feature Schema
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Saving processed data and schema...")
    print("=" * 60)

    # Full audit dataset
    merged_path = os.path.join(output_dir, "nhanes_processed.csv")
    merged.to_csv(merged_path, index=False)
    print(f"  Saved full merged audit dataset to {merged_path} ({len(merged):,} rows)")

    # Model features dataset
    features_cols = [
        "SEQN",
        "caries_risk",
        "age",
        "sugars_intake_g",
        "carbs_intake_g",
        "sex",
        "last_dental_visit",
        "reason_last_dental_visit",
        "unmet_dental_need_cost",
        "self_rated_oral_health",
        "smoking_status",
        "diabetes_status",
    ]
    model_df = clean_df[features_cols]
    model_features_path = os.path.join(output_dir, "nhanes_lifestyle_features.csv")
    model_df.to_csv(model_features_path, index=False)
    print(f"  Saved model features dataset to {model_features_path} ({len(model_df):,} rows)")

    # Define Schema JSON (Used by Gradio UI and inference pipeline)
    schema = {
        "_comment": (
            "Schema for CDC NHANES 2017-March 2020 pre-pandemic lifestyle caries-risk pipeline. "
            "Defines feature fields, valid ranges, choices, and defaults matching Gradio UI."
        ),
        "target_column": "caries_risk",
        "numeric_features": [
            {
                "name": "age",
                "label": "Age (years)",
                "min": 1,
                "max": 80,
                "default": 35
            },
            {
                "name": "sugars_intake_g",
                "label": "Daily sugar intake (grams/day)",
                "min": 0,
                "max": 400,
                "default": 90
            },
            {
                "name": "carbs_intake_g",
                "label": "Daily carbohydrate intake (grams/day)",
                "min": 0,
                "max": 600,
                "default": 220
            }
        ],
        "categorical_features": [
            {
                "name": "sex",
                "label": "Sex",
                "choices": ["female", "male"],
                "default": "female"
            },
            {
                "name": "last_dental_visit",
                "label": "Last dental visit",
                "choices": [
                    "within_6_months",
                    "6_to_12_months",
                    "1_to_2_years",
                    "2_to_3_years",
                    "3_to_5_years",
                    "more_than_5_years",
                    "never"
                ],
                "default": "within_6_months"
            },
            {
                "name": "reason_last_dental_visit",
                "label": "Reason for last dental visit",
                "choices": [
                    "routine_checkup_or_cleaning",
                    "called_in_or_appointment",
                    "pain_or_problem",
                    "other_reason",
                    "never_visited"
                ],
                "default": "routine_checkup_or_cleaning"
            },
            {
                "name": "unmet_dental_need_cost",
                "label": "Needed dental care in past year but could not afford it",
                "choices": ["no", "yes"],
                "default": "no"
            },
            {
                "name": "self_rated_oral_health",
                "label": "Self-rated oral health (teeth and gums)",
                "choices": ["excellent", "very_good", "good", "fair", "poor"],
                "default": "good"
            },
            {
                "name": "smoking_status",
                "label": "Smoking status",
                "choices": ["never", "former", "current"],
                "default": "never"
            },
            {
                "name": "diabetes_status",
                "label": "Doctor diagnosed diabetes",
                "choices": ["no", "yes"],
                "default": "no"
            }
        ]
    }

    schema_path = os.path.join(output_dir, "feature_schema.json")
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"  Updated feature schema at {schema_path}")

    # Save preprocessing summary report
    report_path = os.path.join(output_dir, "nhanes_preprocessing_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved preprocessing summary report to {report_path}")

    print("\nPreprocessing completed successfully!")
    return model_features_path


def main():
    parser = argparse.ArgumentParser(description="Preprocess CDC NHANES data for dental caries risk prediction.")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "nhanes_raw"),
        help="Directory containing downloaded .xpt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data"),
        help="Directory to save processed datasets.",
    )
    args = parser.parse_args()

    preprocess(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
