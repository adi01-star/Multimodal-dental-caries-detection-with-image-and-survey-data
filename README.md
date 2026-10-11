# Multimodal Dental Caries Detection and Behavioral Risk Modeling

A Multimodal Late-Fusion Framework for Dental Caries Screening combining deep learning image localization and epidemiological behavioral risk modeling.

---

## 📌 Project Overview

This screening application integrates two complementary modalities into a late-fusion architecture:

1. **Image Pipeline (Computer Vision)**:
   - Ultralytics **YOLOv8** (`data/best.pt`) detects and localizes visible carious lesions from dental radiographic and intraoral images.
2. **Behavioral & Lifestyle Pipeline (Epidemiological ML)**:
   - Machine learning risk modeling (**XGBoost**, **Random Forest**, **Logistic Regression**) trained on official **CDC NHANES** (National Health and Nutrition Examination Survey, 2017–March 2020 pre-pandemic cycle) data.
   - Predicts individual caries risk based on clinical visit patterns, reason for dental visits, financial access barriers, self-rated oral health, daily dietary sugar and carbohydrate intake, smoking status, diabetes, and demographics.
3. **Late-Fusion Screening**:
   - Computes a late-fusion risk score combining image evidence and behavioral risk indicators via weighted averaging or stacking.

> ⚠️ **Clinical Screening Disclaimer**: This software is designed strictly as an experimental screening and risk-assessment aid. It is **not** an automated diagnostic system or medical device and does not replace examination by a licensed dental professional.

---

## 📂 Repository Structure

```
├── app/
│   ├── gradio-app.py             # Web UI with Image Only, Survey Only, and Combined tabs
│   ├── model.py                  # YOLOv8 image model loader
│   ├── predict.py                # Image detection pipeline (caries bounding boxes & confidence)
│   ├── lifestyle_model.py        # Loads trained NHANES model bundle + feature schema
│   ├── lifestyle_predict.py      # Lifestyle risk prediction with schema validation
│   └── fusion.py                 # Multimodal late-fusion engine (weighted average / meta-model)
├── training/
│   ├── download_nhanes_data.py   # Downloads official CDC NHANES 2017-March 2020 XPT files
│   ├── preprocess_nhanes_data.py # Merges NHANES files on SEQN, defines target, extracts features
│   ├── train_lifestyle_model.py  # 5-fold CV & test evaluation across LogReg, RF, and XGBoost
│   ├── evaluate_pipelines.py     # 3-level evaluation script for paired datasets
│   └── generate_synthetic_lifestyle_data.py  # Legacy synthetic generator (superseded by NHANES)
├── data/
│   ├── best.pt                   # Pre-trained YOLOv8 weights (image pipeline)
│   ├── feature_schema.json       # Feature definitions, bounds, and defaults
│   ├── nhanes_raw/               # Raw downloaded CDC NHANES .xpt files (git-ignored)
│   ├── nhanes_processed.csv      # Merged audit dataset (13,273 participants)
│   ├── nhanes_lifestyle_features.csv # Model-ready feature matrix + target
│   ├── lifestyle_model.pkl       # Best trained pipeline (XGBoost) with fitted preprocessing
│   ├── lifestyle_preprocessor.pkl# Standalone fitted ColumnTransformer
│   ├── lifestyle_model_comparison.json  # Cross-validation & test metrics across models
│   ├── lifestyle_model_evaluation.json  # Held-out test evaluation, CM, classification report
│   ├── feature_importance.csv    # Ranked feature importance table
│   └── feature_importance.json   # Ranked feature importance JSON
├── requirements.txt              # Project dependencies
├── Dockerfile                    # Containerization configuration
└── README.md                     # Documentation
```

---

## 🧬 Official CDC NHANES Data Integration

### 1. Data Sources
The behavioral model uses official pre-pandemic data files from the **CDC National Center for Health Statistics (NCHS) NHANES 2017–March 2020**:

| Component | File | Description | Participants |
|---|---|---|---|
| **Examination** | `P_OHXDEN.xpt` | Oral Health - Dentition Examination | 13,772 |
| **Questionnaire** | `P_OHQ.xpt` | Oral Health Questionnaire (dental visits, access, oral health) | 14,986 |
| **Dietary** | `P_DR1TOT.xpt` | Dietary Interview - Day 1 Total Nutrients (sugars, carbs, energy) | 14,300 |
| **Demographics** | `P_DEMO.xpt` | Demographics (Age, Sex, Income-to-Poverty Ratio) | 15,560 |
| **Questionnaire** | `P_SMQ.xpt` | Smoking - Cigarette Use (lifetime & current status) | 11,137 |
| **Questionnaire** | `P_DIQ.xpt` | Diabetes Questionnaire | 14,986 |

### 2. Participant Merge & Cohort
* **Merge Key**: `SEQN` (Respondent sequence number). Every dataset has 100% unique `SEQN` values (1:1 cardinality, 0 duplicates).
* **Cohort Selection**: Restricted to participants with a **complete dentition examination** (`OHDDESTS == 1`, **N = 13,273**), ensuring clinical ground-truth validity.

### 3. Clinically Grounded Target Construction
The binary target `caries_risk` is derived directly from tooth-level clinical examination calls (`OHX02CTC` through `OHX31CTC` across 28 evaluated teeth):
* **Positive (`caries_risk = 1`)**: At least one evaluated tooth presents with untreated coronal caries lesions:
  - Code `Z`: Permanent tooth with untreated dental carious surface condition.
  - Code `K`: Primary tooth with untreated dental carious surface condition.
  - Code `J`: Retained root tip without restoration due to severe untreated decay.
* **Negative (`caries_risk = 0`)**: No teeth with untreated decay (sound teeth `S`/`D`, unerupted teeth `U`, or restored/filled surfaces without active decay `F`/`A`).
* **Prevalence**: 2,859 positive cases (**21.54%**), 10,414 negative cases (**78.46%**).

### 4. Selected Features
* **Demographics**: `age` (1–80 years), `sex` (`male`, `female`).
* **Dental Care Utilization**: `last_dental_visit` (recency intervals: `<=6 mo` to `never`), `reason_last_dental_visit` (`routine_checkup_or_cleaning`, `pain_or_problem`, `called_in_or_appointment`, `other_reason`, `never_visited`).
* **Access & Need**: `unmet_dental_need_cost` (needed dental care in past year but could not afford it: `no`, `yes`).
* **Self-Reported Health**: `self_rated_oral_health` (`excellent`, `very_good`, `good`, `fair`, `poor`).
* **Systemic & Behavioral Risk**: `smoking_status` (`never`, `former`, `current`), `diabetes_status` (`no`, `yes`).
* **Dietary Intake**: `sugars_intake_g` (total daily sugar consumption in grams), `carbs_intake_g` (total daily carbohydrate intake in grams).

---

## 📊 Model Training & Evaluation Results

Data was split into **80% Training (10,618 samples)** and **20% Held-Out Testing (2,655 samples)** with stratification on `caries_risk`. Candidate models were benchmarked using **5-Fold Stratified Cross-Validation strictly on the training set**:

| Model | 5-Fold CV ROC-AUC | Test ROC-AUC | Test Accuracy | Test Recall (Caries) | Test Precision | Test F1 |
|---|---|---|---|---|---|---|
| **XGBoost (Selected)** | **0.7750 ± 0.0102** | **0.7805** | **0.7356** | **0.6608** | **0.4266** | **0.5185** |
| Logistic Regression | 0.7706 ± 0.0086 | 0.7767 | 0.7330 | 0.6486 | 0.4221 | 0.5114 |
| Random Forest | 0.7602 ± 0.0108 | 0.7597 | 0.7887 | 0.4336 | 0.5113 | 0.4693 |

### Confusion Matrix on Held-Out Test Set (N = 2,655)
* **True Negatives (TN)**: 1,575
* **False Positives (FP)**: 508
* **False Negatives (FN)**: 194
* **True Positives (TP)**: 378 (66.08% sensitivity for detecting untreated caries)

### Top 5 Predictive Features (XGBoost Importance)
1. `unmet_dental_need_cost = yes` (17.59%)
2. `unmet_dental_need_cost = no` (16.71%)
3. `reason_last_dental_visit = pain_or_problem` (11.81%)
4. `self_rated_oral_health = fair` (7.91%)
5. `self_rated_oral_health = poor` (7.03%)

---

## 🚀 Execution & Reproduction Guide

### 1. Setup Environment
```bash
git clone https://github.com/adi01-star/Multimodal-dental-caries-detection-with-image-and-survey-data.git
cd Multimodal-dental-caries-detection-with-image-and-survey-data
pip install -r requirements.txt
```

### 2. Download Real CDC NHANES Data
Downloads official SAS transport files directly from CDC NCHS servers with automatic validation and fallback handlers:
```bash
python training/download_nhanes_data.py
```

### 3. Preprocess & Feature Engineering
Merges datasets by `SEQN`, constructs the clinical caries ground-truth target, and exports the clean training sets:
```bash
python training/preprocess_nhanes_data.py
```

### 4. Train & Evaluate Models
Performs stratified splitting, 5-fold CV comparison, model selection, held-out testing, and saves model bundles:
```bash
python training/train_lifestyle_model.py
```

### 5. Launch the Web Application
```bash
python app/gradio-app.py
```
Open `http://localhost:7860` in your web browser.

---

## 🐳 Docker Deployment

Train the model locally before containerization, then build and run:
```bash
docker build -t caries-detection-app .
docker run -p 7860:7860 caries-detection-app
```
