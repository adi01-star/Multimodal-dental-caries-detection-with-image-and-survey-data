# CDC NHANES Oral Health & Behavioral Risk Dataset Report

## 1. Study and Data Sources

This dataset is compiled from the **CDC National Center for Health Statistics (NCHS) National Health and Nutrition Examination Survey (NHANES) 2017–March 2020 Pre-Pandemic Cycle**. 

Data was obtained directly from the official CDC servers:
- **Oral Health Dentition Examination (`P_OHXDEN.xpt`)**: Comprehensive tooth-by-tooth coronal caries evaluations performed by licensed dental examiners.
- **Oral Health Questionnaire (`P_OHQ.xpt`)**: Self-reported and proxy dental care visits, reasons for visits, financial barriers to dental treatment, and self-rated oral health.
- **Dietary Interview - Total Nutrients Day 1 (`P_DR1TOT.xpt`)**: Detailed dietary recall measuring nutritional intake including total sugars (grams) and carbohydrates (grams).
- **Demographics (`P_DEMO.xpt`)**: Participant age and gender.
- **Smoking Questionnaire (`P_SMQ.xpt`)**: Lifetime history and current smoking status.
- **Diabetes Questionnaire (`P_DIQ.xpt`)**: Doctor-diagnosed diabetes mellitus status.

---

## 2. Merging & Cohort Selection

1. **Identifier**: Records were linked across tables using the unique participant sequence number (`SEQN`).
2. **Cardinality**: Merge key uniqueness was validated (1:1 cardinality with 0 duplicate sequence numbers).
3. **Cohort Filter**: Filtered to participants who completed the dental examination (`OHDDESTS == 1`, **N = 13,273**). Participants with incomplete exams (`OHDDESTS != 1`, N = 499) were excluded to maintain clinical ground truth integrity.

| Merge Step | Participant Count |
|---|---|
| Complete Dentition Cohort (`OHDDESTS == 1`) | 13,273 |
| Inner Merge with Demographics (`P_DEMO`) | 13,273 |
| Left Merge with Oral Health Questionnaire (`P_OHQ`) | 13,273 |
| Left Merge with Dietary Recall (`P_DR1TOT`) | 13,273 |
| Left Merge with Smoking (`P_SMQ`) | 13,273 |
| Left Merge with Diabetes (`P_DIQ`) | 13,273 |

---

## 3. Target Definition: Untreated Coronal Dental Caries

* **Target Variable**: `caries_risk` (binary: 0 or 1).
* **Clinical Criteria**: Evaluated across 28 permanent and primary teeth (`OHX02CTC` through `OHX31CTC`, excluding third molars):
  - **1 (Positive)**: Presence of at least one tooth with active untreated coronal caries:
    - Code `Z`: Permanent tooth with untreated dental carious surface condition.
    - Code `K`: Primary tooth with untreated dental carious surface condition.
    - Code `J`: Retained permanent root tip due to severe untreated carious decay.
  - **0 (Negative)**: Absence of active untreated caries lesions. This includes sound teeth (`S`, `D`), unerupted teeth (`U`), and filled/restored teeth without decay (`F`, `A`).
* **Prevalence**:
  - Positive: 2,859 participants (**21.54%**)
  - Negative: 10,414 participants (**78.46%**)

---

## 4. Feature Selection & Missing Value Handling

1. **Age (`age`)**: Participant age in years (1 to 80).
2. **Sex (`sex`)**: `male` or `female`.
3. **Recency of Dental Care (`last_dental_visit`)**:
   - Mapped from `OHQ030` into: `within_6_months`, `6_to_12_months`, `1_to_2_years`, `2_to_3_years`, `3_to_5_years`, `more_than_5_years`, `never`.
4. **Reason for Last Visit (`reason_last_dental_visit`)**:
   - Mapped from `OHQ033` into: `routine_checkup_or_cleaning`, `called_in_or_appointment`, `pain_or_problem`, `other_reason`, `never_visited`.
5. **Financial Access Constraint (`unmet_dental_need_cost`)**:
   - Mapped from `OHQ770` ("In past year needed dental care but could not afford it"): `no`, `yes`.
6. **Self-Rated Oral Health (`self_rated_oral_health`)**:
   - Mapped from `OHQ845` ("Rate the health of your teeth and gums"): `excellent`, `very_good`, `good`, `fair`, `poor`.
7. **Smoking (`smoking_status`)**:
   - Derived from `SMQ020` and `SMQ040`: `never`, `former`, `current` (participants <18 unasked in SMQ mapped to `never`).
8. **Diabetes (`diabetes_status`)**:
   - Derived from `DIQ010`: `no`, `yes`.
9. **Dietary Sugars (`sugars_intake_g`)**:
   - Daily intake in grams from `DR1TSUGR`. Missing dietary recalls imputed with training median (92.1 g/day).
10. **Dietary Carbohydrates (`carbs_intake_g`)**:
    - Daily intake in grams from `DR1TCARB`. Missing dietary recalls imputed with training median (221.9 g/day).

---

## 5. Machine Learning Evaluation

* **Split**: 80% Train (10,618) / 20% Held-Out Test (2,655), stratified on `caries_risk`.
* **Cross-Validation**: 5-Fold Stratified CV on the training set.
* **Winner**: **XGBoost Classifier** (CV ROC-AUC: **0.7750 ± 0.0102**, Held-Out Test ROC-AUC: **0.7805**).

### Confusion Matrix on Held-Out Test Set (N = 2,655):
```
                 Predicted Negative    Predicted Positive
Actual Negative        1,575                  508
Actual Positive          194                  378
```

### Feature Importance (XGBoost):
1. `unmet_dental_need_cost_yes`: 0.1759
2. `unmet_dental_need_cost_no`: 0.1671
3. `reason_last_dental_visit_pain_or_problem`: 0.1181
4. `self_rated_oral_health_fair`: 0.0791
5. `self_rated_oral_health_poor`: 0.0703
6. `last_dental_visit_more_than_5_years`: 0.0461
7. `self_rated_oral_health_excellent`: 0.0450
8. `age`: 0.0401
9. `sugars_intake_g`: 0.0384
10. `carbs_intake_g`: 0.0357
