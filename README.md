# Fuzzy-Regression Expert System for Career Forecasting

A hybrid expert system that combines gradient-boosted regression with Mamdani fuzzy
inference to forecast career outcomes from educational and extracurricular profiles.
The system predicts four targets — starting salary, years to promotion, career
satisfaction and work–life balance — and expresses each of them both as a number and
as a linguistic term (`low` / `medium` / `high`), so that results can be filtered
against a user's stated preferences.

Reference implementation for the paper *A Practical Solution for Forecasting Career
Expectations: From Education to Professional Realization*
([IEEE Xplore](https://ieeexplore.ieee.org/document/11211982)).

---

## Why a hybrid model

A pure regressor returns a point estimate that is hard to act on: the difference
between a predicted salary of 71,400 and 73,900 carries no practical meaning for a
student choosing a track. A pure rule base, on the other hand, cannot exploit the
statistical structure of a large profile dataset.

This system runs both and merges them:

1. **Regression stage.** A CatBoost model with a `MultiRMSE` objective predicts all
   four targets jointly from 15 profile features.
2. **Fuzzification.** Each prediction is mapped onto a three-term partition
   (`low` / `medium` / `high`) whose memberships sum to 1 at every point.
3. **Inference stage.** A complete 27-rule Mamdani base derives a salary term from
   promotion speed, satisfaction and balance; the result is defuzzified by the
   height method.
4. **Aggregation.** The regression estimate and the fuzzy estimate are combined with
   a configurable weight (`--hybrid-weight`, default 0.5).
5. **Preference filtering.** Candidates are selected by their linguistic categories
   rather than by raw thresholds.

---

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/FANDIM-wolf/Fuzzy-regression-Expert-System.git
cd Fuzzy-regression-Expert-System
pip install -r requirements.txt
```

## Quick start

**Train and evaluate.** The model artefact bundles the estimator, the target scaler
and the feature order, so inference never re-fits any preprocessing:

```bash
python -m career_es train \
  --data education_career_success.csv \
  --model career_model.joblib
```

```
Target                          R^2         MAE   MAE (mean)
Starting_Salary             -0.0274   11982.513    11832.355
Years_to_Promotion          -0.0060       1.255        1.240
Career_Satisfaction         -0.0190       2.585        2.571
Work_Life_Balance           -0.0165       2.517        2.512
```

Every metric is reported next to a mean-predicting baseline, so a model that fails to
beat the constant predictor is visible immediately rather than hidden behind an
absolute error figure.

**Tune hyperparameters** with Optuna over 3-fold cross-validation:

```bash
python -m career_es train --data education_career_success.csv --optimize --trials 20
```

**Forecast a single profile:**

```bash
python -m career_es predict --model career_model.joblib --features '{
  "Age": 22, "Gender": "Male", "High_School_GPA": 3.8, "SAT_Score": 1400,
  "University_Ranking": 50, "University_GPA": 3.6,
  "Field_of_Study": "Computer Science", "Internships_Completed": 2,
  "Projects_Completed": 5, "Certifications": 3, "Soft_Skills_Score": 8,
  "Networking_Score": 7, "Job_Offers": 2, "Current_Job_Level": "Entry",
  "Entrepreneurship": "No"}'
```

```json
{
  "salary": 72429.39,
  "salary_ml": 50172.80,
  "salary_fuzzy": 94685.98,
  "years_to_promotion": 2.91,
  "career_satisfaction": 6.02,
  "work_life_balance": 5.11,
  "categories": {
    "Starting_Salary": "medium",
    "Years_to_Promotion": "medium",
    "Career_Satisfaction": "medium",
    "Work_Life_Balance": "medium"
  }
}
```

**Score a cohort and filter it by preferences:**

```bash
python -m career_es recommend \
  --model career_model.joblib \
  --input test.csv \
  --output predictions.csv \
  --filtered shortlist.csv \
  --preferences '{"Career_Satisfaction": ["medium", "high"],
                  "Work_Life_Balance": ["high"]}'
```

**Run the test suite:**

```bash
python -m pytest tests -q
```

---

## Use as a library

```python
from career_es import ExpertSystem

system = ExpertSystem.from_file("career_model.joblib", hybrid_weight=0.7)
result = system.predict_one(
    Age=24, Gender="Female", High_School_GPA=3.9, SAT_Score=1520,
    University_Ranking=30, University_GPA=3.8, Field_of_Study="Medicine",
    Internships_Completed=3, Projects_Completed=4, Certifications=2,
    Soft_Skills_Score=9, Networking_Score=8, Job_Offers=3,
    Current_Job_Level="Entry", Entrepreneurship="No",
)
print(result["categories"])
```

The fuzzy layer is independent of the regressor and can be reused on its own:

```python
from career_es.fuzzy import fuzzify, fuzzy_salary

promotion    = fuzzify(2.0, 1, 5)    # years to promotion
satisfaction = fuzzify(8.5, 1, 10)
balance      = fuzzify(7.0, 1, 10)

salary, term, strength = fuzzy_salary(promotion, satisfaction, balance)
```

---

## Project layout

| Module | Responsibility |
|---|---|
| `career_es/config.py` | feature schema, target ranges, Russian↔English header mapping |
| `career_es/data.py` | CSV loading, SAT→EGE conversion, removal of target-derived columns |
| `career_es/model.py` | CatBoost training, artefact persistence, metrics, Optuna search |
| `career_es/fuzzy.py` | fuzzy partition, 27-rule base, defuzzification |
| `career_es/expert.py` | regression + inference pipeline, preference filtering |
| `career_es/cli.py` | `train` / `evaluate` / `predict` / `recommend` commands |
| `tests/` | unit tests for the fuzzy layer and data conversion |

Both shipped datasets are read through the same code path: Russian column headers in
`education_career_success_translated.csv` are mapped to the canonical English schema
automatically, and columns derived from target variables (`*_cluster`, `Cluster`) are
dropped on load to prevent target leakage.

## Input schema

| Feature | Type | Notes |
|---|---|---|
| `Age` | numeric | |
| `Gender` | categorical | |
| `High_School_GPA` | numeric | 0–4 |
| `SAT_Score` | numeric | 400–1600, converted to a 40–100 EGE scale |
| `University_Ranking` | numeric | |
| `University_GPA` | numeric | 0–4 |
| `Field_of_Study` | categorical | |
| `Internships_Completed` | numeric | |
| `Projects_Completed` | numeric | |
| `Certifications` | numeric | |
| `Soft_Skills_Score` | numeric | 1–10 |
| `Networking_Score` | numeric | 1–10 |
| `Job_Offers` | numeric | |
| `Current_Job_Level` | categorical | |
| `Entrepreneurship` | categorical | |

Targets and their admissible ranges: `Starting_Salary` (25,000–150,000),
`Years_to_Promotion` (1–5), `Career_Satisfaction` (1–10), `Work_Life_Balance` (1–10).
Predictions are clipped to these ranges.

## Design notes

**Target scaling.** Targets are standardised before training. Under a joint
`MultiRMSE` objective, feeding salary (order 10⁵) and years-to-promotion (order 10⁰)
in their native units lets the salary term dominate the loss and leaves the other
three effectively untrained.

**Categorical handling.** Nominal features are passed to CatBoost as `cat_features`
rather than label-encoded into integers, which would impose an arbitrary ordering on
fields of study.

**Fuzzy partition.** The outer terms are trapezoidal. With three plain triangles the
memberships all vanish at the ends of the range, and a value sitting at the minimum
falls through the comparison chain into the `high` branch.

**Rule coverage.** All 27 antecedent combinations are covered. A partial rule base
produces zero firing strength for uncovered inputs, which silently collapses the
defuzzified output to the lower bound instead of raising an error.

**Defuzzification.** The height method is applied over aggregated term strengths.
Multiplying a term centre by its firing strength conflates the confidence of an
inference with the value of the output variable.

## Dataset notes

The bundled `education_career_success.csv` (5,000 records) is a **synthetic
benchmark** used here to exercise and validate the pipeline end to end. Feature–target
correlations in it are near zero (|r| < 0.04 throughout), so R² stays around zero for
any estimator; the numbers in the Quick start section reflect that and should be read
as a pipeline smoke test, not as a claim about predictive accuracy.

Note also that `test.csv` contains the same 5,000 records as the training file with
the target columns removed. It is a format fixture for the `recommend` command, not a
held-out evaluation set — use the `train`/`evaluate` split for measurement.

Applying the system to real cohort data (university employment-monitoring records,
labour-office registries, longitudinal household surveys) requires only that the
input conform to the schema above.

## Citation

```bibtex
@inproceedings{career_expectations_2025,
  title     = {A Practical Solution for Forecasting Career Expectations:
               From Education to Professional Realization},
  author    = {TODO},
  booktitle = {TODO},
  year      = {TODO},
  pages     = {TODO},
  doi       = {TODO}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
