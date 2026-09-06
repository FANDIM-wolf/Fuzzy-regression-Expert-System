# Fuzzy-Regression Expert System for Career Expectation Forecasting

An expert system that helps align student career expectations with real labour-market
conditions, built as a hybrid of gradient-boosted regression and Mamdani fuzzy
inference.

Reference implementation for the paper *A Practical Solution for Forecasting Career
Expectations: From Education to Professional Realization*
([IEEE Xplore](https://ieeexplore.ieee.org/document/11211982)).

---

## Problem

Students and graduates routinely enter the labour market with expectations that do not
match what employers actually require. The gap is a barrier to effective cooperation
between universities and industry, and it is particularly visible in the Russian
Federation, where higher education institutions often have limited contact with
business and industry. The consequences are lower graduate competitiveness and a drag
on economic growth.

Closing the gap needs a shared, transportable representation of what a given academic
and personal profile can realistically expect — one that a student, a faculty advisor
and an employer can all read the same way. This project implements one such
representation.

## Approach

The system takes a student's academic, professional and personal characteristics and
produces an expectation profile across four dimensions: **starting salary**, **years
to promotion**, **career satisfaction** and **work–life balance**.

Each dimension is reported twice — as a number and as a linguistic term
(`low` / `medium` / `high`). The linguistic form is the operative one. It is what
makes an expectation comparable to an employer's stated conditions, and what lets a
student express a preference (*"satisfaction at least medium, balance high"*) and see
which profiles satisfy it. Point estimates alone do not support that kind of
reasoning: the difference between a predicted salary of 71,400 and 73,900 carries no
guidance value.

The pipeline runs in five stages:

1. **Regression.** A CatBoost model with a `MultiRMSE` objective predicts all four
   dimensions jointly from 15 profile features.
2. **Fuzzification.** Each prediction is mapped onto a three-term partition whose
   memberships sum to 1 at every point in the range.
3. **Inference.** A complete 27-rule Mamdani base derives a salary expectation from
   promotion speed, satisfaction and balance, capturing domain knowledge that a purely
   statistical model cannot recover from tabular features alone.
4. **Aggregation.** The statistical and the rule-based estimates are combined under a
   configurable weight (`--hybrid-weight`, default 0.5), so an institution can decide
   how far to lean on data versus on expert rules.
5. **Preference filtering.** Cohorts are filtered by linguistic category, which is
   what turns a forecast into actionable guidance.

## Intended use

- **For students** — a realistic picture of the outcomes associated with their current
  profile, and an explicit view of which characteristics move that picture.
- **For faculty advisors** — a basis for directing students toward professional
  development that fits both their profile and market conditions.
- **For universities and employers** — a common vocabulary for the expectation gap,
  expressed in terms both sides can act on rather than in raw model output.

The rule base is plain data (`career_es/fuzzy.py`) and is meant to be revised: an
institution with its own expert knowledge about its region and its industries can
replace it without touching the rest of the pipeline.

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

Every metric is reported next to a mean-predicting baseline, so a model that fails to
beat the constant predictor is visible immediately rather than hidden behind an
absolute error figure.

**Tune hyperparameters** with Optuna over 3-fold cross-validation:

```bash
python -m career_es train --data education_career_success.csv --optimize --trials 20
```

**Build an expectation profile for one student:**

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

**Screen a cohort against stated preferences:**

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

The fuzzy layer is independent of the regressor and can be reused on its own — for
instance, to reason over expectations elicited directly from a student rather than
predicted:

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

Admission scores are carried on a 40–100 EGE scale; the loader converts SAT values
linearly from 400–1600, so data from either system can be used without changing the
schema.

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

Output dimensions and their admissible ranges: `Starting_Salary` (25,000–150,000),
`Years_to_Promotion` (1–5), `Career_Satisfaction` (1–10), `Work_Life_Balance` (1–10).
Values are clipped to these ranges.

## Design notes

**Target scaling.** Targets are standardised before training. Under a joint
`MultiRMSE` objective, feeding salary (order 10⁵) and years-to-promotion (order 10⁰)
in their native units lets the salary term dominate the loss and leaves the other
three dimensions effectively untrained.

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

## Datasets

The bundled `education_career_success.csv` (5,000 records) is a **synthetic benchmark**
used to exercise and validate the pipeline end to end. Its feature–target correlations
are near zero throughout, so reported R² stays around zero regardless of the
estimator — the metrics printed by `train` on this file are a smoke test of the
pipeline, not a measure of forecasting accuracy.

`test.csv` holds the same 5,000 records with the target columns removed. It is a
format fixture for the `recommend` command, not a held-out evaluation set; use the
split produced by `train` or `evaluate` for measurement.

Deployment against real cohort data — university employment-monitoring records,
regional labour-office registries, employer surveys — requires only that the input
conform to the schema above.

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
