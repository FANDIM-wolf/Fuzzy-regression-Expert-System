"""Гибридная экспертная система: CatBoost + нечёткий вывод."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import FEATURES, TARGET_RANGES
from .data import load_dataset
from .fuzzy import fuzzify, fuzzy_salary
from .model import CareerModel

FUZZY_TARGETS = {
    "Years_to_Promotion": "promotion",
    "Career_Satisfaction": "satisfaction",
    "Work_Life_Balance": "balance",
}


class ExpertSystem:
    """Объединяет числовой прогноз и нечёткую интерпретацию."""

    def __init__(self, model: CareerModel, hybrid_weight: float = 0.5):
        if not 0.0 <= hybrid_weight <= 1.0:
            raise ValueError("hybrid_weight должен быть в [0, 1]")
        self.model = model
        self.hybrid_weight = hybrid_weight

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "ExpertSystem":
        return cls(CareerModel.load(path), **kwargs)

    def predict_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Предсказывает и обогащает нечёткими оценками весь датафрейм."""
        missing = [c for c in FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"Не хватает признаков: {missing}")

        preds = self.model.predict(X)
        result = X.copy()

        for col in preds.columns:
            result[f"Predicted_{col}"] = preds[col].to_numpy()

        memberships = {
            name: [
                fuzzify(v, *TARGET_RANGES[col])
                for v in preds[col]
            ]
            for col, name in FUZZY_TARGETS.items()
        }

        salary_range = TARGET_RANGES["Starting_Salary"]
        fuzzy_values, fuzzy_categories = [], []
        for i in range(len(preds)):
            value, category, _ = fuzzy_salary(
                memberships["promotion"][i],
                memberships["satisfaction"][i],
                memberships["balance"][i],
                salary_range,
            )
            fuzzy_values.append(value)
            fuzzy_categories.append(category)

        w = self.hybrid_weight
        result["Fuzzy_Salary"] = fuzzy_values
        result["Fuzzy_Salary_Category"] = fuzzy_categories
        result["Hybrid_Salary"] = (
            w * result["Fuzzy_Salary"] + (1 - w) * result["Predicted_Starting_Salary"]
        )

        for col, (lo, hi) in TARGET_RANGES.items():
            source = "Hybrid_Salary" if col == "Starting_Salary" else f"Predicted_{col}"
            result[f"{col}_Category"] = [
                fuzzify(v, lo, hi).category for v in result[source]
            ]

        return result

    def predict_one(self, **features) -> dict:
        """Прогноз для одного студента. Ключи — канонические имена признаков."""
        row = pd.DataFrame([features])
        out = self.predict_frame(row).iloc[0]
        return {
            "salary": float(out["Hybrid_Salary"]),
            "salary_ml": float(out["Predicted_Starting_Salary"]),
            "salary_fuzzy": float(out["Fuzzy_Salary"]),
            "years_to_promotion": float(out["Predicted_Years_to_Promotion"]),
            "career_satisfaction": float(out["Predicted_Career_Satisfaction"]),
            "work_life_balance": float(out["Predicted_Work_Life_Balance"]),
            "categories": {
                col: out[f"{col}_Category"] for col in TARGET_RANGES
            },
        }


def filter_by_preferences(df: pd.DataFrame, preferences: dict) -> pd.DataFrame:
    """Отбирает строки, чьи нечёткие категории попадают в предпочтения."""
    if not preferences:
        return df.copy()

    mask = pd.Series(True, index=df.index)
    for target, allowed in preferences.items():
        column = f"{target}_Category"
        if column not in df.columns:
            raise ValueError(f"Нет колонки {column}; сначала вызовите predict_frame")
        mask &= df[column].isin(allowed)
    return df[mask]


def load_candidates(path: str | Path) -> pd.DataFrame:
    """Читает файл с кандидатами (без целевых переменных)."""
    return load_dataset(path)
