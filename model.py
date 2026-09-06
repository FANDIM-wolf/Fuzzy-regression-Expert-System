"""Регрессионная подсистема: обучение, сохранение, оценка.

Ключевые отличия от исходного кода:

* Целевые переменные стандартизуются, а не растягиваются MinMaxScaler'ом
  в свои же исходные диапазоны. Раньше зарплата (~1e5) и годы до
  повышения (1..5) попадали в общий MultiRMSE в исходном масштабе,
  поэтому лосс почти целиком определялся зарплатой, а три остальные
  цели фактически не обучались.
* Категориальные признаки передаются в CatBoost как категориальные,
  без LabelEncoder + MinMaxScaler (порядок меток был искусственным).
* Модель, препроцессор и порядок признаков сохраняются одним артефактом.
  Раньше энкодеры и скейлеры заново обучались при каждом предсказании,
  и совпадение их с обученной моделью ничем не гарантировалось.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import CATEGORICAL_FEATURES, FEATURES, TARGET_RANGES, TARGETS
from .data import load_dataset, split_xy

DEFAULT_PARAMS = {
    "iterations": 600,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "loss_function": "MultiRMSE",
    "random_seed": 42,
    "verbose": 0,
}


@dataclass
class CareerModel:
    """Обученная модель вместе со всем, что нужно для инференса."""

    model: CatBoostRegressor
    target_scaler: StandardScaler
    feature_columns: list[str]
    categorical_columns: list[str]
    target_columns: list[str]
    categories: dict[str, list[str]]

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.feature_columns].copy()
        for col in self.categorical_columns:
            X[col] = X[col].astype(str)

        pool = Pool(X, cat_features=self.categorical_columns)
        raw = self.model.predict(pool)
        values = self.target_scaler.inverse_transform(np.atleast_2d(raw))

        out = pd.DataFrame(values, columns=self.target_columns, index=X.index)
        for col in self.target_columns:
            lo, hi = TARGET_RANGES[col]
            out[col] = out[col].clip(lo, hi)
        return out

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "CareerModel":
        return joblib.load(path)


def train(
    data_path: str | Path,
    params: dict | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[CareerModel, dict]:
    """Обучает модель и возвращает её вместе с метриками на отложенной выборке."""
    df = load_dataset(data_path)
    X, y = split_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    target_scaler = StandardScaler().fit(y_train)
    y_train_scaled = target_scaler.transform(y_train)

    model = CatBoostRegressor(**(params or DEFAULT_PARAMS))
    model.fit(Pool(X_train, y_train_scaled, cat_features=CATEGORICAL_FEATURES))

    trained = CareerModel(
        model=model,
        target_scaler=target_scaler,
        feature_columns=FEATURES,
        categorical_columns=CATEGORICAL_FEATURES,
        target_columns=TARGETS,
        categories={c: sorted(X[c].unique().tolist()) for c in CATEGORICAL_FEATURES},
    )

    metrics = evaluate(trained, X_test, y_test, X_train, y_train)
    return trained, metrics


def evaluate(
    trained: CareerModel,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
) -> dict:
    """Сравнивает модель с константным baseline (предсказание среднего).

    Без этого сравнения метрики нечитаемы: R^2 ниже нуля означает, что
    модель хуже, чем предсказание среднего значения по обучающей выборке.
    """
    preds = trained.predict(X_test)
    metrics = {}
    for i, col in enumerate(trained.target_columns):
        dummy = DummyRegressor(strategy="mean").fit(X_train, y_train[col])
        baseline = dummy.predict(X_test)
        metrics[col] = {
            "r2": float(r2_score(y_test[col], preds[col])),
            "mae": float(mean_absolute_error(y_test[col], preds[col])),
            "baseline_mae": float(mean_absolute_error(y_test[col], baseline)),
        }
    return metrics


def optimize(
    data_path: str | Path,
    n_trials: int = 20,
    random_state: int = 42,
) -> dict:
    """Подбор гиперпараметров Optuna по кросс-валидации.

    В исходном коде стояло n_trials=1, то есть поиска не было вовсе:
    бралась одна случайная точка пространства параметров.
    """
    import optuna
    from sklearn.model_selection import KFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df = load_dataset(data_path)
    X, y = split_xy(df)
    scaler = StandardScaler().fit(y)
    y_scaled = scaler.transform(y)

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 300, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "loss_function": "MultiRMSE",
            "random_seed": random_state,
            "verbose": 0,
        }
        kf = KFold(n_splits=3, shuffle=True, random_state=random_state)
        errors = []
        for train_idx, val_idx in kf.split(X):
            m = CatBoostRegressor(**params)
            m.fit(
                Pool(
                    X.iloc[train_idx],
                    y_scaled[train_idx],
                    cat_features=CATEGORICAL_FEATURES,
                )
            )
            pred = m.predict(
                Pool(X.iloc[val_idx], cat_features=CATEGORICAL_FEATURES)
            )
            errors.append(float(np.sqrt(np.mean((pred - y_scaled[val_idx]) ** 2))))
        return float(np.mean(errors))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best = dict(study.best_params)
    best.update(
        {"loss_function": "MultiRMSE", "random_seed": random_state, "verbose": 0}
    )
    return best
