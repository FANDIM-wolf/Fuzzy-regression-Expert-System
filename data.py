"""Загрузка датасета и приведение его к канонической схеме."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CATEGORICAL_FEATURES,
    EGE_RANGE,
    FEATURES,
    LEAKY_COLUMN_MARKERS,
    RU_TO_EN,
    SAT_RANGE,
    TARGETS,
)


def sat_to_ege(sat_score):
    """Линейный перевод SAT (400-1600) в шкалу ЕГЭ (40-100).

    Значения вне диапазона обрезаются, а не приводят к исключению:
    в исходном коде ValueError падал прямо посреди `.apply()` и ронял
    обработку всего датафрейма.
    """
    sat_min, sat_max = SAT_RANGE
    ege_min, ege_max = EGE_RANGE
    sat = np.clip(np.asarray(sat_score, dtype=float), sat_min, sat_max)
    ege = ege_min + (sat - sat_min) * (ege_max - ege_min) / (sat_max - sat_min)
    return np.round(ege, 2)


def _drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    leaky = [
        c
        for c in df.columns
        if any(marker in c for marker in LEAKY_COLUMN_MARKERS)
    ]
    return df.drop(columns=leaky) if leaky else df


def load_dataset(path: str | Path, add_ege: bool = True) -> pd.DataFrame:
    """Читает CSV с русскими или английскими заголовками.

    Возвращает датафрейм в канонической (английской) схеме без колонок,
    полученных из целевых переменных.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={k: v for k, v in RU_TO_EN.items() if k in df.columns})
    df = _drop_leaky_columns(df)

    if add_ege and "SAT_Score" in df.columns:
        df["EGE_Score"] = sat_to_ege(df["SAT_Score"])

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Разделяет датафрейм на матрицу признаков и матрицу целей."""
    missing = [c for c in FEATURES + TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"В данных нет колонок: {missing}")
    return df[FEATURES].copy(), df[TARGETS].copy()
