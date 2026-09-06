"""Нечёткая подсистема: разбиение на термы, база правил, дефаззификация.

Отличия от исходной реализации:

1. Крайние термы сделаны трапециевидными. В исходном коде все три
   треугольника обнулялись на границах диапазона, из-за чего значение,
   равное минимуму, получало нулевую принадлежность ко всем термам и
   по ветке `else` объявлялось "high".
2. Сумма принадлежностей равна 1 в любой точке — разбиение корректное.
3. База правил покрывает все 27 комбинаций, а не 10. Раньше при
   непокрытой комбинации все силы правил были нулевыми и система
   молча выдавала минимальную зарплату.
4. Дефаззификация — взвешенное среднее по центрам термов. Раньше
   результат умножался на степень уверенности, то есть низкая
   уверенность механически занижала прогноз зарплаты.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

TERMS = ("low", "medium", "high")

# Положение опорных точек внутри нормированного диапазона [0, 1].
_LOW_PEAK, _MID_PEAK, _HIGH_PEAK = 0.2, 0.5, 0.8


@dataclass(frozen=True)
class Membership:
    """Степени принадлежности значения к термам low / medium / high."""

    low: float
    medium: float
    high: float

    @property
    def category(self) -> str:
        return max(TERMS, key=lambda t: getattr(self, t))

    @property
    def confidence(self) -> float:
        return getattr(self, self.category)

    def as_dict(self) -> dict:
        return {
            "low": round(self.low, 4),
            "medium": round(self.medium, 4),
            "high": round(self.high, 4),
            "category": self.category,
        }

    def __getitem__(self, term: str) -> float:
        return getattr(self, term)


def fuzzify(value: float, min_val: float, max_val: float) -> Membership:
    """Строит принадлежности для значения в диапазоне [min_val, max_val]."""
    if max_val <= min_val:
        raise ValueError("max_val должен быть больше min_val")

    x = (float(value) - min_val) / (max_val - min_val)
    x = min(max(x, 0.0), 1.0)

    if x <= _LOW_PEAK:
        low, medium, high = 1.0, 0.0, 0.0
    elif x <= _MID_PEAK:
        t = (x - _LOW_PEAK) / (_MID_PEAK - _LOW_PEAK)
        low, medium, high = 1.0 - t, t, 0.0
    elif x <= _HIGH_PEAK:
        t = (x - _MID_PEAK) / (_HIGH_PEAK - _MID_PEAK)
        low, medium, high = 0.0, 1.0 - t, t
    else:
        low, medium, high = 0.0, 0.0, 1.0

    return Membership(low, medium, high)


def _ordinal(term: str) -> int:
    return TERMS.index(term)


def build_rule_base() -> list[dict]:
    """Полная база из 27 правил Мамдани.

    Заключение выводится из порядковой свёртки посылок: чем быстрее
    повышение, выше удовлетворённость и лучше баланс, тем выше терм
    зарплаты. Терм `promotion` трактуется как СКОРОСТЬ повышения
    (high = повышение наступает быстро), поэтому в `apply_rules`
    принадлежности по годам до повышения инвертируются.
    """
    rules = []
    for promotion, satisfaction, balance in product(TERMS, repeat=3):
        score = (
            2.0 * _ordinal(satisfaction)
            + 1.5 * _ordinal(promotion)
            + 1.0 * _ordinal(balance)
        ) / 4.5
        if score < 0.67:
            output = "low"
        elif score < 1.34:
            output = "medium"
        else:
            output = "high"
        rules.append(
            {
                "conditions": [
                    ("promotion", promotion),
                    ("satisfaction", satisfaction),
                    ("balance", balance),
                ],
                "output": output,
            }
        )
    return rules


RULE_BASE = build_rule_base()


def _invert(m: Membership) -> Membership:
    """Мало лет до повышения = высокая скорость повышения."""
    return Membership(low=m.high, medium=m.medium, high=m.low)


def apply_rules(
    promotion: Membership,
    satisfaction: Membership,
    balance: Membership,
    rules: list[dict] | None = None,
) -> dict[str, float]:
    """Агрегирует силы правил (min-импликация, max-агрегация)."""
    rules = RULE_BASE if rules is None else rules
    variables = {
        "promotion": _invert(promotion),
        "satisfaction": satisfaction,
        "balance": balance,
    }

    strength = {term: 0.0 for term in TERMS}
    for rule in rules:
        firing = min(variables[var][term] for var, term in rule["conditions"])
        out = rule["output"]
        if firing > strength[out]:
            strength[out] = firing
    return strength


def defuzzify(strength: dict[str, float], min_val: float, max_val: float) -> float:
    """Взвешенное среднее по центрам термов (метод высот)."""
    span = max_val - min_val
    centers = {
        "low": min_val + span * _LOW_PEAK,
        "medium": min_val + span * _MID_PEAK,
        "high": min_val + span * _HIGH_PEAK,
    }
    total = sum(strength.values())
    if total <= 0:
        return min_val + span * _MID_PEAK
    return sum(strength[t] * centers[t] for t in TERMS) / total


def fuzzy_salary(
    promotion: Membership,
    satisfaction: Membership,
    balance: Membership,
    salary_range: tuple[float, float] = (25_000.0, 150_000.0),
) -> tuple[float, str, float]:
    """Возвращает (зарплата, доминирующий терм, сила этого терма)."""
    strength = apply_rules(promotion, satisfaction, balance)
    value = defuzzify(strength, *salary_range)
    category = max(TERMS, key=lambda t: strength[t])
    return value, category, strength[category]
