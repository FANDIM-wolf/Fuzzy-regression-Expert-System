import pytest

from career_es.fuzzy import RULE_BASE, apply_rules, defuzzify, fuzzify, fuzzy_salary


def test_partition_sums_to_one():
    for x in [0, 1, 2.5, 5, 7.5, 9, 10]:
        m = fuzzify(x, 1, 10)
        assert m.low + m.medium + m.high == pytest.approx(1.0)


def test_minimum_is_low_not_high():
    # В исходной реализации значение на нижней границе получало "high".
    assert fuzzify(1, 1, 10).category == "low"
    assert fuzzify(10, 1, 10).category == "high"


def test_out_of_range_is_clipped():
    assert fuzzify(-100, 1, 10).category == "low"
    assert fuzzify(1000, 1, 10).category == "high"


def test_rule_base_is_complete():
    assert len(RULE_BASE) == 27
    combos = {tuple(t for _, t in r["conditions"]) for r in RULE_BASE}
    assert len(combos) == 27


def test_every_input_fires_at_least_one_rule():
    for p in [1, 2, 3, 4, 5]:
        for s in [1, 4, 7, 10]:
            for b in [1, 4, 7, 10]:
                strength = apply_rules(
                    fuzzify(p, 1, 5), fuzzify(s, 1, 10), fuzzify(b, 1, 10)
                )
                assert sum(strength.values()) > 0


def test_best_profile_beats_worst_profile():
    best = fuzzy_salary(fuzzify(1, 1, 5), fuzzify(10, 1, 10), fuzzify(10, 1, 10))[0]
    worst = fuzzy_salary(fuzzify(5, 1, 5), fuzzify(1, 1, 10), fuzzify(1, 1, 10))[0]
    assert best > worst


def test_defuzzify_within_range():
    value = defuzzify({"low": 0.3, "medium": 0.7, "high": 0.0}, 25000, 150000)
    assert 25000 <= value <= 150000
