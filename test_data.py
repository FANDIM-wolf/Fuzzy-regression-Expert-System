import pandas as pd

from career_es.data import sat_to_ege


def test_sat_conversion_endpoints():
    assert sat_to_ege(400) == 40.0
    assert sat_to_ege(1600) == 100.0


def test_sat_out_of_range_clips_instead_of_raising():
    assert sat_to_ege(200) == 40.0
    assert sat_to_ege(2000) == 100.0


def test_sat_vectorised():
    result = sat_to_ege(pd.Series([400, 1000, 1600]))
    assert list(result) == [40.0, 70.0, 100.0]
