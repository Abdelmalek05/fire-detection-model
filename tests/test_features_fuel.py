"""Regression tests for the fuel feature's leak guard.

days_since_last_fire is the strongest feature in the dataset (+20% PR-AUC),
and also the easiest to turn into a leak. If the lag is ever removed, the
first test here fails loudly rather than quietly handing the model a rule
that encodes our sampling buffer.
"""
import numpy as np
import pandas as pd

from firerisk.features import NEVER_BURNED, add_days_since_last_fire

CELL = "367_41"
BUFFER = 3


def _qual(dates):
    return pd.DataFrame({
        "cell_id": [CELL] * len(dates),
        "date": pd.to_datetime(list(dates)),
    })


def test_recent_fires_inside_the_buffer_are_invisible():
    """A fire 1 day earlier must NOT be reported - only negatives are barred
    from that range, so counting it would label the row for free."""
    qual = _qual(["2021-07-10", "2021-07-11"])
    s = pd.DataFrame({"cell_id": [CELL], "date": [pd.Timestamp("2021-07-12")]})
    got = add_days_since_last_fire(s, qual, BUFFER).iloc[0]
    assert got != 1.0, "leak: a fire inside the buffer window was counted"
    assert got == NEVER_BURNED, "no fire older than the lag exists here"


def test_older_fires_are_reported_normally():
    qual = _qual(["2021-06-01"])
    s = pd.DataFrame({"cell_id": [CELL], "date": [pd.Timestamp("2021-07-01")]})
    assert add_days_since_last_fire(s, qual, BUFFER).iloc[0] == 30.0


def test_both_classes_share_the_same_minimum():
    """The core anti-leak invariant.

    A positive sits ON a fire day; a negative sits >= buffer+1 away. With the
    lag applied, the smallest value either can report is buffer+1. If a
    positive can report something smaller, the feature separates the classes
    by construction rather than by fuel.
    """
    fires = ["2021-07-10", "2021-07-11", "2021-07-12"]
    qual = _qual(fires)
    positive = pd.DataFrame({"cell_id": [CELL],
                             "date": [pd.Timestamp("2021-07-12")]})
    negative = pd.DataFrame({"cell_id": [CELL],
                             "date": [pd.Timestamp("2021-07-16")]})
    p = add_days_since_last_fire(positive, qual, BUFFER).iloc[0]
    n = add_days_since_last_fire(negative, qual, BUFFER).iloc[0]
    for v in (p, n):
        assert v == NEVER_BURNED or v >= BUFFER + 1, (
            f"value {v} is closer than the buffer allows - the guard is gone"
        )


def test_cell_with_no_prior_fire_gets_the_sentinel():
    qual = _qual(["2021-09-01"])
    s = pd.DataFrame({"cell_id": [CELL], "date": [pd.Timestamp("2021-07-01")]})
    assert add_days_since_last_fire(s, qual, BUFFER).iloc[0] == NEVER_BURNED


def test_fires_in_other_cells_are_ignored():
    qual = pd.DataFrame({
        "cell_id": ["350_50"],
        "date": [pd.Timestamp("2021-06-01")],
    })
    s = pd.DataFrame({"cell_id": [CELL], "date": [pd.Timestamp("2021-07-01")]})
    assert add_days_since_last_fire(s, qual, BUFFER).iloc[0] == NEVER_BURNED


def test_index_is_preserved_for_a_non_default_index():
    qual = _qual(["2021-06-01"])
    s = pd.DataFrame({"cell_id": [CELL, CELL],
                      "date": pd.to_datetime(["2021-07-01", "2021-07-02"])},
                     index=[17, 42])
    out = add_days_since_last_fire(s, qual, BUFFER)
    assert list(out.index) == [17, 42]
    assert np.allclose(out.values, [30.0, 31.0])
