import pandas as pd

from firerisk.panel import (
    season_days, build_positives, build_exclusion, sample_negatives,
)

# Cell ids must be real grid ids ("{ilat}_{ilon}"). sample_negatives() calls
# grid.neighbours(), which parses the id into integer indices - an opaque
# label like "A" raises ValueError.
CELL = "367_41"          # Kabylie
NEIGHBOUR = "368_41"     # directly north of CELL
OTHER = "350_50"         # far away, not a neighbour


def ts(s):
    return pd.Timestamp(s)


def test_season_days_only_covers_season():
    d = season_days([2020, 2021], "06-01", "10-31")
    assert d.min() == ts("2020-06-01")
    assert d.max() == ts("2021-10-31")
    assert ts("2020-12-25") not in d
    assert len(d) == 153 * 2


def test_positives_capped_per_cell_per_season():
    qual = pd.DataFrame({
        "cell_id": [CELL] * 5,
        "date": [ts(f"2020-07-0{i}") for i in range(1, 6)],
        "year": [2020] * 5,
        "frp": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    uni = pd.DataFrame({"cell_id": [CELL]})
    pos = build_positives(qual, uni, cap=2)
    assert len(pos) == 2
    assert set(pos.cell_id) == {CELL}


def test_positives_cap_keeps_strongest_events():
    """The cap must retain the highest-FRP fires, not an arbitrary slice."""
    qual = pd.DataFrame({
        "cell_id": [CELL] * 4,
        "date": [ts("2020-07-01"), ts("2020-07-02"), ts("2020-07-03"), ts("2020-07-04")],
        "year": [2020] * 4,
        "frp": [1.0, 99.0, 2.0, 50.0],
    })
    uni = pd.DataFrame({"cell_id": [CELL]})
    pos = build_positives(qual, uni, cap=2)
    assert set(pos.date) == {ts("2020-07-02"), ts("2020-07-04")}


def test_positives_restricted_to_universe():
    qual = pd.DataFrame({
        "cell_id": [CELL, OTHER],
        "date": [ts("2020-07-01"), ts("2020-07-01")],
        "year": [2020, 2020],
        "frp": [1.0, 1.0],
    })
    uni = pd.DataFrame({"cell_id": [CELL]})
    pos = build_positives(qual, uni, cap=10)
    assert set(pos.cell_id) == {CELL}


def test_exclusion_includes_low_confidence():
    det = pd.DataFrame({
        "cell_id": [CELL, CELL],
        "date": [ts("2020-07-01"), ts("2020-07-20")],
        "confidence": ["h", "l"],
    })
    ex = build_exclusion(det)
    assert ex[CELL] == {ts("2020-07-01"), ts("2020-07-20")}


def test_negatives_respect_all_buffers():
    """The core correctness test: no negative may violate any guard."""
    days = season_days([2020], "06-01", "10-31")
    pos = pd.DataFrame({"cell_id": [CELL], "date": [ts("2020-07-10")]})
    uni = pd.DataFrame({"cell_id": [CELL]})
    exclusion = {
        CELL: {ts("2020-07-10"), ts("2020-08-15")},   # own-cell fire days
        NEIGHBOUR: {ts("2020-09-01")},                # a NEIGHBOUR fire day
    }
    neg = sample_negatives(pos, uni, exclusion, days, k=20, buffer_days=3, seed=1)

    assert len(neg) == 20
    assert set(neg.cell_id) == {CELL}              # same-cell matching
    assert neg.date.duplicated().sum() == 0        # no repeats

    season = set(days)
    for d in neg.date:
        for fire_day in exclusion[CELL]:
            assert abs((d - fire_day).days) > 3    # +/-3 day buffer, both sides
        assert d != ts("2020-09-01")               # neighbour same-day exclusion
        assert d in season                         # inside fire season


def test_negatives_use_absolute_dates_not_day_of_year():
    """A fire on 2020-07-10 must NOT block 2021-07-10."""
    days = season_days([2020, 2021], "06-01", "10-31")
    pos = pd.DataFrame({"cell_id": [CELL], "date": [ts("2020-07-10")]})
    uni = pd.DataFrame({"cell_id": [CELL]})
    exclusion = {CELL: {ts("2020-07-10")}}
    neg = sample_negatives(pos, uni, exclusion, days, k=200, buffer_days=3, seed=7)
    assert ts("2021-07-10") in set(neg.date)


def test_sampler_is_deterministic_under_seed():
    days = season_days([2020], "06-01", "10-31")
    pos = pd.DataFrame({"cell_id": [CELL], "date": [ts("2020-07-10")]})
    uni = pd.DataFrame({"cell_id": [CELL]})
    ex = {CELL: {ts("2020-07-10")}}
    a = sample_negatives(pos, uni, ex, days, k=5, buffer_days=3, seed=42)
    b = sample_negatives(pos, uni, ex, days, k=5, buffer_days=3, seed=42)
    assert list(a.date) == list(b.date)


def test_insufficient_candidates_returns_what_is_available():
    days = season_days([2020], "06-01", "06-10")
    pos = pd.DataFrame({"cell_id": [CELL], "date": [ts("2020-06-05")]})
    uni = pd.DataFrame({"cell_id": [CELL]})
    ex = {CELL: {ts("2020-06-05")}}
    neg = sample_negatives(pos, uni, ex, days, k=50, buffer_days=3, seed=3)
    # 10 season days minus 2020-06-02..06-08 (buffer) = 3 candidates
    assert len(neg) == 3
