import pandas as pd

from firerisk.universe import build_universe

# Cell ids must be real grid ids ("{ilat}_{ilon}"), not opaque labels:
# build_universe derives each cell's centre coordinates by splitting the id.
KABYLIE = "367_41"      # 2 fire-days across 2 distinct years -> in universe
SAME_YEAR = "350_50"    # 2 fire-days but both in one year   -> excluded
SINGLE = "340_30"       # 1 fire-day                          -> excluded
REPEATED_DAY = "360_20"  # same day recorded twice            -> excluded


def _det(rows):
    return pd.DataFrame(
        [{"cell_id": c, "date": pd.Timestamp(d), "year": pd.Timestamp(d).year}
         for c, d in rows]
    )


def test_requires_two_days_and_two_years():
    det = _det([
        (KABYLIE, "2020-07-01"), (KABYLIE, "2021-08-01"),
        (SAME_YEAR, "2020-07-01"), (SAME_YEAR, "2020-07-05"),
        (SINGLE, "2020-07-01"),
        (REPEATED_DAY, "2020-07-01"), (REPEATED_DAY, "2020-07-01"),
    ])
    u = build_universe(det, min_fire_days=2, min_years=2)
    assert set(u.cell_id) == {KABYLIE}
    assert u.loc[u.cell_id == KABYLIE, "n_fire_days"].iloc[0] == 2
    assert u.loc[u.cell_id == KABYLIE, "n_years"].iloc[0] == 2


def test_adds_cell_centre_coordinates():
    det = _det([("367_41", "2020-07-01"), ("367_41", "2021-08-01")])
    u = build_universe(det, 2, 2)
    assert abs(u.lat.iloc[0] - 36.7) < 1e-9
    assert abs(u.lon.iloc[0] - 4.1) < 1e-9


def test_empty_input_returns_empty_frame_with_schema():
    u = build_universe(pd.DataFrame(columns=["cell_id", "date", "year"]), 2, 2)
    assert len(u) == 0
    assert list(u.columns) == ["cell_id", "n_fire_days", "n_years", "lat", "lon"]
