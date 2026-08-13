import numpy as np
import pandas as pd

from firerisk.features import add_rolling, assign_split, assemble


def _w(precip, n=40, cell="367_41"):
    return pd.DataFrame({
        "cell_id": [cell] * n,
        "date": pd.date_range("2021-01-01", periods=n),
        "temp_max": [30.0] * n, "temp_min": [15.0] * n, "temp_mean": [22.0] * n,
        "rh_min": [35.0] * n, "rh_mean": [55.0] * n,
        "wind_max": [12.0] * n, "wind_mean": [6.0] * n,
        "vpd_max": [3.0] * n,
        "precip_sum": precip,
    })


def test_rolling_windows_are_backward_looking_and_inclusive():
    precip = [0.0] * 39 + [5.0]
    out = add_rolling(_w(precip))
    assert out.precip_7d.iloc[-1] == 5.0
    assert out.precip_7d.iloc[-2] == 0.0
    assert out.precip_30d.iloc[-1] == 5.0


def test_rolling_window_does_not_see_the_future():
    """Rain on day 20 must not appear in day 19's window - that would leak."""
    precip = [0.0] * 40
    precip[20] = 10.0
    out = add_rolling(_w(precip))
    assert out.precip_7d.iloc[19] == 0.0
    assert out.precip_7d.iloc[20] == 10.0


def test_days_since_rain_counts_correctly():
    precip = [0.0] * 40
    precip[10] = 3.0          # rain above 1mm on index 10
    out = add_rolling(_w(precip))
    assert out.days_since_rain_1mm.iloc[10] == 0
    assert out.days_since_rain_1mm.iloc[13] == 3
    # sub-threshold rain does not reset the counter
    precip[20] = 0.4
    out2 = add_rolling(_w(precip))
    assert out2.days_since_rain_1mm.iloc[20] == 10


def test_rolling_does_not_leak_across_cells():
    a = _w([10.0] * 40, cell="367_41")
    b = _w([0.0] * 40, cell="350_50")
    out = add_rolling(pd.concat([a, b], ignore_index=True))
    assert out.loc[out.cell_id == "350_50", "precip_7d"].iloc[-1] == 0.0


def test_assign_split():
    splits = {"train": [2021, 2023], "val": [2024, 2024], "test": [2025, 2025]}
    assert assign_split(2021, splits) == "train"
    assert assign_split(2023, splits) == "train"
    assert assign_split(2024, splits) == "val"
    assert assign_split(2025, splits) == "test"


def test_assemble_joins_and_drops_unmatched():
    samples = pd.DataFrame({
        "cell_id": ["367_41", "367_41", "999_99"],
        "date": pd.to_datetime(["2021-02-05", "2021-02-06", "2021-02-05"]),
        "label": [1, 0, 1],
        "sample_kind": ["positive", "matched_negative", "positive"],
        "n_detections": [3, 0, 1],
        "max_frp": [12.0, np.nan, 4.0],
    })
    weather = add_rolling(_w([0.0] * 40))
    universe = pd.DataFrame({"cell_id": ["367_41"], "lat": [36.7], "lon": [4.1]})

    class Cfg:
        splits = {"train": [2021, 2023], "val": [2024, 2024], "test": [2025, 2025]}

    out = assemble(samples, weather, universe, Cfg())
    assert len(out) == 2                      # cell 999_99 has no weather -> dropped
    assert set(out.cell_id) == {"367_41"}
    assert "fwi" in out.columns and "ffmc" in out.columns
    assert out.split.unique().tolist() == ["train"]
    assert out.lat.iloc[0] == 36.7


def test_assemble_excludes_coordinates_from_feature_list():
    """lat/lon are carried for reference but must NOT be model features -
    that is the whole point of the same-cell negative design."""
    from firerisk.features import BASE_FEATURES
    for banned in ("lat", "lon", "year", "doy", "cell_id"):
        assert banned not in BASE_FEATURES
    assert "vpd_max" in BASE_FEATURES
    assert "fwi" in BASE_FEATURES
