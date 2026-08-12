import pandas as pd
import pytest

from firerisk.fwi import ffmc, dmc, dc, isi, bui, fwi, compute_series


def test_canonical_van_wagner_example():
    """Canonical worked example (Van Wagner 1987 / cffdrs).

    temp=17, rh=42, wind=25, rain=0, month=4, startup FFMC/DMC/DC = 85/6/15.
    These six values are the transcription check: a single mistyped constant
    in any formula moves at least one of them well outside tolerance.
    """
    f = ffmc(17, 42, 25, 0, 85)
    d = dmc(17, 42, 0, 6, 4)
    c = dc(17, 0, 15, 4)
    i = isi(25, f)
    b = bui(d, c)
    w = fwi(i, b)
    assert f == pytest.approx(87.6929, abs=0.01)
    assert d == pytest.approx(8.5450, abs=0.01)
    assert c == pytest.approx(19.0136, abs=0.01)
    assert i == pytest.approx(10.8536, abs=0.01)
    assert b == pytest.approx(8.4904, abs=0.01)
    assert w == pytest.approx(10.0963, abs=0.01)


def test_rain_reduces_moisture_codes():
    dry = dmc(25, 30, 0.0, 50, 7)
    wet = dmc(25, 30, 20.0, 50, 7)
    assert wet < dry
    assert dc(25, 30.0, 300, 7) < dc(25, 0.0, 300, 7)


def test_dc_accumulates_without_rain():
    v = 15.0
    for _ in range(10):
        v = dc(30, 0.0, v, 7)
    assert v > 15.0


def test_ffmc_stays_in_valid_range():
    for rh in (1, 50, 100):
        for rain in (0, 5, 50):
            v = ffmc(35, rh, 10, rain, 85)
            assert 0.0 <= v <= 101.0


def test_isi_increases_with_wind():
    assert isi(30, 90) > isi(5, 90)


def test_bui_non_negative_at_zero_inputs():
    assert bui(0.0, 0.0) == 0.0


def test_fwi_rises_with_both_inputs():
    assert fwi(20.0, 100.0) > fwi(5.0, 100.0)
    assert fwi(10.0, 120.0) > fwi(10.0, 20.0)


def test_compute_series_resets_per_cell_and_year():
    dates = pd.date_range("2020-01-01", periods=3).append(
        pd.date_range("2021-01-01", periods=3)
    )
    df = pd.DataFrame({
        "cell_id": ["A"] * 6,
        "date": dates,
        "temp_max": [20.0] * 6,
        "rh_min": [40.0] * 6,
        "wind_max": [10.0] * 6,
        "precip_sum": [0.0] * 6,
    })
    out = compute_series(df)
    assert list(out.columns[-6:]) == ["ffmc", "dmc", "dc", "isi", "bui", "fwi"]
    # DC restarts at the 2021 boundary, so row 3 must be below row 2
    assert out.dc.iloc[3] < out.dc.iloc[2]


def test_compute_series_does_not_leak_between_cells():
    """Cell B's codes must start from the startup values, not inherit A's."""
    dates = pd.date_range("2020-06-01", periods=30)
    hot = pd.DataFrame({
        "cell_id": ["A"] * 30, "date": dates,
        "temp_max": [35.0] * 30, "rh_min": [10.0] * 30,
        "wind_max": [20.0] * 30, "precip_sum": [0.0] * 30,
    })
    fresh = pd.DataFrame({
        "cell_id": ["B"] * 30, "date": dates,
        "temp_max": [35.0] * 30, "rh_min": [10.0] * 30,
        "wind_max": [20.0] * 30, "precip_sum": [0.0] * 30,
    })
    out = compute_series(pd.concat([hot, fresh], ignore_index=True))
    a = out[out.cell_id == "A"].reset_index(drop=True)
    b = out[out.cell_id == "B"].reset_index(drop=True)
    assert a.dc.tolist() == pytest.approx(b.dc.tolist())


def test_compute_series_does_not_mutate_input():
    df = pd.DataFrame({
        "cell_id": ["A"], "date": [pd.Timestamp("2020-06-01")],
        "temp_max": [30.0], "rh_min": [20.0],
        "wind_max": [15.0], "precip_sum": [0.0],
    })
    before = list(df.columns)
    compute_series(df)
    assert list(df.columns) == before
