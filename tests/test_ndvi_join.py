import numpy as np
import pandas as pd
import pytest

from firerisk.ndvi import attach, climatology, with_anomalies


def _ndvi(rows):
    """(cell_id, start, ndvi) -> a consolidate()-shaped frame."""
    d = pd.DataFrame(rows, columns=["cell_id", "composite_start", "ndvi"])
    d["composite_start"] = pd.to_datetime(d["composite_start"])
    d["composite_end"] = d["composite_start"] + pd.Timedelta(days=15)
    d["composite_doy"] = d["composite_start"].dt.dayofyear
    return d.sort_values(["cell_id", "composite_start"]).reset_index(drop=True)


def _slot(year, doy):
    """The calendar date of composite slot `doy` in `year`.

    MOD13Q1 indexes composites by DAY-OF-YEAR (1, 17, 33, ... 353) and resets
    each Jan 1, so a slot's calendar date shifts by a day in leap years.
    Building fixture dates by hand instead lands leap and common years in
    different slots - which is what this helper exists to prevent.
    """
    return pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)


# --------------------------------------------------------------- climatology

def test_climatology_uses_only_the_baseline_years():
    """2000-2011 only. A normal that includes the modelling years contains
    the very fires being predicted and dampens the anomalies that matter."""
    d = _ndvi([
        ("365_31", _slot(2005, 225), 0.50),
        ("365_31", _slot(2008, 225), 0.60),
        ("365_31", _slot(2021, 225), 0.10),   # modelling year - must be ignored
    ])
    norm = climatology(d, baseline=(2000, 2011))
    assert norm.ndvi_normal.iloc[0] == pytest.approx(0.55)


def test_composite_doy_aligns_slots_across_leap_and_common_years():
    """2008 is a leap year, 2005 is not, so slot 225 falls on 12 August in
    one and 13 August in the other. Grouping the climatology by calendar date
    would split one seasonal slot into two and halve the years behind each
    normal.

    Note this is the OPPOSITE of the fire-season rule, where membership is
    tested on (month, day) precisely because 'June 1' is a calendar concept.
    MODIS composites are defined BY day-of-year, so day-of-year is their key.
    """
    assert _slot(2005, 225).strftime("%m-%d") == "08-13"
    assert _slot(2008, 225).strftime("%m-%d") == "08-12"

    d = _ndvi([("365_31", _slot(y, 225), 0.5) for y in (2005, 2008)])
    assert d.composite_doy.unique().tolist() == [225]
    assert len(climatology(d, baseline=(2000, 2011))) == 1


def test_climatology_is_per_cell_and_per_composite_slot():
    d = _ndvi([
        ("365_31", _slot(2005, 225), 0.50),
        ("365_31", _slot(2005, 161), 0.80),
        ("366_31", _slot(2005, 225), 0.20),
    ])
    norm = climatology(d, baseline=(2000, 2011))
    assert len(norm) == 3
    key = norm.set_index(["cell_id", "composite_doy"]).ndvi_normal
    assert key[("365_31", 225)] == pytest.approx(0.50)
    assert key[("366_31", 225)] == pytest.approx(0.20)


def test_climatology_refuses_a_baseline_overlapping_the_modelling_years():
    """A guard, not a convenience. Silently allowing 2012 into the baseline
    is leakage nothing downstream could detect."""
    d = _ndvi([("365_31", "2005-08-13", 0.50)])
    with pytest.raises(ValueError, match="overlaps"):
        climatology(d, baseline=(2000, 2015))


# ----------------------------------------------------------------- anomalies

def test_anomaly_is_the_departure_from_that_cell_s_own_normal():
    d = _ndvi([
        ("365_31", _slot(2005, 225), 0.60),
        ("365_31", _slot(2021, 225), 0.25),
    ])
    out = with_anomalies(d, climatology(d, baseline=(2000, 2011)))
    got = out[out.composite_start.dt.year == 2021]
    assert got.ndvi_anomaly.iloc[0] == pytest.approx(-0.35)


def test_change_32d_looks_two_composites_back_within_the_same_cell():
    d = _ndvi([
        ("365_31", "2021-06-10", 0.70),
        ("365_31", "2021-06-26", 0.60),
        ("365_31", "2021-07-12", 0.40),
        ("366_31", "2021-07-12", 0.90),
    ])
    norms = climatology(_ndvi([("365_31", "2005-06-10", 0.7)]),
                        baseline=(2000, 2011))
    out = with_anomalies(d, norms)
    got = out[(out.cell_id == "365_31")
              & (out.composite_start == pd.Timestamp("2021-07-12"))]
    assert got.ndvi_change_32d.iloc[0] == pytest.approx(-0.30)


def test_change_32d_does_not_bleed_across_cells():
    """366_31 has one composite and no history, so its change must be NaN -
    never the previous cell's value."""
    d = _ndvi([
        ("365_31", "2021-06-10", 0.70),
        ("365_31", "2021-06-26", 0.60),
        ("365_31", "2021-07-12", 0.40),
        ("366_31", "2021-07-12", 0.90),
    ])
    norms = climatology(_ndvi([("365_31", "2005-06-10", 0.7)]),
                        baseline=(2000, 2011))
    out = with_anomalies(d, norms)
    assert np.isnan(out[out.cell_id == "366_31"].ndvi_change_32d.iloc[0])


# ------------------------------------------------------------------ the join

def _panel(rows):
    d = pd.DataFrame(rows, columns=["cell_id", "date"])
    d["date"] = pd.to_datetime(d["date"])
    return d


def _ready(rows):
    d = _ndvi(rows)
    d["ndvi_normal"] = 0.5
    d["ndvi_anomaly"] = d["ndvi"] - 0.5
    d["ndvi_change_32d"] = 0.0
    return d


def test_attach_carries_the_raw_level_not_only_the_anomaly():
    """The level is the feature that actually earned its place: +0.0128 PR-AUC
    over 12/14 folds, against -0.0009 for the anomaly alone. An attach that
    exported only the anomaly would silently withhold it."""
    ndvi = _ready([("365_31", "2021-06-26", 0.60)])
    out = attach(_panel([("365_31", "2021-07-20")]), ndvi)
    assert out.ndvi.iloc[0] == pytest.approx(0.60)
    assert out.ndvi_normal.iloc[0] == pytest.approx(0.50)


def test_attach_uses_the_most_recent_closed_composite():
    ndvi = _ready([
        ("365_31", "2021-06-26", 0.60),   # closes 2021-07-11
        ("365_31", "2021-07-12", 0.40),   # closes 2021-07-27
    ])
    out = attach(_panel([("365_31", "2021-07-20")]), ndvi)
    # On 07-20 the second window is still open, so the first must be used.
    assert out.ndvi_anomaly.iloc[0] == pytest.approx(0.10)


def test_attach_never_uses_a_window_containing_the_label_date():
    """THE leak test. A fire destroys vegetation, so a composite spanning the
    label date carries the burn scar - the label itself, wearing a disguise."""
    ndvi = _ready([("365_31", "2021-07-12", 0.40)])   # covers 07-12 .. 07-27
    for day in ["2021-07-12", "2021-07-20", "2021-07-27"]:
        out = attach(_panel([("365_31", day)]), ndvi)
        assert out.ndvi_anomaly.isna().all(), f"{day} used an open window"


def test_attach_accepts_the_window_the_day_after_it_closes():
    """The boundary from the other side: 07-28 is the first legitimate day."""
    ndvi = _ready([("365_31", "2021-07-12", 0.40)])   # closes 07-27
    out = attach(_panel([("365_31", "2021-07-28")]), ndvi)
    assert out.ndvi_anomaly.iloc[0] == pytest.approx(-0.10)


def test_stale_days_counts_from_the_window_end():
    ndvi = _ready([("365_31", "2021-07-12", 0.40)])   # closes 07-27
    out = attach(_panel([("365_31", "2021-08-05")]), ndvi)
    assert out.ndvi_stale_days.iloc[0] == 9


def test_attach_does_not_borrow_another_cell_s_vegetation():
    ndvi = _ready([("365_31", "2021-06-26", 0.60)])
    out = attach(_panel([("999_99", "2021-07-20")]), ndvi)
    assert out.ndvi_anomaly.isna().all()


def test_attach_preserves_panel_rows_and_order():
    """A left join that silently dropped rows would shrink the dataset, and
    the row count is the only thing that would show it."""
    ndvi = _ready([("365_31", "2021-06-26", 0.60)])
    panel = _panel([("365_31", "2021-07-20"), ("999_99", "2021-07-20"),
                    ("365_31", "2021-06-01")])
    out = attach(panel, ndvi)
    assert len(out) == 3
    assert out.cell_id.tolist() == panel.cell_id.tolist()
    assert out.date.tolist() == panel.date.tolist()
