import pandas as pd
import pytest

from firerisk.ndvi import MIN_PIXELS, consolidate


@pytest.fixture
def raw_dir(tmp_path):
    """Two years of exports in the shape GEE actually writes them: 'mean' is
    the scaled-int NDVI, 'count' the contributing pixels."""
    pd.DataFrame({
        "cell_id": ["365_31", "366_31", "365_31"],
        "start": ["2021-01-01", "2021-01-01", "2021-01-17"],
        "mean": [4200.0, 3100.0, 4500.0],
        "count": [1600, 1580, 1610],
    }).to_csv(tmp_path / "ndvi_2021.csv", index=False)
    pd.DataFrame({
        "cell_id": ["365_31"],
        "start": ["2022-01-01"],
        "mean": [3900.0],
        "count": [1595],
    }).to_csv(tmp_path / "ndvi_2022.csv", index=False)
    return tmp_path


def test_applies_the_scale_factor(raw_dir):
    """MOD13Q1 NDVI is int16 with a 0.0001 scale. Skipping it yields values
    in the thousands, which every downstream threshold would silently accept."""
    d = consolidate(raw_dir)
    got = d[(d.cell_id == "365_31")
            & (d.composite_start == pd.Timestamp("2021-01-01"))]
    assert got.ndvi.iloc[0] == pytest.approx(0.42)


def test_derives_the_window_end_from_the_start(raw_dir):
    """system:time_start is the START of a 16-day window, so the last day
    included is start + 15. The whole leak guard depends on this."""
    d = consolidate(raw_dir)
    row = d.iloc[0]
    assert (row.composite_end - row.composite_start).days == 15


def test_composite_doy_is_stable_across_years(raw_dir):
    """The cadence resets each Jan 1, so DOY 1 in 2021 and DOY 1 in 2022 are
    the same seasonal slot. That is what makes a climatology groupable."""
    d = consolidate(raw_dir)
    doys = d[d.composite_start.dt.month == 1].composite_doy.unique()
    assert set(doys) == {1, 17}


def test_reads_every_year_file(raw_dir):
    d = consolidate(raw_dir)
    assert set(d.composite_start.dt.year) == {2021, 2022}
    assert len(d) == 4


def test_drops_cells_with_too_few_contributing_pixels(tmp_path):
    """A cell reduced from 12 valid pixels out of ~1,660 is cloud, not
    vegetation. Keeping it would feed the model confident noise."""
    pd.DataFrame({
        "cell_id": ["365_31", "366_31"],
        "start": ["2021-07-12", "2021-07-12"],
        "mean": [4200.0, 900.0],
        "count": [1600, MIN_PIXELS - 1],
    }).to_csv(tmp_path / "ndvi_2021.csv", index=False)
    d = consolidate(tmp_path)
    assert d.cell_id.tolist() == ["365_31"]


def test_missing_directory_raises_rather_than_returning_empty(tmp_path):
    """firms.load_detections returning an empty frame for a missing directory
    once turned the best feature into a constant with zero errors. Not again."""
    with pytest.raises(FileNotFoundError, match="ndvi"):
        consolidate(tmp_path / "does_not_exist")
