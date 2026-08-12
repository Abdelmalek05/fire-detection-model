# tests/test_config.py
from pathlib import Path

import pytest

from firerisk.config import load_config

def test_loads_expected_values(monkeypatch):
    monkeypatch.setenv("FIRMS_MAP_KEY", "testkey123")
    cfg = load_config(Path("config/config.yaml"))
    assert cfg.resolution == 0.1
    assert cfg.bbox == (-2.5, 34.0, 9.0, 37.2)
    assert cfg.firms_source == "VIIRS_SNPP_SP"
    assert cfg.firms_day_range == 5
    assert cfg.firms_confidence == ["n", "h"]
    assert cfg.k_negatives == 3
    assert cfg.temporal_buffer_days == 3
    assert cfg.weather_chunk_size == 150
    assert cfg.map_key == "testkey123"
    # FIRMS day_range max is 5 (the API rejects 10), and the VIIRS SNPP archive
    # starts 2012-01-20, so no configured year may precede it.
    assert cfg.year_start >= 2012
    assert cfg.year_end >= cfg.year_start
    assert cfg.years == list(range(cfg.year_start, cfg.year_end + 1))


def test_splits_partition_the_configured_years_exactly(monkeypatch):
    """Scope changes (e.g. 14 years -> 5) must keep splits consistent.

    Asserting the invariant rather than hardcoded years means this catches a
    real misconfiguration - a year with no split, or a year in two splits -
    without breaking every time the scope is narrowed or widened.
    """
    monkeypatch.setenv("FIRMS_MAP_KEY", "testkey123")
    cfg = load_config(Path("config/config.yaml"))

    covered = []
    for name in ("train", "val", "test"):
        lo, hi = cfg.splits[name]
        assert lo <= hi, f"{name} split is inverted"
        covered.extend(range(lo, hi + 1))

    assert sorted(covered) == cfg.years, "splits must cover every year exactly once"
    # Test years must come after training years - never train on the future.
    assert max(range(*_bounds(cfg.splits["train"]))) < min(
        range(*_bounds(cfg.splits["test"]))
    )


def _bounds(pair):
    return pair[0], pair[1] + 1

def test_missing_key_raises(monkeypatch):
    # Neutralise .env loading so the test verifies load_config's behaviour,
    # not whether this machine happens to have a .env file.
    monkeypatch.setattr("firerisk.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FIRMS_MAP_KEY"):
        load_config(Path("config/config.yaml"))
