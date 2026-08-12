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
    assert cfg.years == list(range(2012, 2026))
    assert cfg.splits["train"] == [2012, 2021]

def test_missing_key_raises(monkeypatch):
    # Neutralise .env loading so the test verifies load_config's behaviour,
    # not whether this machine happens to have a .env file.
    monkeypatch.setattr("firerisk.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FIRMS_MAP_KEY"):
        load_config(Path("config/config.yaml"))
