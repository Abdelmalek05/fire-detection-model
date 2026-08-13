"""How fetch_year_chunk behaves against quota blocks, timeouts and crashes.

Still no live network I/O: the `session=None` seam that already exists in
weather.py is filled with a recording fake, so these assert real control flow
through real code rather than mock interactions.
"""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from firerisk import weather as weather_mod
from firerisk.weather import RateLimitExceeded, fetch_year_chunk

DAILY = {
    "time": ["2023-06-01"],
    "temperature_2m_max": [30.0], "temperature_2m_min": [18.0],
    "temperature_2m_mean": [24.0], "relative_humidity_2m_min": [30],
    "relative_humidity_2m_mean": [55], "wind_speed_10m_max": [12.0],
    "wind_speed_10m_mean": [6.0], "precipitation_sum": [0.0],
    "vapour_pressure_deficit_max": [4.3],
}


class FakeResponse:
    def __init__(self, status_code, payload=None, reason=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else [{"daily": DAILY}]
        self.text = reason
        self._reason = reason

    def json(self):
        if self.status_code == 429:
            return {"reason": self._reason}
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.kwargs = {}

    def get(self, url, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        return self.responses.pop(0)


@pytest.fixture
def cfg(tmp_path):
    return SimpleNamespace(
        data_dir=tmp_path, warmup_start="01-01", season_end="10-31",
        weather_sleep_seconds=0, weather_chunk_size=150,
    )


@pytest.fixture
def cells():
    return pd.DataFrame({"cell_id": ["A"], "lat": [36.7], "lon": [4.1]})


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    monkeypatch.setattr(weather_mod.time, "sleep", lambda *_: None)


# ─────────────────────────────────────────────
# Timeouts
# ─────────────────────────────────────────────

def test_connecting_times_out_far_sooner_than_reading(cfg, cells):
    """A single `timeout=300` sets the connect timeout to 300s as well, so a
    host that accepts the TCP handshake and then goes silent costs five
    minutes before the retry logic even sees a failure. Observed live.

    The two limits describe different things. Reading legitimately takes
    minutes, because a chunk is ~3 MB. Connecting either happens in seconds or
    is never going to happen at all.
    """
    s = FakeSession([FakeResponse(200)])
    fetch_year_chunk(cfg, cells, 2023, 0, session=s)

    connect, read = s.kwargs["timeout"]
    assert connect <= 15, "a dead connection must be abandoned in seconds"
    assert read >= 300, "a healthy but slow response must still get its minutes"


# ─────────────────────────────────────────────
# Quota blocks
# ─────────────────────────────────────────────

def test_an_hourly_block_is_raised_for_the_caller_to_sleep_on(cfg, cells):
    """An hourly or daily block needs a wait far longer than a retry, and
    blocking inside the fetch would hide progress and burn the retry budget.
    Raise, so the caller can sleep on its own schedule and resume from cache.
    """
    s = FakeSession([FakeResponse(429, reason="Hourly API request limit")])
    with pytest.raises(RateLimitExceeded, match="Hourly"):
        fetch_year_chunk(cfg, cells, 2023, 0, session=s)


def test_a_minutely_block_is_simply_waited_out(cfg, cells, monkeypatch):
    """A minutely block clears in a minute - cheaper to sit through than to
    unwind the whole run for."""
    slept = []
    monkeypatch.setattr(weather_mod.time, "sleep", lambda s: slept.append(s))
    s = FakeSession([FakeResponse(429, reason="Minutely limit"),
                     FakeResponse(200)])
    fetch_year_chunk(cfg, cells, 2023, 0, session=s)
    assert 60 in slept


def test_retries_are_bounded_and_report_the_last_error(cfg, cells):
    """Transport failures get the retry budget, then surface as RuntimeError
    so the caller can defer this chunk rather than lose the run."""
    class Dying(FakeSession):
        def get(self, url, **kwargs):
            self.calls += 1
            raise OSError("connection reset by peer")

    s = Dying([])
    with pytest.raises(RuntimeError, match="connection reset"):
        fetch_year_chunk(cfg, cells, 2023, 0, session=s)
    assert s.calls == weather_mod.MAX_RETRIES


# ─────────────────────────────────────────────
# Caching
# ─────────────────────────────────────────────

def test_cached_chunk_is_returned_without_any_request(cfg, cells):
    s = FakeSession([FakeResponse(200)])
    fetch_year_chunk(cfg, cells, 2023, 0, session=s)
    again = fetch_year_chunk(cfg, cells, 2023, 0, session=s)
    assert s.calls == 1 and len(again) == 1


def test_a_crash_mid_write_leaves_no_half_chunk_to_be_trusted_later(
        cfg, cells, monkeypatch):
    """Resumption trusts path.exists(), so a chunk file that exists is treated
    as finished forever. A truncated parquet from an interrupted write would
    be silently accepted for the rest of the project's life."""
    def die_halfway(self, path, *a, **k):
        # Write real bytes first, then die: a truncated parquet on disk is the
        # failure mode, not a missing one. Raising before any write would make
        # this test pass against non-atomic code and prove nothing.
        Path(path).write_bytes(b"PAR1-truncated")
        raise KeyboardInterrupt("died mid-write")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", die_halfway)
    s = FakeSession([FakeResponse(200)])
    with pytest.raises(KeyboardInterrupt):
        fetch_year_chunk(cfg, cells, 2023, 0, session=s)
    assert not weather_mod._cache_path(cfg, 2023, 0).exists()
