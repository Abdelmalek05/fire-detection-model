"""The fetch script's outer loop: quota blocks, broken chunks, interrupts.

Imported by path because scripts/ is not a package.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from firerisk import weather as weather_mod
from firerisk.weather import RateLimitExceeded

_spec = importlib.util.spec_from_file_location(
    "fetch_weather", Path("scripts/fetch_weather.py")
)
fetch_weather = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_weather)


@pytest.fixture
def cfg(tmp_path):
    return SimpleNamespace(data_dir=tmp_path, warmup_start="01-01",
                           season_end="10-31", weather_sleep_seconds=0)


@pytest.fixture
def one_chunk():
    return [pd.DataFrame({"cell_id": ["A"], "lat": [36.7], "lon": [4.1]})]


def _land(cfg, year, idx):
    """Write a chunk file, so the loop's remaining() sees real progress."""
    p = weather_mod._cache_path(cfg, year, idx)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cell_id": ["A"]}).to_parquet(p, index=False)


# ─────────────────────────────────────────────
# Quota blocks
# ─────────────────────────────────────────────

def test_a_quota_block_sleeps_and_resumes(cfg, one_chunk, monkeypatch):
    """The whole point of the loop: wait out the window, then continue."""
    slept, calls = [], []

    def fake_fetch(cfg_, cells, year, idx, session=None):
        calls.append(1)
        if len(calls) == 1:
            raise RateLimitExceeded("Hourly API request limit exceeded")
        _land(cfg_, year, idx)

    monkeypatch.setattr(weather_mod, "fetch_year_chunk", fake_fetch)
    code = fetch_weather.fetch_until_done(
        cfg, one_chunk, [(2023, 0)], sleeper=slept.append)

    assert code == 0 and slept and slept[0] > 0


def test_a_daily_block_escalates_only_after_repeated_blocks():
    """One hourly wait is cheap and clears a rolling block. Waiting until UTC
    midnight is not, so it must take several fruitless blocks to justify."""
    hourly = fetch_weather.sleep_seconds_for("Daily limit", consecutive_blocks=1)
    escalated = fetch_weather.sleep_seconds_for("Daily limit", consecutive_blocks=3)
    assert hourly == fetch_weather.SLEEP_AFTER_HOURLY
    assert escalated >= hourly


# ─────────────────────────────────────────────
# Surviving a chunk that simply will not fetch
# ─────────────────────────────────────────────

def test_a_permanently_failing_chunk_is_deferred_not_fatal(cfg, monkeypatch):
    """A chunk whose retries run out raises RuntimeError. Left uncaught it
    ends the whole run, which is precisely the thing an unattended fetch must
    not do. Defer it and keep going: the other chunks are fine."""
    chunks = [pd.DataFrame({"cell_id": [c], "lat": [36.7], "lon": [4.1]})
              for c in ("A", "B")]

    def fake_fetch(cfg_, cells, year, idx, session=None):
        if idx == 0:
            raise RuntimeError("Open-Meteo fetch failed: connection reset")
        _land(cfg_, year, idx)

    monkeypatch.setattr(weather_mod, "fetch_year_chunk", fake_fetch)
    code = fetch_weather.fetch_until_done(
        cfg, chunks, [(2023, 0), (2023, 1)], sleeper=lambda _: None)

    assert weather_mod._cache_path(cfg, 2023, 1).exists()
    assert code == fetch_weather.EXIT_INCOMPLETE


def test_the_loop_terminates_when_every_chunk_is_abandoned(cfg, one_chunk,
                                                           monkeypatch):
    """The failure counter is what stops deferral from becoming a spin. With
    every chunk abandoned there is no work left that could ever succeed, so
    the run must end rather than re-deriving the same todo list forever."""
    tries = []

    def fake_fetch(cfg_, cells, year, idx, session=None):
        tries.append(idx)
        raise RuntimeError("permanently broken")

    monkeypatch.setattr(weather_mod, "fetch_year_chunk", fake_fetch)
    code = fetch_weather.fetch_until_done(
        cfg, one_chunk, [(2023, 0)], sleeper=lambda _: None)

    assert code == fetch_weather.EXIT_INCOMPLETE
    assert len(tries) == fetch_weather.MAX_CHUNK_FAILURES


def test_a_chunk_that_recovers_loses_its_failure_count(cfg, one_chunk,
                                                       monkeypatch):
    """Consecutive failures, not lifetime ones: a chunk that fails twice to a
    transient error and then succeeds must not carry that history forever."""
    calls = []

    def fake_fetch(cfg_, cells, year, idx, session=None):
        calls.append(idx)
        if len(calls) <= 2:
            raise RuntimeError("transient")
        _land(cfg_, year, idx)

    monkeypatch.setattr(weather_mod, "fetch_year_chunk", fake_fetch)
    assert fetch_weather.fetch_until_done(
        cfg, one_chunk, [(2023, 0)], sleeper=lambda _: None) == 0


def test_a_burnt_chunk_does_not_stop_the_other_chunks(cfg, monkeypatch):
    """One chunk hitting a quota block must not abandon the run - the loop
    sleeps and retries the whole todo list."""
    chunks = [pd.DataFrame({"cell_id": [c], "lat": [36.7], "lon": [4.1]})
              for c in ("A", "B")]
    attempts = []

    def fake_fetch(cfg_, cells, year, idx, session=None):
        attempts.append(idx)
        if idx == 0 and attempts.count(0) == 1:
            raise RateLimitExceeded("Hourly API request limit exceeded")
        _land(cfg_, year, idx)

    monkeypatch.setattr(weather_mod, "fetch_year_chunk", fake_fetch)
    code = fetch_weather.fetch_until_done(
        cfg, chunks, [(2023, 0), (2023, 1)], sleeper=lambda _: None)

    assert code == 0
    assert weather_mod._cache_path(cfg, 2023, 1).exists()


# ─────────────────────────────────────────────
# Interrupts and completion
# ─────────────────────────────────────────────

def test_ctrl_c_exits_cleanly_with_progress_saved(cfg, one_chunk, monkeypatch):
    def interrupted(cfg_, cells, year, idx, session=None):
        raise KeyboardInterrupt()

    monkeypatch.setattr(weather_mod, "fetch_year_chunk", interrupted)
    assert fetch_weather.fetch_until_done(
        cfg, one_chunk, [(2023, 0)], sleeper=lambda _: None) == 130


def test_already_complete_work_returns_immediately(cfg, one_chunk, monkeypatch):
    def explode(*a, **k):  # pragma: no cover
        raise AssertionError("must not fetch a chunk already on disk")

    monkeypatch.setattr(weather_mod, "fetch_year_chunk", explode)
    _land(cfg, 2023, 0)
    assert fetch_weather.fetch_until_done(
        cfg, one_chunk, [(2023, 0)], sleeper=lambda _: None) == 0


# ─────────────────────────────────────────────
# CLI surface
# ─────────────────────────────────────────────

def _run(args):
    env = dict(os.environ, PYTHONPATH="src")
    return subprocess.run([sys.executable, "scripts/fetch_weather.py", *args],
                          capture_output=True, text=True, env=env)


def test_help_documents_the_stages():
    r = _run(["--help"])
    assert r.returncode == 0
    assert "--status" in r.stdout and "--rebuild-universe" in r.stdout
