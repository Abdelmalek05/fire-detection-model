# Algeria Wildfire Dataset Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible pipeline producing `data/processed/dataset.parquet` — ~40–60k labelled (0.1° cell, day) rows over northern Algeria, 2012–2025, with weather and Canadian FWI features.

**Architecture:** Five cached stages — FIRMS ingest → burnable-land universe → positives + buffered same-cell negative sampling → Open-Meteo weather join → FWI/rolling features. Each stage writes an artifact and is independently re-runnable; network stages never re-fetch an existing cache file.

**Tech Stack:** Python 3.13, pandas, pyarrow, requests, PyYAML, python-dotenv, pytest. No geospatial stack (no rasterio/geopandas) in v1.

**Spec:** `docs/superpowers/specs/2026-08-12-algeria-wildfire-risk-dataset-design.md`

## Global Constraints

- Grid resolution `0.1°`; cell id format `"{ilat}_{ilon}"` where `i = floor(coord/0.1 + 0.5)`. Never use Python's `round()` — it is banker's rounding and produces inconsistent cell edges.
- FIRMS source `VIIRS_SNPP_SP` only. Never mix in NOAA-20: it would add a detection-probability step-change at 2018.
- FIRMS `day_range` max is **5** (API rejects 10 with `400 Invalid day range. Expects [1..5]`).
- **Positives** require `type == 0` AND `confidence in {n, h}`. **Exclusion/buffer tests** use `type == 0` at **all** confidences including `l`. These two filters are deliberately different and must not be unified.
- Open-Meteo: max ~200 locations per call (500 → `HTTP 414`); chunk at **150**. Throttle and back off on `HTTP 429`.
- Buffer comparisons use **absolute calendar-date differences**, never day-of-year.
- Secrets: `FIRMS_MAP_KEY` from `.env` only. Never in `config.yaml`, never in source, never in a committed file.
- No test performs live network I/O. Network code is tested via fixtures and monkeypatched transport.
- Season = Jun 1 – Oct 31. Weather warm-up fetch starts Jan 1 for FWI spin-up.
- **NEVER run `git commit` or `git push`.** The user commits personally. Each task ends by running `git add` for its files and *reporting* a proposed commit message as text. `git commit` is denied at the permission layer, so attempting it will fail — stage and report instead.

---

## File Structure

| File | Responsibility |
|---|---|
| `requirements.txt` | pinned dependencies |
| `config/config.yaml` | every tunable; no hardcoded values in source |
| `src/firerisk/config.py` | load YAML + `.env` into a typed `Config` |
| `src/firerisk/grid.py` | lat/lon ↔ cell_id, neighbours (pure functions) |
| `src/firerisk/firms.py` | chunked cached FIRMS fetch, type/confidence filters, snap to grid |
| `src/firerisk/universe.py` | burnable-land mask |
| `src/firerisk/panel.py` | positives, exclusion index, negative sampler |
| `src/firerisk/weather.py` | cached + throttled Open-Meteo client |
| `src/firerisk/fwi.py` | FFMC, DMC, DC, ISI, BUI, FWI |
| `src/firerisk/features.py` | rolling dryness features, final assembly, splits |
| `src/firerisk/build.py` | CLI orchestrator |
| `scripts/00_check_apis.py` | live credential smoke test (run manually, not in pytest) |

---

## Task 1: Scaffolding and configuration

**Files:**
- Create: `requirements.txt`, `config/config.yaml`, `.env.example`, `src/firerisk/__init__.py`, `src/firerisk/config.py`, `pytest.ini`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Config` dataclass with attributes `bbox: tuple[float,float,float,float]`, `resolution: float`, `year_start: int`, `year_end: int`, `season_start: str`, `season_end: str`, `warmup_start: str`, `firms_source: str`, `firms_types: list[int]`, `firms_confidence: list[str]`, `firms_day_range: int`, `k_negatives: int`, `temporal_buffer_days: int`, `max_positives_per_cell_season: int`, `universe_min_fire_days: int`, `universe_min_years: int`, `weather_chunk_size: int`, `splits: dict[str, list[int]]`, `map_key: str`, `data_dir: Path`; and `load_config(path: Path | None = None) -> Config`; property `Config.years -> list[int]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path
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
    monkeypatch.delenv("FIRMS_MAP_KEY", raising=False)
    try:
        load_config(Path("config/config.yaml"))
    except RuntimeError as e:
        assert "FIRMS_MAP_KEY" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'firerisk'`

- [ ] **Step 3: Write the supporting files**

```txt
# requirements.txt
pandas==2.3.3
pyarrow==22.0.0
requests==2.32.3
PyYAML==6.0.2
python-dotenv==1.0.1
scikit-learn==1.7.2
xgboost==3.2.0
pytest==8.3.3
# NOTE: shap is deliberately NOT here. It fails to build on Python 3.13/Windows
# without MSVC Build Tools, and no task in this dataset pipeline imports it.
# It belongs to the modelling spec, which adds it when it is actually needed.
```

```yaml
# config/config.yaml
bbox: [-2.5, 34.0, 9.0, 37.2]
resolution: 0.1
years: [2012, 2025]
season: {start: "06-01", end: "10-31"}
warmup_start: "01-01"
data_dir: "data"
firms:
  source: VIIRS_SNPP_SP
  types: [0]
  confidence: [n, h]
  day_range: 5
sampling:
  k_negatives: 3
  temporal_buffer_days: 3
  max_positives_per_cell_season: 10
  random_seed: 42
universe:
  min_fire_days: 2
  min_distinct_years: 2
weather:
  chunk_size: 150
  sleep_seconds: 6
splits:
  train: [2012, 2021]
  val: [2022, 2023]
  test: [2024, 2025]
```

```txt
# .env.example
FIRMS_MAP_KEY=your_key_from_https://firms.modaps.eosdis.nasa.gov/api/map_key/
```

```ini
# pytest.ini
[pytest]
pythonpath = src
testpaths = tests
```

```python
# src/firerisk/__init__.py
```

```python
# src/firerisk/config.py
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    bbox: tuple
    resolution: float
    year_start: int
    year_end: int
    season_start: str
    season_end: str
    warmup_start: str
    data_dir: Path
    firms_source: str
    firms_types: list
    firms_confidence: list
    firms_day_range: int
    k_negatives: int
    temporal_buffer_days: int
    max_positives_per_cell_season: int
    random_seed: int
    universe_min_fire_days: int
    universe_min_years: int
    weather_chunk_size: int
    weather_sleep_seconds: float
    splits: dict
    map_key: str

    @property
    def years(self) -> list:
        return list(range(self.year_start, self.year_end + 1))


def load_config(path=None) -> Config:
    load_dotenv()
    path = Path(path) if path else Path("config/config.yaml")
    raw = yaml.safe_load(path.read_text())
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "FIRMS_MAP_KEY missing. Copy .env.example to .env and set it."
        )
    return Config(
        bbox=tuple(raw["bbox"]),
        resolution=float(raw["resolution"]),
        year_start=raw["years"][0],
        year_end=raw["years"][1],
        season_start=raw["season"]["start"],
        season_end=raw["season"]["end"],
        warmup_start=raw["warmup_start"],
        data_dir=Path(raw["data_dir"]),
        firms_source=raw["firms"]["source"],
        firms_types=raw["firms"]["types"],
        firms_confidence=raw["firms"]["confidence"],
        firms_day_range=raw["firms"]["day_range"],
        k_negatives=raw["sampling"]["k_negatives"],
        temporal_buffer_days=raw["sampling"]["temporal_buffer_days"],
        max_positives_per_cell_season=raw["sampling"]["max_positives_per_cell_season"],
        random_seed=raw["sampling"]["random_seed"],
        universe_min_fire_days=raw["universe"]["min_fire_days"],
        universe_min_years=raw["universe"]["min_distinct_years"],
        weather_chunk_size=raw["weather"]["chunk_size"],
        weather_sleep_seconds=raw["weather"]["sleep_seconds"],
        splits=raw["splits"],
        map_key=key,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pip install -r requirements.txt && python -m pytest tests/test_config.py -v`
Expected: 2 passed

- [ ] **Step 5: Stage the changes (do NOT commit)**

```bash
git add requirements.txt config/config.yaml .env.example pytest.ini src/firerisk tests/test_config.py
```

Report this proposed commit message to the user; they run the commit themselves:
`feat: project scaffolding and config loader`

---

## Task 2: Grid helpers

**Files:**
- Create: `src/firerisk/grid.py`
- Test: `tests/test_grid.py`

**Interfaces:**
- Consumes: `Config.resolution`
- Produces:
  - `to_cell_id(lat: float, lon: float, res: float = 0.1) -> str`
  - `to_cell_id_vec(lat: pd.Series, lon: pd.Series, res: float = 0.1) -> pd.Series`
  - `cell_center(cell_id: str, res: float = 0.1) -> tuple[float, float]` returning `(lat, lon)`
  - `neighbours(cell_id: str) -> list[str]` — the 8 surrounding cell ids

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grid.py
import pandas as pd
from firerisk.grid import to_cell_id, to_cell_id_vec, cell_center, neighbours

def test_round_trip():
    cid = to_cell_id(36.73, 4.05)
    assert cid == "367_41"
    lat, lon = cell_center(cid)
    assert abs(lat - 36.7) < 1e-9
    assert abs(lon - 4.1) < 1e-9

def test_negative_longitude():
    assert to_cell_id(34.02, -2.46) == "340_-25"
    lat, lon = cell_center("340_-25")
    assert abs(lat - 34.0) < 1e-9
    assert abs(lon - (-2.5)) < 1e-9

def test_boundary_is_half_up_not_bankers():
    # 36.75 -> 367.5 must go UP to 368; Python round() would give 368,
    # but 36.65 -> 366.5 would give 366 under banker's rounding. Both must round up.
    assert to_cell_id(36.75, 0.0).split("_")[0] == "368"
    assert to_cell_id(36.65, 0.0).split("_")[0] == "367"

def test_vectorised_matches_scalar():
    lat = pd.Series([36.73, 34.02, 36.75])
    lon = pd.Series([4.05, -2.46, 0.0])
    got = to_cell_id_vec(lat, lon)
    exp = [to_cell_id(a, b) for a, b in zip(lat, lon)]
    assert list(got) == exp

def test_neighbours():
    n = neighbours("367_41")
    assert len(n) == 8
    assert "367_41" not in n
    assert "366_40" in n and "368_42" in n and "367_40" in n
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_grid.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'firerisk.grid'`

- [ ] **Step 3: Write the implementation**

```python
# src/firerisk/grid.py
"""0.1-degree analysis grid.

Uses explicit half-up rounding (floor(x + 0.5)) rather than Python's round(),
which is banker's rounding and would place .5 boundaries inconsistently.
"""
import numpy as np
import pandas as pd

RES = 0.1


def _idx(value: float, res: float) -> int:
    return int(np.floor(value / res + 0.5))


def to_cell_id(lat: float, lon: float, res: float = RES) -> str:
    return f"{_idx(lat, res)}_{_idx(lon, res)}"


def to_cell_id_vec(lat: pd.Series, lon: pd.Series, res: float = RES) -> pd.Series:
    ilat = np.floor(lat / res + 0.5).astype(int)
    ilon = np.floor(lon / res + 0.5).astype(int)
    return ilat.astype(str) + "_" + ilon.astype(str)


def cell_center(cell_id: str, res: float = RES) -> tuple:
    ilat, ilon = cell_id.split("_")
    return int(ilat) * res, int(ilon) * res


def neighbours(cell_id: str) -> list:
    ilat, ilon = (int(v) for v in cell_id.split("_"))
    return [
        f"{ilat + di}_{ilon + dj}"
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        if not (di == 0 and dj == 0)
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_grid.py -v`
Expected: 5 passed

- [ ] **Step 5: Stage the changes (do NOT commit)**

```bash
git add src/firerisk/grid.py tests/test_grid.py
```

Report this proposed commit message to the user; they run the commit themselves:
`feat: 0.1 degree grid helpers with half-up rounding`

---

## Task 3: FIRMS ingest

**Files:**
- Create: `src/firerisk/firms.py`, `scripts/00_check_apis.py`
- Test: `tests/test_firms.py`

**Interfaces:**
- Consumes: `Config`, `grid.to_cell_id_vec`
- Produces:
  - `season_chunk_starts(year: int, season_start: str, season_end: str, day_range: int) -> list[datetime.date]`
  - `chunk_path(cfg: Config, year: int, start: datetime.date) -> Path`
  - `fetch_chunk(cfg: Config, year: int, start: datetime.date, session=None) -> Path` — cached; returns path, skips fetch if file exists
  - `fetch_all(cfg: Config, session=None) -> None`
  - `load_detections(cfg: Config) -> pd.DataFrame` with columns `cell_id, date (datetime64[ns]), year, confidence, frp, lat, lon` — **all `type==0` at every confidence**
  - `qualifying(det: pd.DataFrame, cfg: Config) -> pd.DataFrame` — subset with `confidence in cfg.firms_confidence`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_firms.py
import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from firerisk.firms import season_chunk_starts, load_detections, qualifying

RAW = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
    "instrument,confidence,version,bright_ti5,frp,daynight,type\n"
    "36.73,4.05,305.0,0.5,0.6,2023-07-20,46,N,VIIRS,n,2,294.0,1.9,N,0\n"
    "36.75,4.06,306.0,0.5,0.6,2023-07-20,47,N,VIIRS,l,2,294.0,2.0,N,0\n"
    "35.10,5.00,320.0,0.5,0.6,2023-07-21,50,N,VIIRS,h,2,300.0,9.0,D,0\n"
    "36.00,3.00,330.0,0.5,0.6,2023-07-21,51,N,VIIRS,h,2,310.0,50.0,D,2\n"
    "36.10,3.10,330.0,0.5,0.6,2023-07-22,52,N,VIIRS,n,2,310.0,50.0,D,3\n"
)


def test_season_chunk_starts_covers_season_with_day_range_5():
    starts = season_chunk_starts(2023, "06-01", "10-31", 5)
    assert starts[0] == dt.date(2023, 6, 1)
    assert starts[-1] <= dt.date(2023, 10, 31)
    assert all((b - a).days == 5 for a, b in zip(starts, starts[1:]))
    assert len(starts) == 31


def test_load_detections_filters_type_and_keeps_all_confidences(tmp_path, monkeypatch):
    d = tmp_path / "raw" / "firms" / "2023"
    d.mkdir(parents=True)
    (d / "2023-07-20.csv").write_text(RAW)

    class Cfg:
        data_dir = tmp_path
        resolution = 0.1
        firms_confidence = ["n", "h"]
        year_start, year_end = 2023, 2023
        @property
        def years(self):
            return [2023]

    det = load_detections(Cfg())
    # type 2 and 3 dropped; all three type-0 rows kept regardless of confidence
    assert len(det) == 3
    assert set(det.confidence) == {"n", "l", "h"}
    assert set(det.cell_id) == {"367_41", "368_41", "351_50"}
    assert det.date.dtype.kind == "M"

    qual = qualifying(det, Cfg())
    assert len(qual) == 2
    assert "l" not in set(qual.confidence)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_firms.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'firerisk.firms'`

- [ ] **Step 3: Write the implementation**

```python
# src/firerisk/firms.py
"""NASA FIRMS ingest: chunked, cached, filtered, snapped to the analysis grid."""
import datetime as dt
import io
import time
from pathlib import Path

import pandas as pd
import requests

from .grid import to_cell_id_vec

BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


def season_chunk_starts(year, season_start, season_end, day_range):
    start = dt.date.fromisoformat(f"{year}-{season_start}")
    end = dt.date.fromisoformat(f"{year}-{season_end}")
    out, d = [], start
    while d <= end:
        out.append(d)
        d += dt.timedelta(days=day_range)
    return out


def chunk_path(cfg, year, start):
    return Path(cfg.data_dir) / "raw" / "firms" / str(year) / f"{start.isoformat()}.csv"


def fetch_chunk(cfg, year, start, session=None):
    path = chunk_path(cfg, year, start)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    w, s, e, n = cfg.bbox
    url = (
        f"{BASE}/{cfg.map_key}/{cfg.firms_source}/{w},{s},{e},{n}"
        f"/{cfg.firms_day_range}/{start.isoformat()}"
    )
    get = (session or requests).get
    last = None
    for attempt in range(4):
        try:
            r = get(url, timeout=180)
            r.raise_for_status()
            path.write_text(r.text)
            return path
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"FIRMS fetch failed for {start}: {last}")


def fetch_all(cfg, session=None):
    for year in cfg.years:
        for start in season_chunk_starts(
            year, cfg.season_start, cfg.season_end, cfg.firms_day_range
        ):
            fetch_chunk(cfg, year, start, session=session)


def load_detections(cfg):
    frames = []
    for year in cfg.years:
        d = Path(cfg.data_dir) / "raw" / "firms" / str(year)
        if not d.exists():
            continue
        for f in sorted(d.glob("*.csv")):
            try:
                df = pd.read_csv(f)
            except pd.errors.EmptyDataError:
                continue
            if df.empty:
                continue
            frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=["cell_id", "date", "year", "confidence", "frp", "lat", "lon"]
        )
    det = pd.concat(frames, ignore_index=True)
    det = det[det["type"] == 0].copy()
    det["date"] = pd.to_datetime(det["acq_date"])
    det["year"] = det["date"].dt.year
    det["cell_id"] = to_cell_id_vec(det["latitude"], det["longitude"], cfg.resolution)
    det = det.rename(columns={"latitude": "lat", "longitude": "lon"})
    det = det[["cell_id", "date", "year", "confidence", "frp", "lat", "lon"]]
    return det.drop_duplicates().reset_index(drop=True)


def qualifying(det, cfg):
    return det[det["confidence"].isin(cfg.firms_confidence)].copy()
```

```python
# scripts/00_check_apis.py
"""Live smoke test for FIRMS + Open-Meteo. Run manually; not part of pytest."""
import sys

import requests

from firerisk.config import load_config

cfg = load_config()

r = requests.get(
    f"https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{cfg.map_key}/ALL",
    timeout=60,
)
print("FIRMS availability:", r.status_code)
print(r.text.splitlines()[0] if r.ok else r.text[:200])
if not r.ok:
    sys.exit(1)

w, s, e, n = cfg.bbox
r = requests.get(
    f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{cfg.map_key}"
    f"/{cfg.firms_source}/{w},{s},{e},{n}/{cfg.firms_day_range}/2023-07-20",
    timeout=180,
)
print("FIRMS area:", r.status_code, len(r.text.splitlines()), "lines")

r = requests.get(
    "https://archive-api.open-meteo.com/v1/archive",
    params={
        "latitude": "36.7", "longitude": "4.1",
        "start_date": "2023-08-01", "end_date": "2023-08-05",
        "daily": "temperature_2m_max,precipitation_sum",
        "timezone": "Africa/Algiers",
    },
    timeout=60,
)
print("Open-Meteo:", r.status_code, r.json().get("daily", {}).get("time"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_firms.py -v`
Expected: 2 passed

- [ ] **Step 5: Verify live access end to end**

Run (PowerShell): `$env:PYTHONPATH="src"; python scripts/00_check_apis.py`
Expected: `FIRMS availability: 200`, `FIRMS area: 200 ... lines`, `Open-Meteo: 200 [...]`
Requires a real `.env` with `FIRMS_MAP_KEY`. This step is the live gate — do not proceed to Task 4 if it fails.

- [ ] **Step 6: Stage the changes (do NOT commit)**

```bash
git add src/firerisk/firms.py scripts/00_check_apis.py tests/test_firms.py
```

Report this proposed commit message to the user; they run the commit themselves:
`feat: cached FIRMS ingest with type and confidence filters`

---

## Task 4: Burnable-land universe

**Files:**
- Create: `src/firerisk/universe.py`
- Test: `tests/test_universe.py`

**Interfaces:**
- Consumes: output of `firms.qualifying`
- Produces: `build_universe(qual: pd.DataFrame, min_fire_days: int, min_years: int) -> pd.DataFrame` with columns `cell_id, n_fire_days, n_years, lat, lon` (lat/lon = cell centre)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_universe.py
import pandas as pd
from firerisk.universe import build_universe

def _det(rows):
    return pd.DataFrame(
        [{"cell_id": c, "date": pd.Timestamp(d), "year": pd.Timestamp(d).year}
         for c, d in rows]
    )

def test_requires_two_days_and_two_years():
    det = _det([
        ("A", "2020-07-01"), ("A", "2021-08-01"),          # 2 days, 2 years -> in
        ("B", "2020-07-01"), ("B", "2020-07-05"),          # 2 days, 1 year  -> out
        ("C", "2020-07-01"),                                # 1 day          -> out
        ("D", "2020-07-01"), ("D", "2020-07-01"),          # duplicate day  -> out
    ])
    u = build_universe(det, min_fire_days=2, min_years=2)
    assert set(u.cell_id) == {"A"}
    assert u.loc[u.cell_id == "A", "n_fire_days"].iloc[0] == 2
    assert u.loc[u.cell_id == "A", "n_years"].iloc[0] == 2

def test_adds_cell_centre_coordinates():
    det = _det([("367_41", "2020-07-01"), ("367_41", "2021-08-01")])
    u = build_universe(det, 2, 2)
    assert abs(u.lat.iloc[0] - 36.7) < 1e-9
    assert abs(u.lon.iloc[0] - 4.1) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_universe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'firerisk.universe'`

- [ ] **Step 3: Write the implementation**

```python
# src/firerisk/universe.py
"""Burnable-land mask: cells with repeated fire activity across distinct years."""
import pandas as pd

from .grid import cell_center


def build_universe(qual, min_fire_days=2, min_years=2):
    if qual.empty:
        return pd.DataFrame(columns=["cell_id", "n_fire_days", "n_years", "lat", "lon"])
    g = qual.groupby("cell_id").agg(
        n_fire_days=("date", "nunique"),
        n_years=("year", "nunique"),
    )
    keep = g[(g.n_fire_days >= min_fire_days) & (g.n_years >= min_years)]
    keep = keep.reset_index()
    coords = keep.cell_id.map(cell_center)
    keep["lat"] = [c[0] for c in coords]
    keep["lon"] = [c[1] for c in coords]
    return keep.reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_universe.py -v`
Expected: 2 passed

- [ ] **Step 5: Stage the changes (do NOT commit)**

```bash
git add src/firerisk/universe.py tests/test_universe.py
```

Report this proposed commit message to the user; they run the commit themselves:
`feat: burnable-land universe mask`

---

## Task 5: Positives and buffered negative sampling

This is the correctness-critical task. The buffer tests are the gate.

**Files:**
- Create: `src/firerisk/panel.py`
- Test: `tests/test_panel.py`

**Interfaces:**
- Consumes: `firms.load_detections` (all confidences), `firms.qualifying`, `universe.build_universe`, `grid.neighbours`
- Produces:
  - `season_days(years: list[int], season_start: str, season_end: str) -> pd.DatetimeIndex`
  - `build_positives(qual: pd.DataFrame, universe: pd.DataFrame, cap: int) -> pd.DataFrame` with `cell_id, date, n_detections, max_frp`
  - `build_exclusion(det_all: pd.DataFrame) -> dict[str, set]` — cell_id → set of `pd.Timestamp` fire days at **all** confidences
  - `sample_negatives(positives, universe, exclusion, days, k, buffer_days, seed) -> pd.DataFrame` with `cell_id, date`
  - `build_samples(det_all, qual, universe, cfg) -> pd.DataFrame` with `cell_id, date, label, sample_kind, n_detections, max_frp`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_panel.py
import pandas as pd
import pytest

from firerisk.panel import (
    season_days, build_positives, build_exclusion, sample_negatives,
)

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
        "cell_id": ["A"] * 5,
        "date": [ts(f"2020-07-0{i}") for i in range(1, 6)],
        "year": [2020] * 5,
        "frp": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    uni = pd.DataFrame({"cell_id": ["A"]})
    pos = build_positives(qual, uni, cap=2)
    assert len(pos) == 2
    assert set(pos.cell_id) == {"A"}

def test_positives_restricted_to_universe():
    qual = pd.DataFrame({
        "cell_id": ["A", "B"],
        "date": [ts("2020-07-01"), ts("2020-07-01")],
        "year": [2020, 2020],
        "frp": [1.0, 1.0],
    })
    uni = pd.DataFrame({"cell_id": ["A"]})
    pos = build_positives(qual, uni, cap=10)
    assert set(pos.cell_id) == {"A"}

def test_exclusion_includes_low_confidence():
    det = pd.DataFrame({
        "cell_id": ["A", "A"],
        "date": [ts("2020-07-01"), ts("2020-07-20")],
        "confidence": ["h", "l"],
    })
    ex = build_exclusion(det)
    assert ex["A"] == {ts("2020-07-01"), ts("2020-07-20")}

def test_negatives_respect_all_buffers():
    """The core correctness test: no negative may violate any guard."""
    days = season_days([2020], "06-01", "10-31")
    pos = pd.DataFrame({"cell_id": ["367_41"], "date": [ts("2020-07-10")]})
    uni = pd.DataFrame({"cell_id": ["367_41"]})
    exclusion = {
        "367_41": {ts("2020-07-10"), ts("2020-08-15")},  # own-cell fire days
        "368_41": {ts("2020-09-01")},                     # a NEIGHBOUR fire day
    }
    neg = sample_negatives(pos, uni, exclusion, days, k=20, buffer_days=3, seed=1)

    assert len(neg) == 20
    assert set(neg.cell_id) == {"367_41"}          # same-cell matching
    assert neg.date.duplicated().sum() == 0        # no repeats

    for d in neg.date:
        for fire_day in exclusion["367_41"]:
            assert abs((d - fire_day).days) > 3    # +/-3 day buffer, both sides
        assert d != ts("2020-09-01")               # neighbour same-day exclusion
        assert d in set(days)                      # inside fire season

def test_negatives_use_absolute_dates_not_day_of_year():
    """A fire on 2020-07-10 must NOT block 2021-07-10."""
    days = season_days([2020, 2021], "06-01", "10-31")
    pos = pd.DataFrame({"cell_id": ["A"], "date": [ts("2020-07-10")]})
    uni = pd.DataFrame({"cell_id": ["A"]})
    exclusion = {"A": {ts("2020-07-10")}}
    neg = sample_negatives(pos, uni, exclusion, days, k=200, buffer_days=3, seed=7)
    assert ts("2021-07-10") in set(neg.date)

def test_sampler_is_deterministic_under_seed():
    days = season_days([2020], "06-01", "10-31")
    pos = pd.DataFrame({"cell_id": ["A"], "date": [ts("2020-07-10")]})
    uni = pd.DataFrame({"cell_id": ["A"]})
    ex = {"A": {ts("2020-07-10")}}
    a = sample_negatives(pos, uni, ex, days, k=5, buffer_days=3, seed=42)
    b = sample_negatives(pos, uni, ex, days, k=5, buffer_days=3, seed=42)
    assert list(a.date) == list(b.date)

def test_insufficient_candidates_returns_what_is_available():
    days = season_days([2020], "06-01", "06-10")
    pos = pd.DataFrame({"cell_id": ["A"], "date": [ts("2020-06-05")]})
    uni = pd.DataFrame({"cell_id": ["A"]})
    ex = {"A": {ts("2020-06-05")}}
    neg = sample_negatives(pos, uni, ex, days, k=50, buffer_days=3, seed=3)
    # 10 season days minus 2020-06-02..06-08 (buffer) = 3 candidates
    assert len(neg) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_panel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'firerisk.panel'`

- [ ] **Step 3: Write the implementation**

```python
# src/firerisk/panel.py
"""Positives and same-cell matched negative sampling.

Negatives are drawn from the SAME cell as the positive, so cell identity is
statistically independent of the label. Three guards apply:
  * own-cell fire days block a +/- buffer_days window (absolute calendar dates)
  * any of the 8 neighbours having a detection blocks that exact day
  * candidate days must lie inside the fire season
Exclusion uses detections at ALL confidences, while positives use only the
qualifying ones - a low-confidence detection is not enough to assert a fire,
but is ample reason not to assert its absence.
"""
import numpy as np
import pandas as pd

from .grid import neighbours


def season_days(years, season_start, season_end):
    parts = [
        pd.date_range(f"{y}-{season_start}", f"{y}-{season_end}", freq="D")
        for y in years
    ]
    return pd.DatetimeIndex(np.concatenate([p.values for p in parts]))


def build_positives(qual, universe, cap):
    if qual.empty:
        return pd.DataFrame(columns=["cell_id", "date", "n_detections", "max_frp"])
    q = qual[qual.cell_id.isin(set(universe.cell_id))]
    if q.empty:
        return pd.DataFrame(columns=["cell_id", "date", "n_detections", "max_frp"])
    agg = (
        q.groupby(["cell_id", "date"])
        .agg(n_detections=("date", "size"), max_frp=("frp", "max"))
        .reset_index()
    )
    agg["year"] = agg["date"].dt.year
    # Cap per cell per season, keeping the strongest events by FRP.
    agg = (
        agg.sort_values(["cell_id", "year", "max_frp"], ascending=[True, True, False])
        .groupby(["cell_id", "year"], group_keys=False)
        .head(cap)
    )
    return (
        agg.drop(columns="year")
        .sort_values(["cell_id", "date"])
        .reset_index(drop=True)
    )


def build_exclusion(det_all):
    if det_all.empty:
        return {}
    return {c: set(g) for c, g in det_all.groupby("cell_id")["date"]}


def sample_negatives(positives, universe, exclusion, days, k, buffer_days, seed):
    rng = np.random.default_rng(seed)
    day_values = pd.DatetimeIndex(days)
    out = []

    for cell_id, grp in positives.groupby("cell_id"):
        n_needed = len(grp) * k

        blocked = np.zeros(len(day_values), dtype=bool)

        # Guard 1: own-cell fire days, +/- buffer_days, absolute calendar dates.
        for fire_day in exclusion.get(cell_id, set()):
            delta = np.abs((day_values - fire_day).days)
            blocked |= delta <= buffer_days

        # Guard 2: any neighbour detection blocks that exact day.
        for nb in neighbours(cell_id):
            for nb_day in exclusion.get(nb, set()):
                blocked |= day_values == nb_day

        candidates = day_values[~blocked]
        if len(candidates) == 0:
            continue
        take = min(n_needed, len(candidates))
        chosen = rng.choice(len(candidates), size=take, replace=False)
        for idx in chosen:
            out.append({"cell_id": cell_id, "date": candidates[idx]})

    if not out:
        return pd.DataFrame(columns=["cell_id", "date"])
    return pd.DataFrame(out).sort_values(["cell_id", "date"]).reset_index(drop=True)


def build_samples(det_all, qual, universe, cfg):
    positives = build_positives(qual, universe, cfg.max_positives_per_cell_season)
    exclusion = build_exclusion(det_all)
    days = season_days(cfg.years, cfg.season_start, cfg.season_end)
    negatives = sample_negatives(
        positives, universe, exclusion, days,
        cfg.k_negatives, cfg.temporal_buffer_days, cfg.random_seed,
    )
    positives = positives.assign(label=1, sample_kind="positive")
    negatives = negatives.assign(
        label=0, sample_kind="matched_negative", n_detections=0, max_frp=np.nan
    )
    cols = ["cell_id", "date", "label", "sample_kind", "n_detections", "max_frp"]
    return (
        pd.concat([positives[cols], negatives[cols]], ignore_index=True)
        .sort_values(["cell_id", "date"])
        .reset_index(drop=True)
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_panel.py -v`
Expected: 8 passed

- [ ] **Step 5: Stage the changes (do NOT commit)**

```bash
git add src/firerisk/panel.py tests/test_panel.py
```

Report this proposed commit message to the user; they run the commit themselves:
`feat: positives and buffered same-cell negative sampling`

---

## Task 6: Open-Meteo weather client

**Files:**
- Create: `src/firerisk/weather.py`
- Test: `tests/test_weather.py`

**Interfaces:**
- Consumes: `Config`, `universe` DataFrame (`cell_id, lat, lon`)
- Produces:
  - `DAILY_VARS: list[str]`
  - `chunk_cells(universe: pd.DataFrame, size: int) -> list[pd.DataFrame]`
  - `parse_response(payload: list | dict, cells: pd.DataFrame) -> pd.DataFrame` with `cell_id, date, temp_max, temp_min, temp_mean, rh_min, rh_mean, wind_max, wind_mean, precip_sum`
  - `fetch_year_chunk(cfg, cells, year, chunk_idx, session=None) -> pd.DataFrame` — cached parquet
  - `load_weather(cfg, universe, session=None) -> pd.DataFrame`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_weather.py
import pandas as pd
from firerisk.weather import chunk_cells, parse_response, DAILY_VARS

def test_chunking_respects_size_limit():
    uni = pd.DataFrame({"cell_id": [str(i) for i in range(325)],
                        "lat": [36.0] * 325, "lon": [4.0] * 325})
    chunks = chunk_cells(uni, 150)
    assert [len(c) for c in chunks] == [150, 150, 25]
    assert all(len(c) <= 200 for c in chunks)  # API hard limit

def test_parse_response_maps_results_to_cells_in_order():
    cells = pd.DataFrame({"cell_id": ["A", "B"], "lat": [36.7, 36.8], "lon": [4.1, 4.2]})
    daily = {
        "time": ["2023-06-01", "2023-06-02"],
        "temperature_2m_max": [30.0, 31.0],
        "temperature_2m_min": [18.0, 19.0],
        "temperature_2m_mean": [24.0, 25.0],
        "relative_humidity_2m_min": [30, 25],
        "relative_humidity_2m_mean": [55, 50],
        "wind_speed_10m_max": [12.0, 15.0],
        "wind_speed_10m_mean": [6.0, 7.0],
        "precipitation_sum": [0.0, 2.5],
    }
    payload = [{"daily": daily}, {"daily": daily}]
    df = parse_response(payload, cells)
    assert len(df) == 4
    assert set(df.cell_id) == {"A", "B"}
    assert df.date.dtype.kind == "M"
    assert df.loc[df.cell_id == "A", "precip_sum"].tolist() == [0.0, 2.5]

def test_parse_response_handles_single_location_dict():
    cells = pd.DataFrame({"cell_id": ["A"], "lat": [36.7], "lon": [4.1]})
    payload = {"daily": {
        "time": ["2023-06-01"],
        "temperature_2m_max": [30.0], "temperature_2m_min": [18.0],
        "temperature_2m_mean": [24.0], "relative_humidity_2m_min": [30],
        "relative_humidity_2m_mean": [55], "wind_speed_10m_max": [12.0],
        "wind_speed_10m_mean": [6.0], "precipitation_sum": [0.0],
    }}
    df = parse_response(payload, cells)
    assert len(df) == 1 and df.cell_id.iloc[0] == "A"

def test_daily_vars_are_the_documented_set():
    assert DAILY_VARS == [
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "relative_humidity_2m_min", "relative_humidity_2m_mean",
        "wind_speed_10m_max", "wind_speed_10m_mean", "precipitation_sum",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_weather.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'firerisk.weather'`

- [ ] **Step 3: Write the implementation**

```python
# src/firerisk/weather.py
"""Open-Meteo archive client: chunked, throttled, cached to parquet.

Measured API limits: ~200 locations per call (500 returns HTTP 414), and a
weighted minutely rate limit that returns HTTP 429. Chunk at 150 and sleep
between calls.
"""
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "relative_humidity_2m_min", "relative_humidity_2m_mean",
    "wind_speed_10m_max", "wind_speed_10m_mean", "precipitation_sum",
]

RENAME = {
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "temperature_2m_mean": "temp_mean",
    "relative_humidity_2m_min": "rh_min",
    "relative_humidity_2m_mean": "rh_mean",
    "wind_speed_10m_max": "wind_max",
    "wind_speed_10m_mean": "wind_mean",
    "precipitation_sum": "precip_sum",
}


def chunk_cells(universe, size):
    return [
        universe.iloc[i:i + size].reset_index(drop=True)
        for i in range(0, len(universe), size)
    ]


def parse_response(payload, cells):
    results = payload if isinstance(payload, list) else [payload]
    frames = []
    for cell_id, result in zip(cells.cell_id, results):
        daily = result["daily"]
        df = pd.DataFrame({RENAME[v]: daily[v] for v in DAILY_VARS})
        df.insert(0, "date", pd.to_datetime(daily["time"]))
        df.insert(0, "cell_id", cell_id)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _cache_path(cfg, year, chunk_idx):
    return (
        Path(cfg.data_dir) / "raw" / "weather" / str(year) / f"chunk_{chunk_idx:03d}.parquet"
    )


def fetch_year_chunk(cfg, cells, year, chunk_idx, session=None):
    path = _cache_path(cfg, year, chunk_idx)
    if path.exists():
        return pd.read_parquet(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "latitude": ",".join(f"{v:.4f}" for v in cells.lat),
        "longitude": ",".join(f"{v:.4f}" for v in cells.lon),
        "start_date": f"{year}-{cfg.warmup_start}",
        "end_date": f"{year}-{cfg.season_end}",
        "daily": ",".join(DAILY_VARS),
        "timezone": "Africa/Algiers",
    }
    get = (session or requests).get
    last = None
    for attempt in range(5):
        try:
            r = get(BASE, params=params, timeout=300)
            if r.status_code == 429:
                time.sleep(60)
                continue
            r.raise_for_status()
            df = parse_response(r.json(), cells)
            df.to_parquet(path, index=False)
            time.sleep(cfg.weather_sleep_seconds)
            return df
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Open-Meteo fetch failed year={year} chunk={chunk_idx}: {last}")


def load_weather(cfg, universe, session=None):
    chunks = chunk_cells(universe, cfg.weather_chunk_size)
    frames = []
    for year in cfg.years:
        for i, cells in enumerate(chunks):
            frames.append(fetch_year_chunk(cfg, cells, year, i, session=session))
    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_weather.py -v`
Expected: 4 passed

- [ ] **Step 5: Stage the changes (do NOT commit)**

```bash
git add src/firerisk/weather.py tests/test_weather.py
```

Report this proposed commit message to the user; they run the commit themselves:
`feat: cached throttled Open-Meteo archive client`

---

## Task 7: Canadian FWI system

**Files:**
- Create: `src/firerisk/fwi.py`
- Test: `tests/test_fwi.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `ffmc(temp, rh, wind, rain, ffmc_prev) -> float`
  - `dmc(temp, rh, rain, dmc_prev, month) -> float`
  - `dc(temp, rain, dc_prev, month) -> float`
  - `isi(wind, ffmc_val) -> float`
  - `bui(dmc_val, dc_val) -> float`
  - `fwi(isi_val, bui_val) -> float`
  - `compute_series(df: pd.DataFrame) -> pd.DataFrame` — per-cell chronological pass adding columns `ffmc, dmc, dc, isi, bui, fwi`; expects `cell_id, date, temp_max, rh_min, wind_max, precip_sum`

Startup values `FFMC=85, DMC=6, DC=15` at each year's first row.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fwi.py
import pandas as pd
import pytest

from firerisk.fwi import ffmc, dmc, dc, isi, bui, fwi, compute_series

# Canonical worked example (Van Wagner 1987 / cffdrs):
# temp=17, rh=42, wind=25, rain=0, month=4, startup 85/6/15
def test_canonical_van_wagner_example():
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fwi.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'firerisk.fwi'`

- [ ] **Step 3: Write the implementation**

```python
# src/firerisk/fwi.py
"""Canadian Forest Fire Weather Index System (Van Wagner 1987).

Driven by daily aggregates (max temp, min RH, max wind, 24h precipitation)
rather than the canonical noon-local observations - the standard published
approximation. Day-length factors are the published tables calibrated for
~46 N; northern Algeria is ~36 N, so these are an accepted approximation.
"""
import math

import numpy as np
import pandas as pd

DAY_LENGTH = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
DAY_LENGTH_DC = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]

FFMC_START, DMC_START, DC_START = 85.0, 6.0, 15.0


def ffmc(temp, rh, wind, rain, ffmc_prev):
    rh = min(rh, 100.0)
    mo = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)
    if rain > 0.5:
        rf = rain - 0.5
        mo += 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
        if mo > 150.0:
            mo += 0.0015 * (mo - 150.0) ** 2 * math.sqrt(rf)
        mo = min(mo, 250.0)

    ed = (
        0.942 * rh ** 0.679
        + 11.0 * math.exp((rh - 100.0) / 10.0)
        + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh))
    )
    if mo > ed:
        ko = 0.424 * (1.0 - (rh / 100.0) ** 1.7) + 0.0694 * math.sqrt(wind) * (
            1.0 - (rh / 100.0) ** 8
        )
        kd = ko * 0.581 * math.exp(0.0365 * temp)
        m = ed + (mo - ed) * 10.0 ** (-kd)
    else:
        ew = (
            0.618 * rh ** 0.753
            + 10.0 * math.exp((rh - 100.0) / 10.0)
            + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh))
        )
        if mo < ew:
            kl = 0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7) + 0.0694 * math.sqrt(
                wind
            ) * (1.0 - ((100.0 - rh) / 100.0) ** 8)
            kw = kl * 0.581 * math.exp(0.0365 * temp)
            m = ew - (ew - mo) * 10.0 ** (-kw)
        else:
            m = mo
    value = 59.5 * (250.0 - m) / (147.2 + m)
    return float(min(max(value, 0.0), 101.0))


def dmc(temp, rh, rain, dmc_prev, month):
    temp = max(temp, -1.1)
    le = DAY_LENGTH[month - 1]
    rk = 1.894 * (temp + 1.1) * (100.0 - rh) * le * 1e-4
    if rain > 1.5:
        rw = 0.92 * rain - 1.27
        wmi = 20.0 + 280.0 / math.exp(0.023 * dmc_prev)
        if dmc_prev <= 33.0:
            b = 100.0 / (0.5 + 0.3 * dmc_prev)
        elif dmc_prev <= 65.0:
            b = 14.0 - 1.3 * math.log(dmc_prev)
        else:
            b = 6.2 * math.log(dmc_prev) - 17.2
        wmr = wmi + 1000.0 * rw / (48.77 + b * rw)
        pr = 43.43 * (5.6348 - math.log(max(wmr - 20.0, 1e-6)))
        base = max(pr, 0.0)
    else:
        base = dmc_prev
    return float(max(base + rk, 0.0))


def dc(temp, rain, dc_prev, month):
    temp = max(temp, -2.8)
    lf = DAY_LENGTH_DC[month - 1]
    pe = max((0.36 * (temp + 2.8) + lf) / 2.0, 0.0)
    if rain > 2.8:
        rw = 0.83 * rain - 1.27
        smi = 800.0 * math.exp(-dc_prev / 400.0)
        base = max(dc_prev - 400.0 * math.log(1.0 + 3.937 * rw / smi), 0.0)
    else:
        base = dc_prev
    return float(max(base + pe, 0.0))


def isi(wind, ffmc_val):
    f_wind = math.exp(0.05039 * wind)
    m = 147.2 * (101.0 - ffmc_val) / (59.5 + ffmc_val)
    f_f = 91.9 * math.exp(-0.1386 * m) * (1.0 + m ** 5.31 / 4.93e7)
    return float(0.208 * f_wind * f_f)


def bui(dmc_val, dc_val):
    denom = dmc_val + 0.4 * dc_val
    if denom <= 0:
        return 0.0
    if dmc_val <= 0.4 * dc_val:
        value = 0.8 * dmc_val * dc_val / denom
    else:
        value = dmc_val - (1.0 - 0.8 * dc_val / denom) * (
            0.92 + (0.0114 * dmc_val) ** 1.7
        )
    return float(max(value, 0.0))


def fwi(isi_val, bui_val):
    if bui_val <= 80.0:
        f_d = 0.626 * bui_val ** 0.809 + 2.0
    else:
        f_d = 1000.0 / (25.0 + 108.64 * math.exp(-0.023 * bui_val))
    b = 0.1 * isi_val * f_d
    if b > 1.0:
        return float(math.exp(2.72 * (0.434 * math.log(b)) ** 0.647))
    return float(b)


def compute_series(df):
    """Chronological pass per (cell_id, year). Codes restart at each year.

    Returns a new frame; never mutates the caller's DataFrame.
    """
    df = df.sort_values(["cell_id", "date"]).reset_index(drop=True).copy()
    out = {k: np.empty(len(df)) for k in ("ffmc", "dmc", "dc", "isi", "bui", "fwi")}

    key = list(zip(df.cell_id, df.date.dt.year))
    prev_key = None
    f_prev, d_prev, c_prev = FFMC_START, DMC_START, DC_START

    for i, row in enumerate(df.itertuples(index=False)):
        if key[i] != prev_key:
            f_prev, d_prev, c_prev = FFMC_START, DMC_START, DC_START
            prev_key = key[i]
        month = row.date.month
        f = ffmc(row.temp_max, row.rh_min, row.wind_max, row.precip_sum, f_prev)
        d = dmc(row.temp_max, row.rh_min, row.precip_sum, d_prev, month)
        c = dc(row.temp_max, row.precip_sum, c_prev, month)
        i_val = isi(row.wind_max, f)
        b_val = bui(d, c)
        out["ffmc"][i], out["dmc"][i], out["dc"][i] = f, d, c
        out["isi"][i], out["bui"][i] = i_val, b_val
        out["fwi"][i] = fwi(i_val, b_val)
        f_prev, d_prev, c_prev = f, d, c

    for k, v in out.items():
        df[k] = v
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fwi.py -v`
Expected: 7 passed

- [ ] **Step 5: Stage the changes (do NOT commit)**

```bash
git add src/firerisk/fwi.py tests/test_fwi.py
```

Report this proposed commit message to the user; they run the commit themselves:
`feat: Canadian FWI system components`

---

## Task 8: Rolling features and assembly

**Files:**
- Create: `src/firerisk/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: weather DataFrame, samples DataFrame, `fwi.compute_series`, `Config.splits`
- Produces:
  - `add_rolling(weather: pd.DataFrame) -> pd.DataFrame` adding `precip_7d, precip_30d, days_since_rain_1mm, temp_max_7d_mean, rh_min_7d_mean`
  - `assign_split(year: int, splits: dict) -> str`
  - `assemble(samples, weather_feat, universe, cfg) -> pd.DataFrame` — final dataset

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features.py
import pandas as pd
import numpy as np
import pytest

from firerisk.features import add_rolling, assign_split, assemble

def _w(precip, n=40, cell="A"):
    return pd.DataFrame({
        "cell_id": [cell] * n,
        "date": pd.date_range("2020-01-01", periods=n),
        "temp_max": [30.0] * n, "temp_min": [15.0] * n, "temp_mean": [22.0] * n,
        "rh_min": [35.0] * n, "rh_mean": [55.0] * n,
        "wind_max": [12.0] * n, "wind_mean": [6.0] * n,
        "precip_sum": precip,
    })

def test_rolling_windows_are_backward_looking_and_inclusive():
    precip = [0.0] * 39 + [5.0]
    out = add_rolling(_w(precip))
    assert out.precip_7d.iloc[-1] == 5.0
    assert out.precip_7d.iloc[-2] == 0.0
    assert out.precip_30d.iloc[-1] == 5.0

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
    a = _w([10.0] * 40, cell="A")
    b = _w([0.0] * 40, cell="B")
    out = add_rolling(pd.concat([a, b], ignore_index=True))
    assert out.loc[out.cell_id == "B", "precip_7d"].iloc[-1] == 0.0

def test_assign_split():
    splits = {"train": [2012, 2021], "val": [2022, 2023], "test": [2024, 2025]}
    assert assign_split(2015, splits) == "train"
    assert assign_split(2022, splits) == "val"
    assert assign_split(2025, splits) == "test"

def test_assemble_joins_and_drops_unmatched():
    samples = pd.DataFrame({
        "cell_id": ["A", "A", "Z"],
        "date": pd.to_datetime(["2020-02-05", "2020-02-06", "2020-02-05"]),
        "label": [1, 0, 1],
        "sample_kind": ["positive", "matched_negative", "positive"],
        "n_detections": [3, 0, 1],
        "max_frp": [12.0, np.nan, 4.0],
    })
    weather = add_rolling(_w([0.0] * 40))
    universe = pd.DataFrame({"cell_id": ["A"], "lat": [36.7], "lon": [4.1]})

    class Cfg:
        splits = {"train": [2012, 2021], "val": [2022, 2023], "test": [2024, 2025]}

    out = assemble(samples, weather, universe, Cfg())
    assert len(out) == 2                      # cell Z has no weather -> dropped
    assert set(out.cell_id) == {"A"}
    assert "fwi" in out.columns and "ffmc" in out.columns
    assert out.split.unique().tolist() == ["train"]
    assert out.lat.iloc[0] == 36.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'firerisk.features'`

- [ ] **Step 3: Write the implementation**

```python
# src/firerisk/features.py
"""Rolling dryness features and final dataset assembly."""
import numpy as np
import pandas as pd

from .fwi import compute_series

BASE_FEATURES = [
    "temp_max", "temp_min", "temp_mean", "rh_min", "rh_mean",
    "wind_max", "wind_mean", "precip_sum",
    "precip_7d", "precip_30d", "days_since_rain_1mm",
    "temp_max_7d_mean", "rh_min_7d_mean",
    "ffmc", "dmc", "dc", "isi", "bui", "fwi",
]


def _days_since_rain(precip):
    out = np.empty(len(precip), dtype=float)
    counter = np.nan
    for i, p in enumerate(precip):
        if p >= 1.0:
            counter = 0.0
        elif not np.isnan(counter):
            counter += 1.0
        out[i] = counter
    return out


def _roll(df, col, window, how):
    """Grouped rolling that returns a Series aligned to df's index.

    Uses groupby(...).rolling(...) then drops the group level - the
    groupby.apply form can return a MultiIndex and misalign on assignment.
    """
    r = df.groupby("cell_id")[col].rolling(window, min_periods=1)
    s = r.sum() if how == "sum" else r.mean()
    return s.reset_index(level=0, drop=True).sort_index()


def add_rolling(weather):
    df = weather.sort_values(["cell_id", "date"]).reset_index(drop=True)
    df["precip_7d"] = _roll(df, "precip_sum", 7, "sum")
    df["precip_30d"] = _roll(df, "precip_sum", 30, "sum")
    df["temp_max_7d_mean"] = _roll(df, "temp_max", 7, "mean")
    df["rh_min_7d_mean"] = _roll(df, "rh_min", 7, "mean")
    df["days_since_rain_1mm"] = (
        df.groupby("cell_id")["precip_sum"]
        .transform(lambda s: pd.Series(_days_since_rain(s.to_numpy()), index=s.index))
    )
    return df


def assign_split(year, splits):
    for name, (lo, hi) in splits.items():
        if lo <= year <= hi:
            return name
    return "unassigned"


def assemble(samples, weather_feat, universe, cfg):
    weather_feat = compute_series(weather_feat)
    merged = samples.merge(weather_feat, on=["cell_id", "date"], how="inner")
    merged = merged.merge(universe[["cell_id", "lat", "lon"]], on="cell_id", how="left")
    merged["year"] = merged["date"].dt.year
    merged["doy"] = merged["date"].dt.dayofyear
    merged["split"] = merged["year"].map(lambda y: assign_split(y, cfg.splits))
    cols = (
        ["cell_id", "lat", "lon", "date", "year", "doy", "label", "sample_kind",
         "n_detections", "max_frp"]
        + BASE_FEATURES
        + ["split"]
    )
    return merged[cols].sort_values(["cell_id", "date"]).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_features.py -v`
Expected: 5 passed

- [ ] **Step 5: Stage the changes (do NOT commit)**

```bash
git add src/firerisk/features.py tests/test_features.py
```

Report this proposed commit message to the user; they run the commit themselves:
`feat: rolling dryness features and dataset assembly`

---

## Task 9: Orchestrator and full run

**Files:**
- Create: `src/firerisk/build.py`, `README.md`
- Test: `tests/test_build.py`

**Interfaces:**
- Consumes: every prior module
- Produces: `main(argv: list[str] | None = None) -> int`; writes `data/interim/universe.parquet`, `data/interim/samples.parquet`, `data/processed/dataset.parquet`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build.py
import subprocess
import sys

def test_cli_help_lists_stages():
    import os
    env = dict(os.environ, PYTHONPATH="src")   # inherit PATH; empty PATH breaks python on Windows
    r = subprocess.run(
        [sys.executable, "-m", "firerisk.build", "--help"],
        capture_output=True, text=True, env=env,
    )
    assert r.returncode == 0
    for stage in ("firms", "universe", "samples", "weather", "features", "all"):
        assert stage in r.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_build.py -v`
Expected: FAIL with `No module named firerisk.build`

- [ ] **Step 3: Write the implementation**

```python
# src/firerisk/build.py
"""CLI orchestrator for the dataset pipeline."""
import argparse
import sys
from pathlib import Path

import pandas as pd

from . import firms, universe as universe_mod, panel, weather as weather_mod, features
from .config import load_config

STAGES = ["firms", "universe", "samples", "weather", "features", "all"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the Algeria wildfire dataset.")
    ap.add_argument("stage", choices=STAGES, help=f"one of: {', '.join(STAGES)}")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config))
    interim = Path(cfg.data_dir) / "interim"
    processed = Path(cfg.data_dir) / "processed"
    interim.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    run_all = args.stage == "all"

    if run_all or args.stage == "firms":
        print("[1/5] FIRMS ingest ...")
        firms.fetch_all(cfg)

    det_all = firms.load_detections(cfg)
    qual = firms.qualifying(det_all, cfg)
    print(f"      detections: {len(det_all)} (qualifying {len(qual)})")

    if run_all or args.stage == "universe":
        print("[2/5] universe ...")
        uni = universe_mod.build_universe(
            qual, cfg.universe_min_fire_days, cfg.universe_min_years
        )
        uni.to_parquet(interim / "universe.parquet", index=False)
        print(f"      universe cells: {len(uni)}")

    uni = pd.read_parquet(interim / "universe.parquet")

    if run_all or args.stage == "samples":
        print("[3/5] positives + negatives ...")
        samples = panel.build_samples(det_all, qual, uni, cfg)
        samples.to_parquet(interim / "samples.parquet", index=False)
        n_pos = int((samples.label == 1).sum())
        print(f"      samples: {len(samples)} (positives {n_pos}, "
              f"negatives {len(samples) - n_pos})")

    if run_all or args.stage == "weather":
        print("[4/5] weather ...")
        weather_mod.load_weather(cfg, uni)
        print("      weather cached")

    if run_all or args.stage == "features":
        print("[5/5] features ...")
        samples = pd.read_parquet(interim / "samples.parquet")
        wx = weather_mod.load_weather(cfg, uni)
        wx = features.add_rolling(wx)
        dataset = features.assemble(samples, wx, uni, cfg)
        dataset.to_parquet(processed / "dataset.parquet", index=False)
        print(f"      dataset rows: {len(dataset)}")
        print(dataset.groupby(["split", "label"]).size().to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_build.py -v`
Expected: 1 passed

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -v`
Expected: 34 passed

- [ ] **Step 6: Execute the real pipeline**

Run (PowerShell): `$env:PYTHONPATH="src"; python -m firerisk.build all`
Expected: stages print in order; final line shows a `split`/`label` breakdown with non-zero counts in `train`, `val` and `test`. First run takes roughly 1–2 hours (FIRMS ~434 requests, Open-Meteo ~200 throttled calls); reruns are near-instant from cache.

- [ ] **Step 7: Write `README.md`**

````markdown
# Algeria Wildfire Risk — Dataset Pipeline

Builds a labelled fire/no-fire dataset for northern Algeria from **NASA FIRMS**
satellite detections and **Open-Meteo** historical weather, covering 14 fire
seasons (2012–2025). Replaces the 244-row UCI "Algerian Forest Fires" dataset
with ~40–60k rows built from live APIs.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # then paste your key from
                              # https://firms.modaps.eosdis.nasa.gov/api/map_key/
$env:PYTHONPATH="src"; python scripts/00_check_apis.py   # live smoke test
```

## Run

```bash
$env:PYTHONPATH="src"; python -m firerisk.build all
```

Stages are individually runnable: `firms`, `universe`, `samples`, `weather`,
`features`. Every network stage caches to disk, so reruns are near-instant.
Output: `data/processed/dataset.parquet`.

## Design decisions

| Decision | Choice | Why |
|---|---|---|
| Unit of analysis | one 0.1° cell on one day | matches the ERA5-Land weather grid; one fire emits dozens of pixels, so pixel-level rows would pseudo-replicate each fire ~12× and leak it across CV folds |
| Detection filter | `type == 0` only | measured over the 2023 season, static industrial sources (`type=2`, 6050) **outnumbered** real vegetation fires (4443) and recur at fixed coordinates year-round |
| Candidate cells | land that burned ≥2 days in ≥2 distinct years | every negative is land that *can* burn, so "is there fuel here?" cannot be the discriminator |
| Negatives | same cell, different day | each cell appears as both positive and negative, making cell identity statistically independent of the label |
| Buffers | ±3 days own-cell, same-day neighbours | fires burn for days and VIIRS misses days to cloud; the day *before* ignition is the driest day on record and labelling it 0 teaches the inverse of fire risk |
| Splits | by year, never random | a random split would scatter one multi-day fire across train and test |
| Features | weather + FWI, **no coordinates** | prevents geography re-entering through lat/lon |

## Known limitations

1. **"No detection" ≠ "no fire".** VIIRS cannot see through cloud. Cloudy days
   are systematically wetter *and* less observable, so part of any learned
   "rain → no fire" relationship reflects satellite blindness rather than fire
   physics. This cannot be fixed by sampling. It is the most important caveat
   on any result from this project.
2. Fires smaller than the 375 m detection threshold are invisible, biasing
   labels toward larger events.
3. The universe is previously-burned land, so the model does not generalise to
   never-burned forest — and, being tree-based, cannot extrapolate beyond its
   training range. A query for desert or urban terrain will return a confident
   and meaningless "extreme risk". The serving layer must reject out-of-universe
   points rather than score them.
4. The universe is built from all 14 years including the test years, so test
   metrics describe fire-prone land, not arbitrary terrain.
5. FWI day-length factors are calibrated for ~46°N; northern Algeria is ~36°N.
6. Weather is ~11 km reanalysis, not station data — slope, aspect and valley
   winds are unresolved.
7. Ignition cause is absent. Most Algerian wildfires are human-caused; this
   model predicts fire-conducive *conditions*, not human behaviour.

## Data sources

- NASA FIRMS — https://firms.modaps.eosdis.nasa.gov/api/ (VIIRS SNPP, archive from 2012-01-20)
- Open-Meteo Archive — https://archive-api.open-meteo.com/v1/archive (ERA5-Land)
````

- [ ] **Step 8: Stage the changes (do NOT commit)**

```bash
git add src/firerisk/build.py tests/test_build.py README.md
```

Report this proposed commit message to the user; they run the commit themselves:
`feat: pipeline orchestrator CLI and README`

---

## Deferred to later plans

- Modelling (RF/XGBoost, PR-AUC, SHAP, GroupKFold-by-cell diagnostic, the with/without-coordinates comparison)
- The **natural-panel evaluation set** at true base rate (spec §6) — belongs with modelling, since it exists to calibrate the decision threshold
- The serving-layer **domain guard** rejecting out-of-universe queries — belongs with the FastAPI plan. *Not currently written in the spec:* it came out of the "why only previously-burned places?" discussion, whose conclusion was that the universe restriction is correct for the dataset but leaves the trained model undefined outside burnable land, where tree models cannot extrapolate and will confidently return extreme risk for bare desert. The fix belongs in the API, not the dataset. Worth adding to the spec before the serving plan is written.
- FastAPI service, Flutter client
