"""Open-Meteo archive client: chunked, throttled, cached to parquet.

Measured API limits (verified live, not assumed): ~200 locations per call
(500 returns HTTP 414 Request-URI Too Large), and a weighted minutely rate
limit that returns HTTP 429. Chunk at 150 and sleep between calls.

Fetches Jan 1 -> Oct 31 per cell, not just the fire season: the FWI moisture
codes and the rolling dryness windows need continuous antecedent weather, and
DC in particular has a ~50-day memory.
"""
import os
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://archive-api.open-meteo.com/v1/archive"

MAX_RETRIES = 5

# Separate connect and read limits, and the distinction is not cosmetic.
# requests applies a bare `timeout=300` to BOTH, so a host that accepts the TCP
# handshake and then goes silent holds a chunk for five full minutes. Reading
# legitimately needs minutes, because one chunk is ~3 MB; connecting either
# happens in seconds or is never going to happen.
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 300
REQUEST_TIMEOUT = (CONNECT_TIMEOUT, READ_TIMEOUT)

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "relative_humidity_2m_min", "relative_humidity_2m_mean",
    "wind_speed_10m_max", "wind_speed_10m_mean", "precipitation_sum",
    "vapour_pressure_deficit_max",
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
    "vapour_pressure_deficit_max": "vpd_max",
}


def chunk_cells(universe, size):
    return [
        universe.iloc[i:i + size].reset_index(drop=True)
        for i in range(0, len(universe), size)
    ]


def parse_response(payload, cells):
    """Map Open-Meteo results onto cells positionally.

    Open-Meteo returns a JSON array for multi-location requests and a bare
    object for a single location, in the same order the coordinates were sent.
    """
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
        Path(cfg.data_dir) / "raw" / "weather" / str(year)
        / f"chunk_{chunk_idx:03d}.parquet"
    )


class RateLimitExceeded(RuntimeError):
    """Open-Meteo quota exhausted for a window longer than we should sleep on.

    Open-Meteo enforces minutely, hourly and daily quotas, weighted by
    locations x days - so one 150-cell, 304-day request costs far more than
    one "call". A minutely block clears in a minute and is worth waiting out;
    an hourly or daily block is not, and blindly sleeping on it burns the
    retry budget and reports a useless error. Raise instead, so the caller
    can stop, report honestly, and resume later from cache.
    """


def _write_atomic(df, path):
    """Write via a temp name and rename into place.

    Resumption trusts `path.exists()`, so a chunk file that exists is treated
    as finished forever. A crash partway through a direct write would leave a
    truncated parquet that every later run silently accepts. os.replace is
    atomic on the same filesystem, so the real name only ever appears once the
    bytes are all there. The `.parquet.tmp` suffix deliberately does not match
    the `chunk_*.parquet` glob in build._load_cached_weather.
    """
    tmp = path.with_suffix(".parquet.tmp")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def fetch_year_chunk(cfg, cells, year, chunk_idx, session=None):
    """Fetch one (year, chunk) of weather, or return it from cache.

    Wait out a minutely block, which clears in a minute and is worth sitting
    through. Raise on an hourly or daily one, so the caller can sleep on its
    own schedule and resume from cache rather than blocking here.
    """
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
    minutely_waits = 0
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            r = get(BASE, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                reason = ""
                try:
                    reason = r.json().get("reason", "")
                except ValueError:
                    reason = r.text[:200]

                if "inutely" in reason and minutely_waits < 5:
                    minutely_waits += 1
                    time.sleep(60)
                    continue
                raise RateLimitExceeded(
                    f"year={year} chunk={chunk_idx}: {reason or 'HTTP 429'}"
                )
            r.raise_for_status()
            df = parse_response(r.json(), cells)
            _write_atomic(df, path)
            time.sleep(cfg.weather_sleep_seconds)
            return df
        except RateLimitExceeded:
            raise
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(10 * (attempt + 1))
            attempt += 1
    raise RuntimeError(
        f"Open-Meteo fetch failed year={year} chunk={chunk_idx}: "
        f"{last if last is not None else 'retries exhausted with no exception'}"
    )


def load_weather(cfg, universe, session=None):
    chunks = chunk_cells(universe, cfg.weather_chunk_size)
    frames = []
    for year in cfg.years:
        for i, cells in enumerate(chunks):
            frames.append(fetch_year_chunk(cfg, cells, year, i, session=session))
    return pd.concat(frames, ignore_index=True)
