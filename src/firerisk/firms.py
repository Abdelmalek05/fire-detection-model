"""NASA FIRMS ingest: chunked, cached, filtered, snapped to the analysis grid.

Two filters with deliberately different widths:
  * POSITIVES require type==0 AND confidence in the configured set (n, h).
  * EXCLUSION/buffer tests use type==0 at ALL confidences, including 'l'.
A low-confidence detection is not enough to assert a fire, but is ample reason
not to assert its absence - so load_detections() keeps every confidence and
qualifying() narrows to the trustworthy ones.
"""
import datetime as dt
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
