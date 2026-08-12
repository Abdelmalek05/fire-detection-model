"""Resumable weather download. Run this in your own terminal; it survives
this session and can be interrupted freely.

Open-Meteo's free tier bills by locations x days, so the full job cannot
finish inside one hourly quota window. This script fetches what it can,
sleeps when the quota is exhausted, wakes up and continues - until done.

    python scripts/fetch_weather.py            # fetch until complete
    python scripts/fetch_weather.py --status   # just report progress

Safe to Ctrl+C at any point: every completed chunk is cached to disk, and
rerunning skips everything already fetched.
"""
import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from firerisk import firms, universe as universe_mod, weather as weather_mod
from firerisk.config import load_config

SLEEP_AFTER_QUOTA = 65 * 60  # a little over an hour, so the window has reset


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def ensure_universe(cfg, rebuild=False):
    """Universe must match the configured years: a cell that only burned in
    2013 is irrelevant to a 2021-2025 dataset and would waste a fetch."""
    path = Path(cfg.data_dir) / "interim" / "universe.parquet"
    if path.exists() and not rebuild:
        u = pd.read_parquet(path)
        log(f"universe: {len(u)} cells (cached)")
        return u
    log("building universe from FIRMS detections ...")
    det = firms.load_detections(cfg)
    qual = firms.qualifying(det, cfg)
    u = universe_mod.build_universe(
        qual, cfg.universe_min_fire_days, cfg.universe_min_years
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    u.to_parquet(path, index=False)
    log(f"universe: {len(u)} cells from {len(qual)} qualifying detections")
    return u


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", action="store_true", help="report progress and exit")
    ap.add_argument("--rebuild-universe", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    u = ensure_universe(cfg, rebuild=args.rebuild_universe)
    chunks = weather_mod.chunk_cells(u, cfg.weather_chunk_size)
    jobs = [(y, i) for y in cfg.years for i in range(len(chunks))]

    def remaining():
        return [
            (y, i) for (y, i) in jobs
            if not weather_mod._cache_path(cfg, y, i).exists()
        ]

    todo = remaining()
    log(f"years {cfg.year_start}-{cfg.year_end} | {len(chunks)} chunks/year "
        f"| {len(jobs)} total | {len(jobs) - len(todo)} done | {len(todo)} to go")

    if args.status:
        return 0
    if not todo:
        log("nothing to do - weather is complete")
        return 0

    started = time.time()
    completed_this_run = 0

    while True:
        todo = remaining()
        if not todo:
            log(f"WEATHER FETCH COMPLETE - {len(jobs)} chunks on disk")
            return 0
        try:
            for year, idx in todo:
                weather_mod.fetch_year_chunk(cfg, chunks[idx], year, idx)
                completed_this_run += 1
                done = len(jobs) - len(remaining())
                rate = completed_this_run / max(time.time() - started, 1) * 3600
                log(f"  {year} chunk {idx:02d} ok   {done}/{len(jobs)} "
                    f"({100 * done / len(jobs):.0f}%)  ~{rate:.1f} chunks/h")
        except weather_mod.RateLimitExceeded as exc:
            wake = datetime.now() + timedelta(seconds=SLEEP_AFTER_QUOTA)
            done = len(jobs) - len(remaining())
            log(f"quota exhausted ({exc})")
            log(f"  progress {done}/{len(jobs)}; sleeping until {wake:%H:%M} ...")
            try:
                time.sleep(SLEEP_AFTER_QUOTA)
            except KeyboardInterrupt:
                log("interrupted - progress is saved, just rerun to continue")
                return 130
        except KeyboardInterrupt:
            log("interrupted - progress is saved, just rerun to continue")
            return 130


if __name__ == "__main__":
    sys.exit(main())
