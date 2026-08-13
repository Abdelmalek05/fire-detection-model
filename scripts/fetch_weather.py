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
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from firerisk import firms, universe as universe_mod, weather as weather_mod
from firerisk.config import load_config

SLEEP_AFTER_HOURLY = 65 * 60  # a little over an hour, so the window has reset

# Consecutive failures before a chunk is abandoned for this run. Bounded so
# that deferring a broken chunk cannot become an infinite spin; the chunk is
# still on the todo list for the next run, since nothing was written.
MAX_CHUNK_FAILURES = 3

EXIT_INCOMPLETE = 75  # finished what it could; some chunks were abandoned


def sleep_seconds_for(reason, consecutive_blocks=0):
    """How long to wait after a quota block.

    Quotas are enforced per client IP. We do not actually know whether a
    "Daily API request limit exceeded" block clears on a rolling window or
    only at UTC midnight - the one apparent recovery we saw coincided with a
    network change, so it tells us nothing about waiting it out.

    So this adapts instead of assuming: try an hour first (cheap if the block
    was hourly, or rolling), and escalate to waiting for the UTC-midnight
    reset only after several consecutive blocks land with no chunk in between
    - which is the signal that hourly waits genuinely are not clearing it.
    """
    if consecutive_blocks >= 3 and "aily" in reason:
        now = datetime.now(timezone.utc)
        reset = (now + timedelta(days=1)).replace(
            hour=0, minute=5, second=0, microsecond=0
        )
        return max((reset - now).total_seconds(), SLEEP_AFTER_HOURLY)
    return SLEEP_AFTER_HOURLY


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", action="store_true", help="report progress and exit")
    ap.add_argument("--rebuild-universe", action="store_true")
    return ap


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


def fetch_until_done(cfg, chunks, jobs, session=None, sleeper=time.sleep):
    """Fetch every outstanding chunk, surviving everything survivable.

    Three failure modes, three different answers:

    * `RateLimitExceeded` - our address is spent for a window longer than a
      retry. Sleep, then retry the whole todo list.
    * `RuntimeError` - this chunk's retries ran out. Defer it and move to the
      next; it is retried on the following pass. After MAX_CHUNK_FAILURES
      consecutive failures it is abandoned and reported, which is also what
      stops deferral from becoming an infinite spin.
    * `KeyboardInterrupt` - stop. Every finished chunk is already on disk.

    Returns 0 when every chunk landed, EXIT_INCOMPLETE when some were
    abandoned, 130 on interrupt. It never raises.
    """
    started = time.time()
    completed_this_run = 0
    consecutive_blocks = 0
    failures: dict[tuple, int] = {}

    def remaining():
        return [(y, i) for (y, i) in jobs
                if not weather_mod._cache_path(cfg, y, i).exists()]

    def workable(todo):
        return [j for j in todo if failures.get(j, 0) < MAX_CHUNK_FAILURES]

    while True:
        todo = remaining()
        if not todo:
            log(f"WEATHER FETCH COMPLETE - {len(jobs)} chunks on disk")
            return 0

        live = workable(todo)
        if not live:
            log(f"STOPPING - {len(todo)} chunk(s) failed "
                f"{MAX_CHUNK_FAILURES}x each and were abandoned:")
            for year, idx in todo:
                log(f"    {year} chunk {idx:02d}")
            log("  everything else is on disk; rerun to retry these.")
            return EXIT_INCOMPLETE

        try:
            for year, idx in live:
                try:
                    weather_mod.fetch_year_chunk(cfg, chunks[idx], year, idx,
                                                 session=session)
                except weather_mod.RateLimitExceeded:
                    raise                       # quota: handled below
                except RuntimeError as exc:
                    failures[(year, idx)] = failures.get((year, idx), 0) + 1
                    log(f"  {year} chunk {idx:02d} failed "
                        f"({failures[(year, idx)]}/{MAX_CHUNK_FAILURES}): {exc}")
                    continue

                failures.pop((year, idx), None)  # consecutive, not lifetime
                completed_this_run += 1
                consecutive_blocks = 0           # progress resets escalation
                done = len(jobs) - len(remaining())
                rate = completed_this_run / max(time.time() - started, 1) * 3600
                log(f"  {year} chunk {idx:02d} ok   {done}/{len(jobs)} "
                    f"({100 * done / len(jobs):.0f}%)  ~{rate:.1f} chunks/h")

        except weather_mod.RateLimitExceeded as exc:
            consecutive_blocks += 1
            secs = sleep_seconds_for(str(exc), consecutive_blocks)
            wake = datetime.now() + timedelta(seconds=secs)
            done = len(jobs) - len(remaining())
            log(f"quota exhausted ({exc})")
            log(f"  progress {done}/{len(jobs)}; sleeping {secs / 3600:.1f}h "
                f"until {wake:%a %H:%M} ...")
            try:
                sleeper(secs)
            except KeyboardInterrupt:
                log("interrupted - progress is saved, just rerun to continue")
                return 130
        except KeyboardInterrupt:
            log("interrupted - progress is saved, just rerun to continue")
            return 130


def main():
    args = build_parser().parse_args()

    cfg = load_config()
    u = ensure_universe(cfg, rebuild=args.rebuild_universe)
    chunks = weather_mod.chunk_cells(u, cfg.weather_chunk_size)
    jobs = [(y, i) for y in cfg.years for i in range(len(chunks))]
    todo = [(y, i) for (y, i) in jobs
            if not weather_mod._cache_path(cfg, y, i).exists()]

    log(f"years {cfg.year_start}-{cfg.year_end} | {len(chunks)} chunks/year "
        f"| {len(jobs)} total | {len(jobs) - len(todo)} done | {len(todo)} to go")

    if args.status:
        return 0
    if not todo:
        log("nothing to do - weather is complete")
        return 0

    return fetch_until_done(cfg, chunks, jobs)


if __name__ == "__main__":
    sys.exit(main())
