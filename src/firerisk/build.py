"""CLI orchestrator for the dataset pipeline.

    python -m firerisk.build all         # every stage
    python -m firerisk.build features    # just re-assemble from cache

Every stage is cached, so re-running is cheap and safe. The weather stage is
the slow one and is usually run separately via scripts/fetch_weather.py, which
survives quota blocks; `build weather` here is the same fetch without the
sleep-and-resume loop.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from . import features, firms, panel
from . import universe as universe_mod
from . import weather as weather_mod
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

    universe_path = interim / "universe.parquet"
    if run_all or args.stage == "universe" or not universe_path.exists():
        print("[2/5] universe ...")
        uni = universe_mod.build_universe(
            qual, cfg.universe_min_fire_days, cfg.universe_min_years
        )
        uni.to_parquet(universe_path, index=False)
        print(f"      universe cells: {len(uni)}")

    uni = pd.read_parquet(universe_path)

    samples_path = interim / "samples.parquet"
    if run_all or args.stage == "samples" or not samples_path.exists():
        print("[3/5] positives + negatives ...")
        samples = panel.build_samples(det_all, qual, uni, cfg)
        samples.to_parquet(samples_path, index=False)
        n_pos = int((samples.label == 1).sum())
        print(f"      samples: {len(samples)} (positives {n_pos}, "
              f"negatives {len(samples) - n_pos})")

    if run_all or args.stage == "weather":
        print("[4/5] weather ...")
        weather_mod.load_weather(cfg, uni)
        print("      weather cached")

    if run_all or args.stage == "features":
        print("[5/5] features ...")
        samples = pd.read_parquet(samples_path)
        wx = _load_cached_weather(cfg, uni)
        wx = features.add_rolling(wx)
        dataset = features.assemble(samples, wx, uni, cfg, qual)
        if dataset["days_since_last_fire"].nunique() < 10:
            raise SystemExit(
                "days_since_last_fire is near-constant. FIRMS detections are "
                "missing or empty - refusing to write a dataset whose best "
                "feature carries no signal."
            )
        out = processed / "dataset.parquet"
        _write_dataset(dataset, out, cfg.temporal_buffer_days)
        print(f"      dataset rows: {len(dataset)}  ->  {out}")
        print(dataset.groupby(["split", "label"]).size().to_string())

    return 0


def _write_dataset(df, path, buffer_days):
    """Write the parquet, recording the sampling buffer in its metadata.

    days_since_last_fire is derived from that buffer, so storing it makes a
    mismatch detectable rather than silent.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[b"temporal_buffer_days"] = str(buffer_days).encode()
    pq.write_table(table.replace_schema_metadata(meta), path)


def _load_cached_weather(cfg, universe):
    """Read only the weather chunks already on disk.

    The fetch is long and quota-limited, so the feature stage must work on a
    partial download rather than demanding all of it - that is what lets a
    complete dataset be built for the years that have finished.
    """
    root = Path(cfg.data_dir) / "raw" / "weather"
    paths = sorted(root.glob("*/chunk_*.parquet")) if root.exists() else []
    if not paths:
        raise SystemExit(
            "No weather cached yet. Run: python scripts/fetch_weather.py"
        )
    frames = [pd.read_parquet(p) for p in paths]
    wx = pd.concat(frames, ignore_index=True)
    wx = wx[wx.cell_id.isin(set(universe.cell_id))]
    years = sorted(wx.date.dt.year.unique())
    print(f"      weather: {len(paths)} chunks, years {years}")
    return wx.drop_duplicates(subset=["cell_id", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    sys.exit(main())
