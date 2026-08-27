"""Vegetation state from MOD13Q1: the level, its seasonal normal, and both
ways of comparing them.

Measured, not assumed. The design started from "feed anomalies, never raw
NDVI" - reasoning that matched sampling made cell identity independent of the
label, so the level could only encode which cell you were in. That was tested
and is WRONG:

    + ndvi_anomaly alone        -0.0009 PR-AUC    3/5 folds   (nothing)
    + ndvi (raw level)          +0.0128 PR-AUC   12/14 folds  p=0.0007

The level wins because it is a fuel-type signature - scrub at 0.25 and oak
forest at 0.65 need different amounts of drying before they burn - and fuel
type INTERACTS with weather even though it has no main effect. Subtracting the
normal deletes exactly that. The anomaly is worthless alone yet adds a little
alongside the level, which fits: "12% below normal" means different things for
scrub and for forest.

So this module exports all of it and lets features.py choose. Two decisions
remain load-bearing, and each fails by producing plausible numbers rather than
an error:

1. The climatological normal comes from 2000-2011, disjoint from the
   2012-2025 modelling years. A normal computed over all years contains the
   very fires being predicted.

2. A composite may only be used once its 16-day window has CLOSED. A fire
   destroys vegetation, so a window spanning the label date carries the burn
   scar - the label wearing a disguise.
"""
from pathlib import Path

import pandas as pd

SCALE = 0.0001           # MOD13Q1 NDVI is int16
WINDOW_DAYS = 16         # composite covers start .. start + 15
MIN_PIXELS = 200         # of ~1,660; below this the cell is mostly cloud
BASELINE = (2000, 2011)  # climatology years - MUST NOT overlap the model years
MODEL_YEARS = (2012, 2025)
CHANGE_LAG = 2           # composites; 2 x 16 days = 32


def consolidate(raw_dir) -> pd.DataFrame:
    """Per-year GEE CSVs -> one tidy frame of (cell, composite) NDVI."""
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("ndvi_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"no ndvi_*.csv under {raw_dir}. Run scripts/fetch_ndvi.py."
        )

    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d.rename(columns={"mean": "ndvi_raw", "count": "n_pixels"})
    d = d.dropna(subset=["ndvi_raw"])
    d = d[d["n_pixels"] >= MIN_PIXELS]

    d["composite_start"] = pd.to_datetime(d["start"])
    d["composite_end"] = d["composite_start"] + pd.Timedelta(days=WINDOW_DAYS - 1)
    d["composite_doy"] = d["composite_start"].dt.dayofyear
    d["ndvi"] = d["ndvi_raw"] * SCALE

    cols = ["cell_id", "composite_start", "composite_end", "composite_doy",
            "ndvi", "n_pixels"]
    return (d[cols]
            .sort_values(["cell_id", "composite_start"])
            .reset_index(drop=True))


def climatology(ndvi, baseline=BASELINE) -> pd.DataFrame:
    """Each cell's normal NDVI for each 16-day slot, from baseline years only.

    A normal computed over the modelling years contains the fires being
    predicted: a cell that burned in 2021 pulls its own 2021 normal down, so
    the anomaly that should read "this place is scorched" reads closer to
    zero instead.
    """
    lo, hi = baseline
    if hi >= MODEL_YEARS[0]:
        raise ValueError(
            f"baseline {baseline} overlaps the modelling years {MODEL_YEARS}. "
            "The normal would contain the fires being predicted."
        )
    years = ndvi["composite_start"].dt.year
    base = ndvi[(years >= lo) & (years <= hi)]
    return (base.groupby(["cell_id", "composite_doy"], as_index=False)["ndvi"]
                .mean()
                .rename(columns={"ndvi": "ndvi_normal"}))


def with_anomalies(ndvi, normals) -> pd.DataFrame:
    """Attach the departure from normal and the 32-day trend."""
    out = ndvi.merge(normals, on=["cell_id", "composite_doy"], how="left")
    out = out.sort_values(["cell_id", "composite_start"]).reset_index(drop=True)
    out["ndvi_anomaly"] = out["ndvi"] - out["ndvi_normal"]
    out["ndvi_change_32d"] = out.groupby("cell_id")["ndvi"].diff(CHANGE_LAG)
    return out


# Everything the join carries. features.py decides which of these the model
# reads - the level earns its place, the anomaly alone does not.
NDVI_JOIN_COLS = ["ndvi", "ndvi_normal", "ndvi_anomaly", "ndvi_change_32d",
                  "ndvi_stale_days"]


def attach(panel, ndvi) -> pd.DataFrame:
    """Join the most recent composite whose window CLOSED before each date.

    merge_asof with allow_exact_matches=False on composite_end is the whole
    guard: a composite whose window ends on or after the label date saw the
    fire. It would look like a spectacular feature and be worthless in
    production.
    """
    left = (panel.assign(_row=range(len(panel)))
                 .sort_values("date")
                 .reset_index(drop=True))
    right = (ndvi[["cell_id", "composite_end", "ndvi", "ndvi_normal",
                   "ndvi_anomaly", "ndvi_change_32d"]]
             .sort_values("composite_end")
             .reset_index(drop=True))

    merged = pd.merge_asof(
        left, right,
        left_on="date", right_on="composite_end",
        by="cell_id",
        direction="backward",
        allow_exact_matches=False,   # the window must have CLOSED, not be closing
    )
    merged["ndvi_stale_days"] = (merged["date"] - merged["composite_end"]).dt.days
    return (merged.sort_values("_row")
                  .drop(columns=["_row", "composite_end"])
                  .reset_index(drop=True))
