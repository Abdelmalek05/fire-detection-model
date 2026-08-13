"""Rolling dryness features and final dataset assembly.

Coordinates are deliberately absent from BASE_FEATURES. The same-cell negative
sampling exists to make cell identity carry no label information; feeding lat
and lon back in as features would hand that shortcut straight back. They are
kept as columns for reference and mapping, not for the model. A variant that
adds them is trained separately, as an explicit comparison.
"""
import numpy as np
import pandas as pd

from .fwi import compute_series

BASE_FEATURES = [
    # same-day conditions
    "temp_max", "temp_min", "temp_mean", "rh_min", "rh_mean",
    "wind_max", "wind_mean", "precip_sum", "vpd_max",
    # antecedent dryness - what a single day cannot tell you
    "precip_7d", "precip_30d", "days_since_rain_1mm",
    "temp_max_7d_mean", "rh_min_7d_mean",
    # Canadian FWI system
    "ffmc", "dmc", "dc", "isi", "bui", "fwi",
]


def _days_since_rain(precip):
    """Days since the last wetting rain (>=1mm). NaN until the first one."""
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
    """Backward-looking windows only - a window must never see the future."""
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
