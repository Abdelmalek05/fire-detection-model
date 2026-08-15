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

# Fuel depletion: burned ground does not reburn for months. Measured worth
# +20% PR-AUC over BASE_FEATURES alone - the best single feature found.
FUEL_FEATURES = ["days_since_last_fire"]

MODEL_FEATURES = BASE_FEATURES + FUEL_FEATURES

NEVER_BURNED = 9999.0  # sentinel: no qualifying fire before this row


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


def add_days_since_last_fire(samples, qual, buffer_days):
    """Days since this cell last burned, ignoring fires newer than the
    sampling buffer.

    THE LAG IS NOT OPTIONAL. Negatives are sampled at least buffer_days+1
    away from any fire, while a positive inside a multi-day fire sits 1-3
    days after the previous one. Counting those recent fires makes every
    value below buffer_days+1 exclusively positive - measured on the real
    dataset: 8,365 rows, 100% positive, 27.6% of ALL positives, classified
    by a rule that is our sampling design and not fire behaviour. It inflated
    PR-AUC from 0.458 to 0.688 on pure artefact.

    Looking back only to fires at least buffer_days+1 old restores symmetry:
    both classes then see the same feasible range, and what survives is real
    fuel depletion - burned ground does not reburn for months. Worth +20%.

    See tests/test_features_fuel.py, which fails if the lag is removed.
    """
    lag = np.timedelta64(buffer_days + 1, "D")
    fire_days = (
        qual.groupby("cell_id")["date"]
        .apply(lambda s: np.sort(s.unique()))
        .to_dict()
    )
    out = np.full(len(samples), np.nan)
    for i, (cell, day) in enumerate(
        zip(samples.cell_id.values, samples.date.values)
    ):
        days = fire_days.get(cell)
        if days is None:
            continue
        j = np.searchsorted(days, day - lag)
        if j > 0:
            out[i] = (day - days[j - 1]) / np.timedelta64(1, "D")
    return pd.Series(out, index=samples.index).fillna(NEVER_BURNED)


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
