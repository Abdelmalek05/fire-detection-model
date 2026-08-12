"""Positives and same-cell matched negative sampling.

Negatives are drawn from the SAME cell as the positive, so cell identity is
statistically independent of the label - the model cannot discriminate by
geography, because every cell appears on both sides. Three guards apply:

  * own-cell fire days block a +/- buffer_days window (absolute calendar dates)
  * any of the 8 neighbours having a detection blocks that exact day
  * candidate days must lie inside the fire season

Exclusion uses detections at ALL confidences, while positives use only the
qualifying ones: a low-confidence detection is not enough to assert a fire,
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
    # Cap per cell per season, keeping the strongest events by FRP so that
    # hotspot cells (up to 50 fire-days in a season) cannot dominate.
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

        # Guard 1: own-cell fire days, +/- buffer_days.
        # Absolute calendar-date difference, NOT day-of-year: a fire on
        # 2020-07-10 must not block 2021-07-10.
        for fire_day in exclusion.get(cell_id, set()):
            delta = np.abs((day_values - fire_day).days)
            blocked |= delta <= buffer_days

        # Guard 2: any neighbour detection blocks that exact day. A fire near a
        # cell boundary spills pixels into adjacent cells, and at 11 km those
        # neighbours share one weather pixel.
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
