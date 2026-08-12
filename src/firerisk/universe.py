"""Burnable-land mask: cells with repeated fire activity across distinct years.

Requiring >=2 fire-days in >=2 distinct years does two things: it screens out
one-off false detections, and it mitigates the inclusion bias where a cell
would enter the universe solely because of the single fire we are trying to
predict.
"""
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
