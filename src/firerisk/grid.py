"""0.1-degree analysis grid.

Uses explicit half-up rounding (floor(x/res + 0.5), epsilon-corrected) rather
than Python's round(), which is banker's rounding and would place .5
boundaries inconsistently.
"""
import numpy as np
import pandas as pd

RES = 0.1

# Guards against binary floating-point representation error (e.g. 4.05 / 0.1
# == 40.49999999999999 in IEEE 754), which would otherwise round .5 boundary
# values down instead of up. Far smaller than any real coordinate spacing.
_EPS = 1e-9


def _index(value, res):
    """Half-up grid index, epsilon-corrected. THE single source of the
    rounding rule - both the scalar and vectorised paths must call this."""
    return np.floor(value / res + 0.5 + _EPS)


def _idx(value: float, res: float) -> int:
    return int(_index(value, res))


def to_cell_id(lat: float, lon: float, res: float = RES) -> str:
    return f"{_idx(lat, res)}_{_idx(lon, res)}"


def to_cell_id_vec(lat: pd.Series, lon: pd.Series, res: float = RES) -> pd.Series:
    ilat = _index(lat, res).astype(int)
    ilon = _index(lon, res).astype(int)
    return ilat.astype(str) + "_" + ilon.astype(str)


def cell_center(cell_id: str, res: float = RES) -> tuple:
    ilat, ilon = cell_id.split("_")
    return int(ilat) * res, int(ilon) * res


def neighbours(cell_id: str) -> list:
    ilat, ilon = (int(v) for v in cell_id.split("_"))
    return [
        f"{ilat + di}_{ilon + dj}"
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
        if not (di == 0 and dj == 0)
    ]
