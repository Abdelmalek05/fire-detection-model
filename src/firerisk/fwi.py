"""Canadian Forest Fire Weather Index System (Van Wagner 1987).

Driven by daily aggregates (max temp, min RH, max wind, 24h precipitation)
rather than the canonical noon-local observations - the standard published
approximation, and effectively what the UCI Algerian Forest Fires dataset
used. Values therefore run slightly hot versus an official calculation.

Day-length factors are the published tables calibrated for ~46 N; northern
Algeria is ~36 N, so these are an accepted approximation (documented, not
hidden). Correcting them is a v2 refinement.
"""
import math

import numpy as np

DAY_LENGTH = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
DAY_LENGTH_DC = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]

FFMC_START, DMC_START, DC_START = 85.0, 6.0, 15.0


def ffmc(temp, rh, wind, rain, ffmc_prev):
    """Fine Fuel Moisture Code - responds within hours."""
    rh = min(rh, 100.0)
    mo = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)
    if rain > 0.5:
        rf = rain - 0.5
        mo += 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
        if mo > 150.0:
            mo += 0.0015 * (mo - 150.0) ** 2 * math.sqrt(rf)
        mo = min(mo, 250.0)

    ed = (
        0.942 * rh ** 0.679
        + 11.0 * math.exp((rh - 100.0) / 10.0)
        + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh))
    )
    if mo > ed:
        ko = 0.424 * (1.0 - (rh / 100.0) ** 1.7) + 0.0694 * math.sqrt(wind) * (
            1.0 - (rh / 100.0) ** 8
        )
        kd = ko * 0.581 * math.exp(0.0365 * temp)
        m = ed + (mo - ed) * 10.0 ** (-kd)
    else:
        ew = (
            0.618 * rh ** 0.753
            + 10.0 * math.exp((rh - 100.0) / 10.0)
            + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh))
        )
        if mo < ew:
            kl = 0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7) + 0.0694 * math.sqrt(
                wind
            ) * (1.0 - ((100.0 - rh) / 100.0) ** 8)
            kw = kl * 0.581 * math.exp(0.0365 * temp)
            m = ew - (ew - mo) * 10.0 ** (-kw)
        else:
            m = mo
    value = 59.5 * (250.0 - m) / (147.2 + m)
    return float(min(max(value, 0.0), 101.0))


def dmc(temp, rh, rain, dmc_prev, month):
    """Duff Moisture Code - responds over weeks."""
    temp = max(temp, -1.1)
    le = DAY_LENGTH[month - 1]
    rk = 1.894 * (temp + 1.1) * (100.0 - rh) * le * 1e-4
    if rain > 1.5:
        rw = 0.92 * rain - 1.27
        wmi = 20.0 + 280.0 / math.exp(0.023 * dmc_prev)
        if dmc_prev <= 33.0:
            b = 100.0 / (0.5 + 0.3 * dmc_prev)
        elif dmc_prev <= 65.0:
            b = 14.0 - 1.3 * math.log(dmc_prev)
        else:
            b = 6.2 * math.log(dmc_prev) - 17.2
        wmr = wmi + 1000.0 * rw / (48.77 + b * rw)
        pr = 43.43 * (5.6348 - math.log(max(wmr - 20.0, 1e-6)))
        base = max(pr, 0.0)
    else:
        base = dmc_prev
    return float(max(base + rk, 0.0))


def dc(temp, rain, dc_prev, month):
    """Drought Code - deep moisture, ~50-day memory."""
    temp = max(temp, -2.8)
    lf = DAY_LENGTH_DC[month - 1]
    pe = max((0.36 * (temp + 2.8) + lf) / 2.0, 0.0)
    if rain > 2.8:
        rw = 0.83 * rain - 1.27
        smi = 800.0 * math.exp(-dc_prev / 400.0)
        base = max(dc_prev - 400.0 * math.log(1.0 + 3.937 * rw / smi), 0.0)
    else:
        base = dc_prev
    return float(max(base + pe, 0.0))


def isi(wind, ffmc_val):
    """Initial Spread Index - wind plus fine fuel moisture."""
    f_wind = math.exp(0.05039 * wind)
    m = 147.2 * (101.0 - ffmc_val) / (59.5 + ffmc_val)
    f_f = 91.9 * math.exp(-0.1386 * m) * (1.0 + m ** 5.31 / 4.93e7)
    return float(0.208 * f_wind * f_f)


def bui(dmc_val, dc_val):
    """Buildup Index - total available fuel."""
    denom = dmc_val + 0.4 * dc_val
    if denom <= 0:
        return 0.0
    if dmc_val <= 0.4 * dc_val:
        value = 0.8 * dmc_val * dc_val / denom
    else:
        value = dmc_val - (1.0 - 0.8 * dc_val / denom) * (
            0.92 + (0.0114 * dmc_val) ** 1.7
        )
    return float(max(value, 0.0))


def fwi(isi_val, bui_val):
    """Fire Weather Index - the headline number."""
    if bui_val <= 80.0:
        f_d = 0.626 * bui_val ** 0.809 + 2.0
    else:
        f_d = 1000.0 / (25.0 + 108.64 * math.exp(-0.023 * bui_val))
    b = 0.1 * isi_val * f_d
    if b > 1.0:
        return float(math.exp(2.72 * (0.434 * math.log(b)) ** 0.647))
    return float(b)


def compute_series(df):
    """Chronological pass per (cell_id, year). Codes restart at each year.

    The codes are recursive - each day's value depends on the previous day's -
    so this must run in date order within a cell, and must reset at a cell or
    year boundary rather than carrying one cell's drought into the next.

    Returns a new frame; never mutates the caller's DataFrame.
    """
    df = df.sort_values(["cell_id", "date"]).reset_index(drop=True).copy()
    out = {k: np.empty(len(df)) for k in ("ffmc", "dmc", "dc", "isi", "bui", "fwi")}

    key = list(zip(df.cell_id, df.date.dt.year))
    prev_key = None
    f_prev, d_prev, c_prev = FFMC_START, DMC_START, DC_START

    for i, row in enumerate(df.itertuples(index=False)):
        if key[i] != prev_key:
            f_prev, d_prev, c_prev = FFMC_START, DMC_START, DC_START
            prev_key = key[i]
        month = row.date.month
        f = ffmc(row.temp_max, row.rh_min, row.wind_max, row.precip_sum, f_prev)
        d = dmc(row.temp_max, row.rh_min, row.precip_sum, d_prev, month)
        c = dc(row.temp_max, row.precip_sum, c_prev, month)
        i_val = isi(row.wind_max, f)
        b_val = bui(d, c)
        out["ffmc"][i], out["dmc"][i], out["dc"][i] = f, d, c
        out["isi"][i], out["bui"][i] = i_val, b_val
        out["fwi"][i] = fwi(i_val, b_val)
        f_prev, d_prev, c_prev = f, d, c

    for k, v in out.items():
        df[k] = v
    return df
