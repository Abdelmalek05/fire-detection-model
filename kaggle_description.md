121,869 labelled `(0.1° cell, day)` rows for northern Algeria, built from **NASA FIRMS**
satellite fire detections, **Open-Meteo** ERA5-Land reanalysis weather, and **MODIS**
vegetation indices across 14 fire seasons (2012–2025).

This is not a scraped CSV — it is a sampled panel with a deliberate design, and the
design is the point. Please read the four warnings below before modelling.

---

## ⚠️ Read this before you model

**1. Three columns leak the label perfectly. Drop them.**

| Column | Why it leaks |
|---|---|
| `sample_kind` | literally `positive` / `matched_negative` |
| `n_detections` | exactly `0` for every negative, `≥1` for every positive |
| `max_frp` | `NaN` for all 91,518 negatives |

Any one of them alone gives PR-AUC 1.0. They exist for analysis (how big was the fire?)
and only have values *because* a fire happened. Use the **26 model features** listed
below. A realistic honest score is **PR-AUC ≈ 0.46–0.58**.

**⚠️ Do not build the feature list by exclusion.** Two columns are traps in opposite
directions: `doy` looks like bookkeeping but **is** a model feature, and `ndvi_anomaly`
looks like a feature but is **not** trained on (see warning 4). Getting both wrong still
yields 26 columns, so the count looks right while the set is not. List them explicitly.

**2. The 24.9% positive rate is not the real fire rate.**

It is a sampling artefact — every positive was matched with 3 negatives by construction.
The real rate is roughly **1 fire-day per 100 eligible cell-days in season**. Any absolute
probability a model outputs is on the sampled scale and must be prior-corrected before it
means anything.

**3. `label = 0` means "no satellite detection", not "no fire".**

VIIRS cannot see through cloud, and cloudy days are systematically both wetter *and* less
observable. Part of any learned "rain → no fire" relationship is satellite blindness
rather than fire physics. No sampling scheme fixes this. It is the most important caveat
on any result from this data.

**4. For vegetation, use the raw level — not the anomaly.**

This is counter-intuitive and was measured the hard way. Measured on this data:

| Feature | Δ PR-AUC |
|---|---|
| `ndvi_anomaly` alone | **−0.0009** (nothing) |
| `ndvi` raw level | **+0.0128** (12/14 folds, p = 0.0007) |

`ndvi_normal` is a **fuel-type signature** — scrub sits near 0.25, oak forest near 0.65,
and they need very different amounts of drying before they burn. Fuel type has no *main*
effect here (matched sampling removed that) but it **interacts** with weather, and
subtracting the norm deletes exactly that interaction.

The generalisable lesson: *"no main effect"* does not mean *"useless"*. It means look for
the interactions. `ndvi_anomaly` is shipped so you can reproduce the comparison; it is
not in the trained feature set.

---

## Why the negatives are the interesting part

A naive wildfire dataset samples negatives from random places and times. A model then
scores 0.95 by learning *"is this the Sahara in January?"* — which is useless.

Here, every negative is drawn from the **same cell as a positive, on a different day**,
in the same season, at least 4 days from any detection, with same-day neighbouring cells
excluded too. Cell identity is therefore statistically independent of the label
(measured: per-cell positive rate σ = 0.0074).

Two consequences worth knowing:

- The geographic shortcut is not discouraged — it is **arithmetically unavailable**.
- Any purely static per-cell feature (elevation, land cover, slope) has **no main
  effect** by construction. Only time-varying information can help.

Candidate cells are land that burned on ≥2 days in ≥2 distinct years, so "is there fuel
here?" cannot be the discriminator either.

---

## Files

| File | Rows | Notes |
|---|---|---|
| `dataset.csv` | 121,869 | the full panel |
| `dataset.parquet` | 121,869 | same data, typed and 3× smaller |

Grain is one `(cell_id, date)` pair — unique, zero duplicates.

---

## Columns (37)

**Keys (6)** — `cell_id`, `lat`, `lon`, `date`, `year`, `doy`
`lat`/`lon` are for mapping. Feeding them to a model hands back the geographic shortcut
the sampling design removed. **`doy` is the exception — it IS a model feature** (+0.006
PR-AUC, rank 3 by gain), because fire probability swings 4.3× across the season.

**Label and provenance (4)** — `label`, `sample_kind`, `n_detections`, `max_frp`
The last three are the leaks described above.

**Same-day weather (9)** — `temp_max`, `temp_min`, `temp_mean`, `rh_min`, `rh_mean`,
`wind_max`, `wind_mean`, `precip_sum`, `vpd_max`

**Antecedent dryness (5)** — `precip_7d`, `precip_30d`, `days_since_rain_1mm`,
`temp_max_7d_mean`, `rh_min_7d_mean`
All backward-looking. Computed from Jan 1, so a June 1 row already carries five months of
history.

**Canadian FWI system (6)** — `ffmc`, `dmc`, `dc`, `isi`, `bui`, `fwi`
⚠️ These run **well above** canonical FWI ranges (official DMC rarely exceeds ~150).
Three reasons: daily aggregates instead of noon-local observations, day-length tables
calibrated for ~46°N against Algeria's ~36°N, and a Jan 1 reset that lets the Drought Code
accumulate through a rainless summer. They are internally consistent and usable as
features — **do not compare them to published FWI thresholds.**

**Fuel (1)** — `days_since_last_fire`, range 5–9999 (`9999` = never burned in the record)
The single most important feature: **26% of model gain**, worth +20% PR-AUC over weather
alone. Burned ground does not reburn for months, and nothing in the weather knows that.
⚠️ It carries a mandatory **4-day lag**, which is why the minimum is 5 and not 0. Without
it the feature encodes the sampling design rather than fire behaviour — negatives sit at
least 4 days from any fire, so every value below 4 would be 100% positive (8,365 rows,
27.6% of all positives) and PR-AUC inflates to 0.688 on pure artefact. **If you recompute
this from your own fire data, apply the same lag.**

**Vegetation (5)** — `ndvi`, `ndvi_normal`, `ndvi_anomaly`, `ndvi_change_32d`,
`ndvi_stale_days`
From **MODIS MOD13Q1 v061** (250 m, 16-day composites), aggregated *server-side* in
Google Earth Engine over each 0.1° cell polygon — about 1,660 pixels per cell.
Point-sampling a cell centre would describe one hillside and call it the cell.
`ndvi_normal` is that cell's average for the same 16-day slot over **2000–2011**, a
baseline period deliberately disjoint from the modelling years so that no normal contains
the fires being predicted.
⚠️ A composite is used only once its 16-day window has **closed before** the label date —
a fire destroys vegetation, so a window spanning the label date would carry the burn scar,
which is the label in disguise. That is why `ndvi_stale_days` is never 0 (median 9).
⚠️ Train on `ndvi`, not `ndvi_anomaly` — see warning 4 above.
Missing on 0.05% of rows: four coastal cells that are ~95% sea, where MODIS masks the
water and too few land pixels remain.

**Split (1)** — `split`: `train` (2012–2023) / `val` (2024) / `test` (2025)

---

## Evaluate by year, never randomly

One wildfire burns for several days, so its cell-days are near-duplicates. A random split
puts the same fire on both sides of the boundary.

Measured on this data, same model, one line different:

| Split | PR-AUC | Fold σ |
|---|---|---|
| Random | 0.5567 | ±0.004 |
| **Year-grouped** | **0.4594** | **±0.076** |

21% inflated — and a fold spread **19× too tight**. The collapsed spread is the worse
half: it makes a fragile estimate look settled. Group by `year`.

Also note the base rate is unstable across years (43.0% in 2012, 7.6% in 2024), so
val and test PR-AUC are not comparable to each other. Prefer PR-AUC over ROC-AUC, and
compare models only within the same split.

---

## Baseline results

5-fold CV grouped by year. Every one of these is reproducible from the columns in this
file — nothing here depends on data that isn't distributed:

Each rung adds one *kind* of information, not one column:

| Features | PR-AUC | ROC-AUC |
|---|---|---|
| 20 weather + FWI | 0.4594 ±0.076 | 0.7356 |
| + `days_since_last_fire` | 0.5514 ±0.074 | 0.7767 |
| + `doy` | 0.5545 ±0.084 | 0.7771 |
| + vegetation (26) | 0.5700 ±0.092 | 0.7891 |
| **LightGBM tuned, all 26** | **0.5787 ±0.089** | **0.7938** |

LightGBM 0.5787, MLP(32) 0.5670, XGBoost 0.5664, RandomForest 0.5660 — a spread of 0.013
against ±0.089 fold noise. Boosted trees, bagged trees and a neural network converging
that tightly indicates an **information ceiling**, not an algorithmic one.

Vegetation was the largest remaining gap and is now measured. It is real but small: held
at the same 80% recall, deployed precision moves from about 1 alert in 52 to 1 in 50. The
ceiling held against the lever most likely to break it.

Things already measured and found not to help: NDVI **anomalies** instead of the raw
level (−0.0009), feature scaling (0.0004), class weighting (+0.001), undersampling to
50/50 (−0.002), predicting "fire within 3 days" (−0.06).

Not yet tested: NDMI (moisture, MOD09A1), terrain–weather interactions, ignition proxies
such as road or settlement distance, and sequence models over daily weather instead of
hand-built rolling windows.

---

## Limitations

1. `label = 0` means no detection, not no fire (see above).
2. Fires below the 375 m VIIRS detection threshold are invisible, biasing labels toward
   larger events.
3. Cells are previously-burned land only — the data does not describe never-burned forest,
   and a tree model trained on it cannot extrapolate to desert or urban terrain.
4. The cell universe was built from all 14 years including the test year, so test metrics
   describe fire-prone land rather than arbitrary terrain.
5. Weather is ~11 km reanalysis, not station data — slope, aspect and valley winds are
   unresolved.
6. Ignition cause is absent. Most Algerian wildfires are human-caused; this describes
   fire-conducive *conditions*, not human behaviour.
7. **The study area is a bounding box, not a national border** — see below.

---

## Why some cells look like they are in the sea

They are in **Spain**. The study area comes from `bbox: [-2.5, 34.0, 9.0, 37.2]`, a
rectangle centred on northern Algeria, and rectangles do not follow coastlines or borders.

| Group | Cells | Extent |
|---|---|---|
| Almería, **Spain** | 16 | lat 36.8–37.2, lon −2.5 to −1.9 — isolated across the Mediterranean |
| Western edge (partly **Morocco**) | 57 | lat 34.1–35.2, lon −2.5 to −1.6 |
| Eastern edge (partly **Tunisia**) | 92 | lat 34.3–37.0, lon ≥ 8.5 |

Only the Spanish group is unambiguous — the sea separates it from everything else. The
western and eastern groups sit on borders that do not follow lines of longitude, so the
exact split is not determined here. Together the three are ~12% of cells.

The Spanish cells are 427 rows (0.35%) with an identical positive rate (24.8% vs 24.9%),
but a different climate: their fire days average 27.3 °C max temperature and 38.0% minimum
RH, against 32.6 °C and 26.5% for Algerian fire days. A cooler, wetter fire regime.

**This is not leakage.** Negatives are matched within the same cell, so a Moroccan cell is
only ever compared with itself. It widens the study area; it does not contaminate the
label. If you want strictly Algerian rows, filter with a country polygon
(e.g. Natural Earth admin-0) — a latitude/longitude rule cannot do it correctly.

---

## Source code

The full pipeline that built this — API ingest, universe construction, negative sampling,
FWI implementation, and the modelling notebooks — is open source:

**https://github.com/Abdelmalek05/fire-detection-model**

---

## Licence and attribution

Released under **CC BY 4.0**. Please attribute:

> Weather data by Open-Meteo.com, licensed under CC BY 4.0
> (https://creativecommons.org/licenses/by/4.0/).
>
> We acknowledge the use of imagery from the NASA LANCE FIRMS
> (https://earthdata.nasa.gov/firms), part of the NASA Earth Science Data and
> Information System (ESDIS).
>
> Vegetation indices from MODIS/Terra MOD13Q1 v061, courtesy of the NASA Land Processes
> Distributed Active Archive Center (LP DAAC), USGS/EROS, accessed through Google Earth
> Engine.

**This data has been modified from its sources.** FIRMS 375 m detection pixels are
filtered to `type == 0` and collapsed to (0.1° cell, day) events; Open-Meteo hourly
ERA5-Land values are reduced to daily aggregates, extended with rolling windows, and used
to compute Canadian FWI components; MOD13Q1 NDVI is quality-masked (`SummaryQA <= 1`),
averaged over each 0.1° cell polygon at 250 m, scaled by 0.0001, and differenced against
a 2000–2011 per-cell seasonal normal. Rows are a sampled panel, not a complete grid.
Neither the raw detections, the raw reanalysis, nor any MODIS imagery are redistributed
here. MODIS land products carry no restrictions on subsequent use or redistribution.
Neither NASA nor Open-Meteo endorses this work.
