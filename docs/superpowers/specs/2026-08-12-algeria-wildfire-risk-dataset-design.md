# Algeria Wildfire Risk Classifier — Dataset Pipeline Design

**Date:** 2026-08-12
**Scope of this spec:** the data pipeline only (FIRMS ingest → negative sampling → Open-Meteo join → labelled dataset). Modelling, FastAPI serving and the Flutter client are deliberately out of scope and get their own specs.

---

## 1. Goal

Build a labelled dataset for binary wildfire-risk classification over forested northern Algeria, from real satellite fire detections and real historical weather — replacing the 244-row, 2-region, single-season UCI "Algerian Forest Fires" dataset with a dataset two orders of magnitude larger spanning 14 fire seasons.

**Expected volume.** The measured 2023 season yielded 1,346 positive cell-days, implying ~19,000 raw positives across 14 seasons. The universe filter (§4.2) and the per-cell seasonal cap (§4.3) both reduce this, so the realistic figure is lower — on the order of **10,000–15,000 positives and 40,000–60,000 total rows**. The exact count is an output of the pipeline, not a target; the ~19,000 figure is the pre-filter upper bound and should not be quoted as the dataset size.

**Success criterion for this phase:** a reproducible `processed/dataset.parquet` whose labels and negatives survive scrutiny — specifically, a dataset where a tree model *cannot* score well by learning geography or seasonality instead of fire weather.

### Non-goals

- Model training, tuning, SHAP analysis (next spec)
- FastAPI service, Flutter client (later specs)
- Fire severity, burned area, or fire spread prediction — this is ignition-day risk only
- Real-time / forecast weather (v1 uses the historical archive only)

---

## 2. Verified API facts

All of the following were confirmed live from the development machine on 2026-08-12, not taken from documentation.

### NASA FIRMS

| Fact | Value | Note |
|---|---|---|
| Auth | `MAP_KEY` in URL path | free registration |
| Rate limit | 5,000 transactions / 10 min | ample for this pipeline |
| `day_range` max | **5** | documented as 10 in places; the API returns `400 Invalid day range. Expects [1..5]` |
| `VIIRS_SNPP_SP` coverage | 2012-01-20 → 2026-04-27 | covers all 14 target seasons |
| `VIIRS_NOAA20_SP` coverage | 2018-04-01 → 2026-05-31 | **not used** — see §4.1 |
| Area endpoint | `/api/area/csv/{KEY}/{SOURCE}/{W,S,E,N}/{day_range}/{start}` | |

**Critical finding — static heat sources dominate.** Over the full northern-Algeria bbox for the 2023 season:

| `type` | Meaning | Count |
|---|---|---|
| 0 | presumed vegetation fire | 4,443 |
| **2** | **other static land source** | **6,050** |
| 3 | offshore | 204 |

`type=2` detections (gas flares, industrial heat) **outnumber real vegetation fires**, and recur at identical coordinates across seasons — 6 of 8 static 1 km cells observed in July 2023 also appeared in January 2023. Without the `type==0` filter, the majority of "positives" would be fixed-coordinate industrial sources that any tree model can memorise by location. **This filter is the single highest-impact decision in the pipeline.**

**Confidence is categorical for VIIRS** (`l` / `n` / `h`), unlike MODIS's 0–100 scale. In the 2023 season sample, low-confidence was ~13% of vegetation detections.

**Pixel redundancy.** One fire emits dozens of 375 m pixels. Measured over the Kabylie bbox: 894 vegetation pixels → 48 distinct 0.1° cells → 71 distinct (cell, date) pairs. Treating pixels as rows would pseudo-replicate each fire ~12×, inflating the dataset and leaking single fires across CV folds.

### Open-Meteo Archive

| Fact | Value |
|---|---|
| Auth | none |
| Multi-location | comma-separated `latitude` / `longitude` → JSON array |
| **Max locations/call** | **~200** (500 → `HTTP 414 Request-URI Too Large`) → chunk at 150 |
| Rate limiting | weighted by locations × days; `HTTP 429 Minutely API request limit exceeded` hit readily |
| Grid snapping | requests for 36.71 and 36.75 both returned `36.731106` — **ERA5-Land ~0.1° (~11 km)** |
| Throughput | 200 locations × 214 days = 1.42 MB in ~11 s |

The grid-snapping observation is load-bearing: weather resolution is ~11 km **regardless of how precisely coordinates are specified**, which is what makes a 0.1° analysis cell the natural unit.

---

## 3. Unit of analysis

**One row = one 0.1° grid cell on one day.**

```
cell_id = f"{round(lat/0.1):d}_{round(lon/0.1):d}"
label   = 1 if >=1 qualifying detection in that cell that day else 0
```

Chosen because it matches the ERA5-Land grid exactly. A finer 0.05° grid would place four analysis cells inside one weather pixel, producing identical feature vectors with conflicting labels — irreducible noise that caps achievable performance. A coarser wilaya-level unit would average weather over ~3,000 km² of heterogeneous terrain and yield too few rows for year-based splits.

---

## 4. Pipeline

```
[1] FIRMS ingest      5-day chunks x 14 seasons     -> data/raw/firms/{year}/{start}.csv
      | type==0, confidence in {n,h}, snap to 0.1 deg
      v
[2] Universe          burnable-land mask            -> data/interim/universe.parquet
      v
[3] Panel + sampling  positives + buffered negatives-> data/interim/samples.parquet
      v
[4] Weather           Open-Meteo, cached/throttled  -> data/raw/weather/{year}/{chunk}.parquet
      v
[5] Features + FWI    rolling dryness + FWI         -> data/processed/dataset.parquet
```

Every stage writes a cached artifact and is independently re-runnable. Network stages are idempotent: an existing cache file is never re-fetched.

### 4.1 Stage 1 — FIRMS ingest

- Source: **`VIIRS_SNPP_SP` only.**
  *Rationale:* adding `VIIRS_NOAA20_SP` (2018+) would increase overpasses and detection probability from 2018 onward, placing a step-change in label probability in the middle of the time series. Year-based splits would then read a sensor-fleet change as real signal. **Label consistency across all 14 years beats raw sensitivity.**
- BBox `-2.5, 34.0, 9.0, 37.2` (Tell Atlas forested belt).
- Season Jun 1 – Oct 31; 5-day chunks → 31 requests/season → ~434 requests total.
- Filters: `type == 0`, `confidence in {n, h}`.
- Retry with backoff; cache per chunk.

### 4.2 Stage 2 — Universe (burnable-land mask)

A cell is eligible if it recorded **≥2 fire-days across ≥2 distinct years** in 2012–2025.

*Rationale:* every negative is then drawn from land demonstrably capable of burning, so "is there fuel here?" cannot be the discriminator. Requiring two distinct years (rather than ≥1 detection) mitigates the inclusion bias where a cell enters the universe solely because of the one fire we are trying to predict, and screens out one-off false detections.

*Accepted limitation:* never-burned forest is excluded, so the model answers *"when does fire-prone land burn?"* rather than *"where can fire occur?"*. This matches the stated project goal. An external landcover mask (ESA WorldCover) is the v2 upgrade if a true fuel definition is wanted.

### 4.3 Stage 3 — Panel construction and negative sampling

Positives are all `(cell, date)` pairs in the universe with a qualifying detection, capped at **10 positives per cell per season** so that hotspot cells (max observed: 50 fire-days in one season, vs. median 1) cannot dominate.

For each positive `(c, d)`, draw `k = 3` negatives `(c, d')` satisfying **all** of:

```
d'  in fire season (Jun 1 - Oct 31) of any year in 2012..2025
no type-0 detection in cell c on d'
no type-0 detection in any of c's 8 neighbours on d'
abs(d' - d'') > 3 calendar days for EVERY fire-day d'' recorded in cell c
```

**Two definitional points that must not be left implicit:**

- **The exclusion tests use *all* `type==0` detections, including low-confidence (`l`) ones — a deliberately wider filter than the one used to define positives (`{n, h}`).** The two filters answer different questions. A positive must be a trustworthy fire; a negative must be a day with *no reason to suspect* fire. A low-confidence detection is insufficient evidence to assert a fire, but ample reason not to assert its absence. Using `{n, h}` for both would silently convert every low-confidence detection into a labelled negative — injecting exactly the false negatives the buffers exist to prevent.
- **`abs(d' - d'')` is an absolute calendar-date difference, not a day-of-year difference.** Fire-days in other years are therefore automatically far apart and impose no practical constraint; the buffer binds only within the positive's own season, which is the intent. Comparing day-of-year would wrongly exclude the same calendar week across all 14 years.

Each guard blocks a specific failure mode:

- **Temporal buffer, both directions (±3 days).** Fires burn across multiple days and VIIRS misses days to cloud cover and overpass timing, so days adjacent to a detection are frequently un-detected fire days. More importantly the day *before* ignition is typically the driest, highest-risk day in the record; labelling it `0` teaches the model the exact inverse of fire risk.
- **Neighbour buffer.** A fire near a cell boundary emits pixels into adjacent cells, and at 11 km resolution neighbouring cells share one ERA5 weather pixel — an unbuffered neighbour is a near-duplicate feature vector carrying the opposite label.
- **Same cell on both sides of the label.** Every cell appears as both positive and negative, so cell identity is statistically independent of the label. The geographic shortcut is not discouraged; it is arithmetically unavailable.

Negatives may be drawn from any year in scope, not only the positive's own year. Inter-annual dryness variation (2021 and 2023 were severe seasons) is genuine signal the model should be able to use, and it is expressed through the weather features rather than through a year identifier.

### 4.4 Stage 4 — Weather join

For every **universe cell**, fetch daily weather for **Jan 1 → Oct 31** of each year — not per sample.

Two reasons: FWI and rolling-window features require continuous antecedent weather, and fetch cost then scales with cell count (~1,500–2,500) rather than sample count (~76,000).

- Chunk 150 cells per call; ~14 chunks × 14 years ≈ 200 calls.
- Throttle plus exponential backoff on `429`; persistent parquet cache keyed by `(year, chunk)` so reruns cost nothing.
- Variables: `temperature_2m_max`, `temperature_2m_min`, `temperature_2m_mean`, `relative_humidity_2m_min`, `relative_humidity_2m_mean`, `wind_speed_10m_max`, `wind_speed_10m_mean`, `precipitation_sum`.
- Cell centroid coordinates are used for the request; Open-Meteo snaps to its own grid and the returned coordinates are stored for verification.

**Daily aggregates, not hourly.** Canonical FWI is defined on noon local-standard-time observations. Daily max-temp / min-RH / max-wind / 24 h-precip is the standard published approximation and is effectively what the UCI Algerian dataset uses. Hourly would be ~24× the data volume for v1. The deviation is documented; hourly extraction at 12:00 local is the v2 upgrade.

### 4.5 Stage 5 — Features

Base features (weather only — **no coordinates**):

| Group | Features |
|---|---|
| Same-day | `temp_max`, `temp_min`, `temp_mean`, `rh_min`, `rh_mean`, `wind_max`, `wind_mean`, `precip_sum` |
| Antecedent dryness | `precip_7d`, `precip_30d`, `days_since_rain_1mm`, `temp_max_7d_mean`, `rh_min_7d_mean` |
| FWI system | `ffmc`, `dmc`, `dc`, `isi`, `bui`, `fwi` |

**Coordinates are excluded from the baseline feature set** so geography cannot re-enter through `lat`/`lon`. A second variant adding `lat`, `lon`, `elevation` will be trained as an explicit comparison — the gap between the two is itself an informative result for the project writeup.

**FWI implementation.** Standard Van Wagner / Canadian Forest Service formulas for FFMC, DMC, DC, ISI, BUI, FWI. Spin-up begins Jan 1 each year with the conventional startup values `FFMC=85, DMC=6, DC=15`; by Jun 1 that is 152 days of spin-up, adequate for FFMC (hours–days) and DMC (weeks), and sufficient for DC given its ~50-day time constant. Day-length factors `Le` (DMC) and `Lf` (DC) come from the standard published tables, which are calibrated for ~46°N; northern Algeria is ~36°N, so these are an approximation and are documented as such.

---

## 5. Dataset schema

`data/processed/dataset.parquet`

| Column | Type | Notes |
|---|---|---|
| `cell_id` | str | `"367_41"` |
| `lat`, `lon` | float | cell centroid |
| `date` | date | |
| `year`, `doy` | int | for splitting / diagnostics, **not** features |
| `label` | int8 | 1 = fire, 0 = no fire |
| `sample_kind` | str | `positive` / `matched_negative` / `panel_negative` |
| `n_detections` | int | pixels backing a positive; 0 for negatives |
| `max_frp` | float | fire radiative power, positives only — diagnostic, not a feature |
| *feature columns* | float | as listed in §4.5 |
| `split` | str | `train` / `val` / `test` |

---

## 6. Splits and evaluation

**Split by year, never randomly.** A random split would scatter one multi-day fire across train and test.

| Split | Years |
|---|---|
| train | 2012–2021 |
| val | 2022–2023 |
| test | 2024–2025 |

**Two evaluation sets, reported separately:**

1. **Matched test set** (1:3) — measures whether the model learned fire weather with location held constant.
2. **Natural-panel test set** — an unmatched random subsample of all eligible cell-days in the test years, at the true base rate.

This distinction is essential and must be stated in any reported result: matched sampling deliberately destroys the base rate (1:3 versus a true rate nearer 1:2000). **Recall transfers between the two; precision does not.** Reporting only matched-set precision would materially overstate real-world performance. Decision threshold is calibrated on the natural panel, not the matched set.

Metrics: precision, recall, F1, PR-AUC (primary — appropriate under imbalance), ROC-AUC (secondary). Accuracy is not reported.

An additional `GroupKFold` by `cell_id` diagnostic is run to confirm the model does not depend on cell identity.

---

## 7. Known limitations

These are inherent to FIRMS-derived labels and are documented rather than fixed.

1. **"No detection" ≠ "no fire."** VIIRS cannot see through cloud. Cloudy days are systematically both wetter and less observable, so part of any learned "rain → no fire" relationship reflects satellite blindness rather than fire physics. This cannot be corrected by sampling. It is the most important caveat on any result this project produces.
2. **Small fires below the 375 m detection threshold are invisible**, biasing labels toward larger events.
3. **Universe restricted to previously-burned land**, so the model does not generalise to never-burned forest.
4. **The universe is defined using all 14 years, including the test years.** Test-set cells are therefore pre-selected as cells known to burn at some point in 2012–2025. This does not leak the label of any individual `(cell, day)` — positives and negatives are drawn from the same cells, so the selection is label-neutral within a cell — but it does mean test metrics describe performance *on fire-prone land*, not on arbitrary terrain. Rebuilding the universe from training years only would remove this, at the cost of shrinking the test universe; it is recorded here as a deliberate, documented trade rather than an oversight.
5. **FWI day-length factors** are calibrated for Canadian latitudes (§4.5).
6. **Weather is ~11 km reanalysis**, not station observation; local topographic effects (slope, aspect, valley winds) are unresolved.
7. **Ignition cause is absent.** Most Algerian wildfires are human-caused; weather governs spread conditions and receptiveness, not ignition itself. The model predicts fire-conducive conditions, not human behaviour.

---

## 8. Configuration

All tunables in `config/config.yaml`; nothing hardcoded.

```yaml
bbox:        [-2.5, 34.0, 9.0, 37.2]
resolution:  0.1
years:       [2012, 2025]
season:      {start: "06-01", end: "10-31"}
warmup_start: "01-01"
firms:
  source:      VIIRS_SNPP_SP
  types:       [0]
  confidence:  [n, h]
  day_range:   5
sampling:
  k_negatives:                    3
  temporal_buffer_days:           3
  spatial_buffer_neighbours:      true
  max_positives_per_cell_season: 10
universe:
  min_fire_days:  2
  min_distinct_years: 2
weather:
  chunk_size: 150
splits:
  train: [2012, 2021]
  val:   [2022, 2023]
  test:  [2024, 2025]
```

---

## 9. Repository layout

```
fire-detection/
  .env                     FIRMS_MAP_KEY  (gitignored)
  .env.example
  config/config.yaml
  data/
    raw/firms/  raw/weather/  interim/  processed/     (gitignored)
  src/firerisk/
    config.py    grid.py     firms.py    universe.py
    panel.py     weather.py  fwi.py      features.py    build.py
  scripts/00_check_apis.py
  tests/
    test_grid.py  test_universe.py  test_panel.py  test_fwi.py
  docs/superpowers/specs/
```

---

## 10. Testing

Pure logic is unit-tested against fixtures; no test performs live network I/O.

| Test | Asserts |
|---|---|
| `test_grid` | round-trip lat/lon ↔ cell_id; neighbour sets; boundary rounding |
| `test_universe` | ≥2-days/≥2-years rule; single-year repeat cells excluded |
| `test_panel` | **no sampled negative violates any buffer** — the core correctness test; neighbour exclusion honoured; per-cell positive cap enforced; every negative's cell also appears as a positive |
| `test_fwi` | components reproduce published Van Wagner worked examples within tolerance |

A `scripts/00_check_apis.py` smoke test verifies live credentials and endpoint behaviour on demand, separately from the unit suite.

---

## 11. Security note

The FIRMS `MAP_KEY` was pasted into a chat transcript during design. It is free and rate-limited, but **regenerating it is recommended**. It must live in `.env` (gitignored) and never in `config.yaml` or source.
