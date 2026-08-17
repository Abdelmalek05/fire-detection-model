# `dataset.parquet` — data dictionary

A single flat table. Same shape as a CSV; Parquet only changes how it is written
to disk (typed, compressed, faster to load).

```python
import pandas as pd
d = pd.read_parquet("data/processed/dataset.parquet")
d.to_csv("dataset.csv", index=False)   # if you want it in Excel
```

---

## Shape

| | |
|---|---|
| Rows | **121,869** |
| Columns | **32** (21 model features + 11 keys, labels and bookkeeping) |
| File size | 9.8 MB on disk (~48 MB in memory) |
| Grain | one **(cell, date)** pair — unique, 0 duplicates |
| Cells | 1,423 distinct 0.1° squares (~11 km) |
| Dates | 2012-06-01 → 2025-10-31, season only (Jun 1 – Oct 31), 2,142 distinct days |
| Rows per cell | 85.6 mean, 552 max |
| Positives | 30,351 (24.9%) |
| Negatives | 91,518 (75.1%) |

The 3:1 ratio is by construction — `k_negatives: 3` in [config.yaml](../config/config.yaml).

Note: the samples table has 122,024 rows but the dataset has 121,869. The 155
missing rows are `(cell, date)` pairs with no weather cached; `assemble` uses an
inner join, so they drop out rather than arriving as NaN.

---

## Columns

### Keys — identify the row (6)

| Column | Type | Range | Meaning |
|---|---|---|---|
| `cell_id` | str | 1,423 uniq | grid square, `"<lat_idx>_<lon_idx>"` e.g. `344_73` |
| `lat` | float64 | 34.0 – 37.2 | cell centre latitude |
| `lon` | float64 | -2.5 – 9.0 | cell centre longitude |
| `date` | datetime64 | 2012-06-01 – 2025-10-31 | the day |
| `year` | int32 | 2012 – 2025 | `date.year`; also the CV grouping key |
| `doy` | int32 | 152 – 305 | day of year (Jun 1 = 152, Oct 31 = 305) |

`lat`/`lon` are for mapping and inspection. **They are not model features** — see
*Design invariants* below.

### Label and provenance (4)

| Column | Type | Range | Meaning |
|---|---|---|---|
| `label` | int64 | 0 / 1 | 1 = qualifying fire detection in this cell on this day |
| `sample_kind` | str | `positive` / `matched_negative` | how the row was sampled |
| `n_detections` | int64 | 0 – 569 | VIIRS pixels behind the label; always 0 for negatives |
| `max_frp` | float64 | 0.02 – 690.6 | strongest fire radiative power (MW); **NaN for all 91,518 negatives** |

`max_frp` is the only column with nulls, and its nulls are exactly the negatives.
Both it and `n_detections` are label-derived — **never feed them to the model.**

### Same-day weather (9)

ERA5-Land daily aggregates for the cell centre, Open-Meteo archive.

| Column | Unit | Range | Mean |
|---|---|---|---|
| `temp_max` | °C | 4.5 – 49.3 | 30.2 |
| `temp_min` | °C | -4.1 – 34.9 | 18.2 |
| `temp_mean` | °C | 1.1 – 40.9 | 23.8 |
| `rh_min` | % | 3 – 100 | 33.2 |
| `rh_mean` | % | 8 – 100 | 56.4 |
| `wind_max` | km/h | 3.6 – 54.8 | 15.0 |
| `wind_mean` | km/h | 1.9 – 37.2 | 8.0 |
| `precip_sum` | mm | 0 – 119.5 | 0.66 |
| `vpd_max` | kPa | 0 – 11.3 | 3.14 |

### Antecedent dryness (5)

Rolling windows, **backward-looking only** — computed over Jan 1 → Oct 31, so a
June 1 row already has five months of history behind it.

| Column | Unit | Range | Meaning |
|---|---|---|---|
| `precip_7d` | mm | 0 – 205.5 | rain over the trailing 7 days |
| `precip_30d` | mm | 0 – 375.2 | rain over the trailing 30 days |
| `days_since_rain_1mm` | days | 0 – 173 | since the last ≥1 mm wetting rain |
| `temp_max_7d_mean` | °C | 11.2 – 47.0 | 7-day mean of daily max temp |
| `rh_min_7d_mean` | % | 5.4 – 82.4 | 7-day mean of daily min RH |

### Canadian FWI system (6)

Recursive moisture codes, reset at every (cell, year) boundary. See [fwi.py](../src/firerisk/fwi.py).

| Column | Range | Responds over | Meaning |
|---|---|---|---|
| `ffmc` | 0 – 100.7 | hours | fine fuel moisture — litter, grass |
| `dmc` | 0.8 – 1518 | weeks | duff moisture — loosely packed organic layer |
| `dc` | 2.7 – 2116 | ~50 days | drought code — deep compact organic layer |
| `isi` | 0 – 199.9 | — | initial spread index (wind × FFMC) |
| `bui` | 0.9 – 1477 | — | buildup index (DMC + DC) — total available fuel |
| `fwi` | 0 – 225.4 | — | the headline fire weather index |

⚠️ These run **well above** canonical FWI ranges (official DMC rarely exceeds
~150). Three reasons, all documented: daily aggregates instead of noon-local
observations, day-length tables calibrated for ~46°N against Algeria's ~36°N, and
a Jan 1 reset that lets DC accumulate uninterrupted through a rainless summer.
They are internally consistent and usable as features; **do not compare them to
published FWI thresholds.**

### Fuel (1)

| Column | Unit | Range | Meaning |
|---|---|---|---|
| `days_since_last_fire` | days | 5 – 9999 | since this cell last burned; `9999` = never burned in the record |

The single most important feature — **26% of model gain**, and worth +20% PR-AUC
over the weather features alone. Burned ground does not reburn for months, and
nothing in the weather knows that.

⚠️ **It carries a mandatory lag of `temporal_buffer_days + 1` = 4 days**, which is
why the minimum is 5 rather than 0. Without the lag it encodes the sampling design
instead of fire behaviour: negatives sit at least 4 days from any fire, while a
positive inside a multi-day fire sits 1–3 days after the previous one, so every
value below 4 would be 100% positive — 8,365 rows, 27.6% of all positives. That
version scored 0.688 on pure artefact.

If you recompute this feature from your own fire data, apply the same lag.
[test_features_fuel.py](../tests/test_features_fuel.py) fails if it is removed.

### Split (1)

| Column | Type | Values |
|---|---|---|
| `split` | str | `train` / `val` / `test` |

---

## Splits

Split **by year, never randomly** — one wildfire burns for several days, so a
random split would scatter the same fire across train and test.

| Split | Years | Rows | Neg | Pos | Positive rate |
|---|---|---|---|---|---|
| `train` | 2012–2023 | 104,979 | 76,393 | 28,586 | 27.2% |
| `val` | 2024 | 8,423 | 7,780 | 643 | **7.6%** |
| `test` | 2025 | 8,467 | 7,345 | 1,122 | **13.3%** |

### ⚠️ The base rate is not stable across years

| Year | 2012 | 2014 | 2018 | 2021 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| Positive rate | 43.0% | 40.1% | 14.8% | 19.0% | 12.7% | **7.6%** | 13.3% |

Total rows per year are nearly flat (~8,400–9,400), but the positive share swings
by more than 5×. Two causes compound:

1. **Real fire activity varies by season** — 2012 and 2014 were severe.
2. **Structural.** Negatives are drawn from the pooled season-days of *all 14
   years* ([panel.py:101](../src/firerisk/panel.py#L101)), so a cell's negatives
   scatter roughly evenly across years, while its positives land only in the
   years it actually burned.

Consequences to respect:

- **Val is the hardest split** and not because 2024 is intrinsically harder — it
  has the lowest base rate in the dataset. Val PR-AUC is not comparable to test
  PR-AUC, and neither is comparable to train.
- **Any absolute probability the model emits is meaningless as a real-world fire
  probability.** The 25% base rate is a sampling artefact of `k=3`, not the true
  rate of fire on a random cell-day. Rank ordering is the usable output;
  calibration is only ever relative to this sampling scheme.
- Compare models on the same split, and prefer PR-AUC over ROC-AUC.

---

## Design invariants

Things that will silently break the dataset's guarantees if changed:

| Invariant | Why | Enforced by |
|---|---|---|
| `lat`/`lon` excluded from features | same-cell negative sampling exists to make cell identity carry no label information; feeding coordinates back hands the shortcut straight back | `BASE_FEATURES` in [features.py](../src/firerisk/features.py) |
| `n_detections`, `max_frp` excluded | derived from the label — direct leakage | (convention; not enforced in code) |
| `days_since_last_fire` carries a `buffer_days + 1` lag | without it, every value < 4 is 100% positive by sampling construction, inflating PR-AUC 0.458 → 0.688 on pure artefact | [test_features_fuel.py](../tests/test_features_fuel.py) |
| Splits by year | one fire spans days; random split leaks it across folds | `cfg.splits` |
| Rolling windows backward-only | a window that sees the future is not available at serving time | [features.py:58](../src/firerisk/features.py#L58) |

### Everything the model needs is in the file

All 21 model features are stored columns. Read the parquet directly:

```python
import pandas as pd
d = pd.read_parquet("data/processed/dataset.parquet")
```

or through `load_dataset()` in [notebooks/00_setup.py](../notebooks/00_setup.py),
which adds a guard against a near-constant fuel column and resolves paths from
the repo root:

```python
from importlib import import_module
setup = import_module("00_setup")     # with notebooks/ on sys.path
d = setup.load_dataset()
FEATURES = setup.FEATURES             # 21
```

| Feature set | Features | PR-AUC |
|---|---|---|
| `BASE_FEATURES` | 20 (weather + FWI) | 0.4594 ±0.076 |
| `MODEL_FEATURES` | 21 (+ `days_since_last_fire`) | 0.5514 ±0.074 (0.5594 tuned) |

`days_since_last_fire` derives from `temporal_buffer_days`, so `build.py` records
that buffer in the parquet's own metadata — a mismatch between the stored column
and the buffer the rows were sampled under is then detectable rather than silent:

```python
import pyarrow.parquet as pq
pq.read_schema("data/processed/dataset.parquet").metadata[b"temporal_buffer_days"]
# b'3'
```

Earlier versions of this file omitted the column and left every consumer to
recompute it. That made the dataset unusable on its own — the feature carrying
26% of model gain could not be derived without the raw FIRMS detections, which
are not distributed here.

---

## Caveats on the label itself

1. **`label = 0` means "no detection", not "no fire".** VIIRS cannot see through
   cloud, and cloudy days are systematically both wetter *and* less observable —
   so part of any learned "rain → no fire" relationship is satellite blindness,
   not fire physics. No sampling scheme fixes this. It is the most important
   caveat on any result from this project.
2. Fires below the 375 m detection threshold are invisible, biasing labels toward
   larger events.
3. The 1,423 cells are **previously-burned land only**, chosen as land that burned
   ≥2 days in ≥2 distinct years. The model does not generalise to never-burned
   forest, and being tree-based cannot extrapolate outside its training range — a
   query for desert or urban terrain would return a confident, meaningless
   "extreme risk". **The serving layer must reject out-of-universe points rather
   than score them.**
4. The universe is built from all 14 configured years, including the test year, so
   test metrics describe fire-prone land rather than arbitrary terrain.
5. Ignition cause is absent. Most Algerian wildfires are human-caused; this models
   fire-conducive *conditions*, not human behaviour.
6. **The study area is a bounding box, not a national border.** Cells come from
   `bbox: [-2.5, 34.0, 9.0, 37.2]`, centred on northern Algeria but clipping into
   neighbouring territory. See below.

### The study area crosses borders

`bbox` is a rectangle, so the cell universe is not exactly Algeria:

| Group | Cells | Extent |
|---|---|---|
| Almería, **Spain** | 16 | lat 36.8–37.2, lon −2.5 to −1.9 — isolated across the Mediterranean |
| Western edge (partly **Morocco**) | 57 | lat 34.1–35.2, lon −2.5 to −1.6 |
| Eastern edge (partly **Tunisia**) | 92 | lat 34.3–37.0, lon ≥ 8.5 |

Only the Spanish group is unambiguous — the sea separates it. The western and
eastern groups are contiguous with Algerian land and straddle borders that do not
follow lines of longitude, so the exact split is not determined here. Together the
three groups are ~12% of cells.

The Spanish cells are 427 rows (0.35%) with an identical positive rate (24.8% vs
24.9% elsewhere), but a measurably different climate — their fire days average
27.3 °C max temperature and 38.0% minimum RH, against 32.6 °C and 26.5% for
Algerian fire days, with FWI 17.6% lower. A cooler, wetter fire regime.

**This does not create leakage.** Negatives are matched within the same cell, so a
Moroccan cell is only ever compared with itself. The effect is a wider study area
than the name suggests, not a contaminated label. The Maghreb fire regime is also
broadly continuous across the Algeria–Morocco and Algeria–Tunisia borders; Almería
is the one genuinely distinct group.

Filtering to true national boundaries would need a country-polygon source
(e.g. Natural Earth admin-0). It can be applied at the assemble stage, which leaves
the cell universe — and therefore the cached weather, indexed by position within it
— untouched.

---

## Licence and attribution

The **code** in this repository is MIT (see [LICENSE](../LICENSE)). The **dataset**
is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
matching the licence of its weather source.

### Attribution

> Weather data by [Open-Meteo.com](https://open-meteo.com/), licensed under
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
>
> We acknowledge the use of imagery from the NASA LANCE FIRMS
> (https://earthdata.nasa.gov/firms), part of the NASA Earth Science Data and
> Information System (ESDIS).

Open-Meteo's data is CC BY 4.0, which permits redistribution and adaptation with
credit, a link to the licence, and a statement of changes. Their free API tier is
separately limited to non-commercial use — that condition governs *making the API
calls*, not the resulting data. NASA promotes full and open sharing of FIRMS data
with no period of exclusive access; the citation above is requested rather than
required.

### Changes made to the source data

CC BY 4.0 requires stating that the material was modified. It was:

- FIRMS 375 m detection pixels are filtered to `type == 0` and collapsed to
  (0.1° cell, day) events. Confidence filters differ deliberately between
  positives (`n`/`h`) and exclusions (all, including `l`).
- Open-Meteo hourly ERA5-Land values are reduced to daily aggregates, extended
  with backward-looking rolling windows, and used to compute Canadian FWI
  components.
- Rows are a **sampled panel**, not a complete grid: every qualifying positive
  plus three matched negatives per positive, drawn from the same cell on
  different days.

Neither the raw detections nor the raw reanalysis are redistributed here.
Neither NASA nor Open-Meteo endorses this work.
