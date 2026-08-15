# Algeria Wildfire Risk — Dataset Pipeline

Builds a labelled fire / no-fire dataset for northern Algeria from **NASA FIRMS**
satellite fire detections and **Open-Meteo** historical weather.

Replaces the 244-row UCI "Algerian Forest Fires" dataset (2 regions, one summer)
with a dataset built from live APIs across whole fire seasons.

## Structure

```
src/firerisk/          pipeline (ingest -> sampling -> features)
  modeling/            evaluation protocol, training, calibration
scripts/               operational entry points
data/sample/           3.6k committed rows - explore without an API key
artifacts/             trained model, threshold, metrics
docs/                  design spec and implementation plan
tests/                 82 tests, none touching the network
```

## Results

| | PR-AUC | ROC-AUC | Precision | Recall |
|---|---|---|---|---|
| Weather features only | 0.458 | 0.734 | — | — |
| **+ fuel depletion, tuned** | **0.559** | 0.776 | 0.386 | **0.810** |

5-fold CV grouped by year, threshold fitted on an inner held-out year.
Against a 24.9% sampled base rate, that is a 2.2x lift.

Boosted trees, bagged trees and a neural network all land within 0.01 of each
other, which is the signature of an information ceiling rather than an
algorithmic one. The remaining headroom is in data - vegetation state (NDVI)
above all - not in model choice.

## Setup

```powershell
pip install -r requirements.txt
copy .env.example .env      # then paste your free key from
                            # https://firms.modaps.eosdis.nasa.gov/api/map_key/
$env:PYTHONPATH="src"
python scripts/00_check_apis.py     # live smoke test
```

## Run

```powershell
$env:PYTHONPATH="src"
python scripts/fetch_weather.py     # long, resumable - see note below
python -m firerisk.build features   # assemble the dataset
```

Stages can be run individually: `firms`, `universe`, `samples`, `weather`,
`features`, or `all`. Every network stage caches to disk, so re-running costs
nothing and an interrupted run resumes exactly where it stopped.

Output: `data/processed/dataset.parquet`.

### Modelling

```powershell
python scripts/run_experiment.py compare      # model families, honest CV
python scripts/run_experiment.py thresholds   # operating-point table
python scripts/run_experiment.py train        # fit + persist artifacts
```

Two things the model cannot be served without, both saved beside it in
`artifacts/`:

* **The threshold.** Defaults to whatever meets an 80% recall target (0.18),
  not 0.5. At 0.5 this model misses four fires in five.
* **The base-rate correction.** Training data is 1:3 by construction; real
  fire-days are ~1 in 100 cell-days in season. Raw probabilities are on the
  sampled scale and read as alarming everywhere until corrected.

Risk is reported as bands relative to the base rate ("3x a typical day here"),
not as a yes/no alarm - the same shape as published fire-danger ratings.

### About the weather fetch

Open-Meteo's free tier bills by **locations × days** and enforces per-IP hourly
and daily quotas, so the weather download takes hours and cannot complete in one
sitting. `scripts/fetch_weather.py` handles this: it fetches what the quota
allows, sleeps, wakes and continues, and is safe to interrupt at any point.

The feature stage deliberately works on a **partial** download, so a complete
dataset can be built for whichever years have finished.

#### Failure handling

The fetch is long enough to run unattended, so each failure gets its own answer:

| Situation | Response |
|---|---|
| Minutely block | wait it out — it clears in a minute |
| Hourly or daily block | sleep the window, then retry the whole todo list |
| A chunk fails 3 times | abandon it, finish the rest, report it, exit 75 |
| Ctrl+C | exit 130, every finished chunk already on disk |

Exit codes: `0` complete, `75` finished what it could, `130` interrupted.
Nothing is ever left half-written — chunks are written to a temp name and
renamed into place, since resumption trusts that a file which exists is done.

## Design decisions

| Decision | Choice | Why |
|---|---|---|
| Unit of analysis | one 0.1° cell on one day | matches the ERA5-Land weather grid. One fire emits dozens of satellite pixels — measured: 894 pixels collapsed to 71 real (cell, day) events. Pixel-level rows would pseudo-replicate each fire ~12× and leak it across CV folds |
| Detection filter | `type == 0` only | over the 2023 season, static industrial sources (`type=2`, 6050) **outnumbered** real vegetation fires (4443) and recur at fixed coordinates year-round. Unfiltered, most "positives" would be gas flares a tree model can memorise by location |
| Confidence | positives need `n`/`h`; exclusions use all | a low-confidence detection is not enough to assert a fire, but is ample reason not to assert its absence |
| Candidate cells | land that burned ≥2 days in ≥2 distinct years | every negative is land that *can* burn, so "is there fuel here?" cannot be the discriminator |
| Negatives | **same cell, different day** | each cell appears as both positive and negative, making cell identity statistically independent of the label. The geographic shortcut is not discouraged — it is arithmetically unavailable |
| Buffers | ±3 days own-cell, same-day neighbours | fires burn for days and VIIRS misses days to cloud; the day *before* ignition is the driest day on record, and labelling it 0 teaches the inverse of fire risk. Neighbours share one 11 km weather pixel |
| Splits | by year, never random | a random split would scatter one multi-day fire across train and test |
| Features | weather + FWI, **no coordinates** | prevents geography re-entering through lat/lon. A variant *with* coordinates is trained separately as a comparison |

## Features (20)

**Same-day (9):** `temp_max`, `temp_min`, `temp_mean`, `rh_min`, `rh_mean`,
`wind_max`, `wind_mean`, `precip_sum`, `vpd_max`

**Antecedent dryness (5):** `precip_7d`, `precip_30d`, `days_since_rain_1mm`,
`temp_max_7d_mean`, `rh_min_7d_mean`

**Canadian FWI system (6):** `ffmc`, `dmc`, `dc`, `isi`, `bui`, `fwi`

Weather is fetched from Jan 1 (not the season start) so the FWI moisture codes
and rolling windows have the antecedent history they need — the Drought Code
alone has a ~50-day memory.

## Known limitations

1. **"No detection" ≠ "no fire".** VIIRS cannot see through cloud. Cloudy days
   are systematically wetter *and* less observable, so part of any learned
   "rain → no fire" relationship reflects satellite blindness rather than fire
   physics. This cannot be fixed by sampling. It is the most important caveat on
   any result from this project.
2. Fires below the 375 m detection threshold are invisible, biasing labels
   toward larger events.
3. The universe is previously-burned land, so the model does not generalise to
   never-burned forest — and, being tree-based, cannot extrapolate beyond its
   training range. A query for desert or urban terrain would return a confident
   and meaningless "extreme risk". **The serving layer must reject
   out-of-universe points rather than score them.**
4. The universe is built from all configured years including the test years, so
   test metrics describe fire-prone land, not arbitrary terrain.
5. FWI is driven by daily aggregates (max temp, min RH, max wind, 24 h rain)
   rather than canonical noon-local observations — the standard published
   approximation, but values run slightly hot.
6. FWI day-length factors are calibrated for ~46°N; northern Algeria is ~36°N.
7. Weather is ~11 km reanalysis, not station data — slope, aspect and valley
   winds are unresolved.
8. Ignition cause is absent. Most Algerian wildfires are human-caused; this
   models fire-conducive *conditions*, not human behaviour.

## Data sources

- NASA FIRMS — https://firms.modaps.eosdis.nasa.gov/api/ (VIIRS SNPP, archive from 2012-01-20)
- Open-Meteo Archive — https://archive-api.open-meteo.com/v1/archive (ERA5-Land)

## Tests

```powershell
python -m pytest
```

No test performs live network I/O. The correctness-critical test is
`tests/test_panel.py::test_negatives_respect_all_buffers`, which asserts that no
sampled negative violates any guard.
