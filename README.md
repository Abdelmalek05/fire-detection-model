# Algeria Wildfire Risk

Predicting daily wildfire risk for northern Algeria from **NASA FIRMS** satellite fire
detections and **Open-Meteo** reanalysis weather — dataset built from live APIs, not
downloaded ready-made.

**121,869 labelled cell-days · 14 fire seasons (2012–2025) · 1,423 grid cells · 6.06M weather readings**

```
PR-AUC 0.559 · ROC-AUC 0.781 · recall 0.810 at threshold 0.18
5-fold cross-validation grouped by year
```

The comparable public dataset — UCI "Algerian Forest Fires" — has 244 rows from two
regions in one summer. This one covers every fire season since VIIRS came online.

**Documentation:** [dataset dictionary](docs/DATASET.md) ·
[modelling notebooks](notebooks/)

---

## The result

Five-fold cross-validation, **grouped by year**, threshold fitted on an inner held-out
year and never on the fold being scored:

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Weather features only | 0.4594 ±0.076 | 0.7356 | 0.374 | 0.770 | 0.502 |
| **+ fuel depletion** | **0.5514** ±0.074 | 0.7767 | 0.465 | 0.653 | 0.536 |
| LightGBM (tuned) | 0.5594 ±0.074 | 0.7814 | 0.450 | 0.693 | 0.537 |
| MLP (32 hidden) | 0.5551 ±0.073 | 0.7772 | 0.451 | 0.674 | 0.538 |
| XGBoost | 0.5494 ±0.073 | 0.7743 | 0.441 | 0.692 | 0.538 |
| RandomForest | 0.5463 ±0.077 | 0.7725 | 0.463 | 0.640 | 0.534 |
| ExtraTrees | 0.5130 ±0.089 | 0.7613 | 0.446 | 0.643 | 0.524 |

Against a 24.9% base rate, 0.559 is a **2.2× lift**.

**Shipped model:** LightGBM at threshold 0.18 — recall **0.810**, precision 0.386,
out-of-fold PR-AUC 0.558. Threshold, base rates and band edges are stored in
`artifacts/metrics.json` beside the model, because the model is unusable without them.

### One feature carries the model

| Feature | Gain |
|---|---|
| `days_since_last_fire` | **26.0%** |
| `dc` (Drought Code) | 10.8% |
| `rh_mean` | 9.5% |
| `fwi` | 8.2% |
| `ffmc` | 5.3% |

Twenty weather and fire-weather-index features reach 0.459. Adding one fuel-depletion
feature — how long since this cell last burned — takes it to 0.551. Burned ground does
not reburn for months, and nothing in the weather knows that.

### Every model family lands in the same place

LightGBM 0.5594, MLP 0.5551, XGBoost 0.5494, RandomForest 0.5463 — a spread of 0.013
against a fold standard deviation of ±0.074. Boosted trees, bagged trees and a neural
network failing *identically* is the signature of an **information ceiling**, not an
algorithmic one. More model tuning is not the lever; vegetation data (NDVI) is.

---

## Two numbers you cannot serve this model without

**The threshold is 0.18, not 0.5.** Threshold choice alone moved F1 from 0.282 to 0.50
with no change to the model. At the 0.5 default it misses two fires in three:

| Threshold | Precision | Recall | Fires missed | False alarms | Deployed precision |
|---|---|---|---|---|---|
| 0.10 | 0.319 | 0.935 | 1,966 | 60,507 | 0.014 |
| **0.18** | **0.386** | **0.810** | **5,777** | **39,017** | **0.019** |
| 0.25 | 0.449 | 0.690 | 9,416 | 25,648 | 0.024 |
| 0.50 | 0.652 | 0.320 | 20,636 | 5,187 | 0.054 |

**Probabilities need base-rate correction before display.** Training is 1:3 by
construction; real fire-days are ~1 in 100 eligible cell-days in season. Raw
probabilities are on the sampled scale and read as alarming everywhere.

That last column is the honest one: at the operating point roughly **1 alert in 53** is a
real fire. Precision measured on sampled data is optimistic; recall is not, because recall
only involves actual fires.

So risk is reported as **multiples of the base rate**, not absolute cutoffs — corrected
probabilities compress into ~0–0.11, and "Extreme > 0.5" would never fire:

| Band | Fire rate |
|---|---|
| Low (<1× base rate) | 12.5% |
| Moderate (1–3×) | 35.5% |
| High (3–8×) | 60.7% |
| **Extreme (>8×)** | **81.4%** |

A monotone 12.5% → 81.4% climb. For an application that ranks *where attention should go
today*, this ranking is the product — not the raw probability.

---

## What didn't work

Most of the effort in this project went into finding out which improvements were real.
These are the ones that weren't.

### A label leak in the best feature

`days_since_last_fire` at lag 0 inflated PR-AUC from 0.458 to **0.688**. It looked like
the best result in the project. It was an artifact of the sampling design.

Negatives are sampled at least 4 days from any fire, while a positive inside a multi-day
fire sits 1–3 days after the previous one. So every value below 4 was **100% positive** —
8,365 rows, 27.6% of all positives — classified by a rule that encoded how the dataset was
built, not how fire behaves.

The fix is a mandatory lag of `temporal_buffer_days + 1`. It was caught because +50% from
five features was implausible, not because anything errored.
`tests/test_features_fuel.py` fails if the guard is removed.

### What a random split would have reported

| Split | PR-AUC | Fold σ |
|---|---|---|
| Random | 0.5567 | ±0.004 |
| **Year-grouped (honest)** | **0.4594** | **±0.076** |

21% inflated — and the fold spread **19× too tight**. The fake precision is the worse
half: it makes a fragile estimate look settled. Random folds all look alike because each
contains fragments of the same multi-day fires.

### Null results, each measured rather than assumed

| Idea | Result |
|---|---|
| Extended weather features | **−3.8%** — more features from the same measurements is not more information |
| Predicting "fire within 3 days" | **0.454 → 0.391** — a wider window adds label noise, it doesn't absorb it |
| Feature scaling for LightGBM | **0.0004** spread across raw/Standard/MinMax/Quantile — trees are invariant |
| `class_weight="balanced"` | 0.5594 → 0.5606 — inside noise |
| Undersampling to 50/50 | 0.5594 → 0.5570 — discards half the negatives, gains nothing |

The balancing experiment carries its own lesson. Trained on balanced data and scored on
**balanced** folds, the same model reports **0.7746** — because the PR-AUC baseline moved
from 0.249 to 0.50. Nothing improved. Every arm above is scored on untouched folds for
exactly that reason: *the training distribution may change, the evaluation distribution
may not.*

### Predictions that were wrong

Kept because they're the reason the ablations exist:

- "The extended weather features are where I'd put my money" → they hurt by 3.8%
- "Tree models will beat the neural net" → the MLP came second by 0.004
- "The real base rate is ~1 in 2000" → it is ~1 in 100

---

## Dataset design

The negative class is where a fire dataset is won or lost. Random negatives — desert
points, winter days — produce a model that scores 0.95 by learning "is this the Sahara in
January?"

| Decision | Choice | Why |
|---|---|---|
| Unit of analysis | one 0.1° cell on one day | matches the ERA5-Land grid. One fire emits dozens of pixels — measured: 894 pixels collapsed to 71 real (cell, day) events. Pixel rows would pseudo-replicate each fire ~12× |
| Detection filter | `type == 0` only | over the 2023 season, static industrial sources (`type=2`, 6,050) **outnumbered** real vegetation fires (4,443) and recur at fixed coordinates year-round |
| Confidence | positives need `n`/`h`; exclusions use all | a low-confidence detection is not enough to assert a fire, but is ample reason not to assert its absence |
| Candidate cells | land that burned ≥2 days in ≥2 distinct years | every negative is land that *can* burn, so "is there fuel here?" cannot be the discriminator |
| **Negatives** | **same cell, different day** | each cell appears as both positive and negative, making cell identity statistically independent of the label. Measured: per-cell positive rate σ = 0.0074. The geographic shortcut isn't discouraged — it's arithmetically unavailable |
| Buffers | ±3 days own-cell, same-day neighbours | fires burn for days and VIIRS misses days to cloud. The day *before* ignition is the driest on record; labelling it 0 teaches the inverse of fire risk |
| Splits | by year, never random | one multi-day fire would otherwise scatter across train and test |
| Features | weather + FWI, **no coordinates** | prevents geography re-entering through lat/lon |

A consequence worth naming: because negatives come from the same cells as positives, **any
purely static per-cell feature carries no main effect** — elevation, land cover and slope
are all provably empty in this design. Only time-varying information can help.

### Features (21)

Nine same-day weather variables, five antecedent-dryness windows, the six Canadian FWI
components, and one fuel-depletion feature. Weather is fetched from January 1 rather than
the season start so the FWI moisture codes have their antecedent history — the Drought
Code alone has a ~50-day memory.

Full column reference — types, units, ranges, means, and the per-year base-rate
instability that makes val and test incomparable — in
**[docs/DATASET.md](docs/DATASET.md)**.

---

## Repository

```
src/firerisk/     pipeline: grid, firms, universe, panel, weather, fwi, features, build
notebooks/        all modelling
  01              dataset overview and physical sanity checks
  02              model comparison + the random-split leakage demonstration
  03              threshold calibration, base-rate correction, final artifact
scripts/          operational entry points
artifacts/        model.joblib + metrics.json (threshold and base rates live here)
data/sample/      3,600 committed rows — explore without an API key
docs/             dataset dictionary
tests/            68 tests, none touching the network
```

## Setup

```powershell
pip install -r requirements.txt
copy .env.example .env      # free key: https://firms.modaps.eosdis.nasa.gov/api/map_key/
$env:PYTHONPATH="src"
python scripts/00_check_apis.py     # live API smoke test
```

## Run

```powershell
$env:PYTHONPATH="src"
python scripts/fetch_weather.py     # long, resumable — see below
python -m firerisk.build all        # assemble the dataset
python -m pytest                    # 68 tests
```

Stages run individually: `firms`, `universe`, `samples`, `weather`, `features`, `all`.
Every network stage caches to disk, so re-running costs nothing and an interrupted run
resumes where it stopped. Output: `data/processed/dataset.parquet`.

Modelling lives in the notebooks:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/02_model_comparison.ipynb
```

### The weather fetch

Open-Meteo's free tier bills by **locations × days** with per-IP hourly and daily quotas,
so the download takes hours and cannot finish in one sitting.
`scripts/fetch_weather.py` fetches what the quota allows, sleeps, wakes and continues, and
is safe to interrupt at any point.

| Situation | Response |
|---|---|
| Minutely block | wait it out — clears in a minute |
| Hourly or daily block | sleep the window, retry the whole todo list |
| A chunk fails 3 times | abandon it, finish the rest, report it, exit 75 |
| Ctrl+C | exit 130, every finished chunk already on disk |

Exit codes: `0` complete, `75` finished what it could, `130` interrupted. Chunks are
written to a temp name and renamed into place, since resumption trusts that a file which
exists is complete.

The feature stage deliberately works on a **partial** download, so a dataset can be built
for whichever years have finished.

---

## Limitations

1. **"No detection" ≠ "no fire".** VIIRS cannot see through cloud, and cloudy days are
   systematically wetter *and* less observable. Part of any learned "rain → no fire"
   relationship reflects satellite blindness rather than fire physics. This cannot be fixed
   by sampling and is the most important caveat on every result here.
2. Fires below the 375 m detection threshold are invisible, biasing labels toward larger
   events.
3. The universe is previously-burned land, so the model does not generalise to never-burned
   forest — and being tree-based, cannot extrapolate. A desert or urban query would return
   a confident, meaningless "extreme risk". **A serving layer must reject out-of-universe
   points rather than score them.**
4. The universe is built from all configured years including test years, so test metrics
   describe fire-prone land, not arbitrary terrain.
5. **The FWI values run well above canonical ranges** and must not be compared to
   published fire-danger thresholds — daily aggregates instead of noon-local observations,
   day-length tables calibrated for ~46°N against Algeria's ~36°N, and ~11 km reanalysis
   that leaves slope, aspect and valley winds unresolved. All three are quantified in
   [docs/DATASET.md](docs/DATASET.md).
6. **Ignition cause is absent.** Most Algerian wildfires are human-caused. This models
   fire-conducive *conditions*, not human behaviour — and some of the gap between 0.56 and
   1.0 is not information any vegetation index can buy.

This is a learning and portfolio project. It is not validated for operational fire
management.

## Status

Complete: dataset pipeline, modelling, calibrated artifact.
Planned: FastAPI service, Flutter app.

Highest-value next step is vegetation state — NDVI and NDMI — which is the one information
gap the ablations above could not close.

## Data sources

- **NASA FIRMS** — https://firms.modaps.eosdis.nasa.gov/api/ (VIIRS SNPP, archive from 2012-01-20)
- **Open-Meteo Archive** — https://archive-api.open-meteo.com/v1/archive (ERA5-Land)

## License

MIT — see [LICENSE](LICENSE).
