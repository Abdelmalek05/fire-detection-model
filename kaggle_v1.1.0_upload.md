# Kaggle v1.1.0 upload — paste-ready content

Working notes for the "New Version" upload. Not part of the dataset.

---

## Version notes (paste into the New Version box)

```
v1.1.0 — MODIS NDVI vegetation features

BREAKING: 32 -> 37 columns, and the model feature set changes from 21 to 26.
Code that selects columns by position, or rebuilds the feature list by
exclusion, will need updating.

New columns: ndvi, ndvi_normal, ndvi_anomaly, ndvi_change_32d, ndvi_stale_days
Also: doy is now a model feature (it was previously bookkeeping only).

From MODIS MOD13Q1 v061, 250 m 16-day composites, aggregated server-side over
each 0.1 degree cell polygon (~1,660 pixels per cell) in Google Earth Engine.
Climatological normals come from 2000-2011, disjoint from the 2012-2025
modelling years.

Baseline moves 0.5594 -> 0.5787 PR-AUC (5-fold CV grouped by year); the
operating threshold moves 0.18 -> 0.19.

Train on `ndvi`, NOT `ndvi_anomaly`. The anomaly measured -0.0009 PR-AUC while
the raw level measured +0.0128 (12/14 leave-one-year-out folds, p = 0.0007).
The level is a fuel-type signature that interacts with weather; subtracting the
seasonal norm deletes it. ndvi_anomaly ships so the ablation is reproducible.

Row count, cell universe, sampling design and all pre-existing columns are
unchanged.
```

---

## Column descriptions (Data tab, 5 new fields)

**ndvi**

```
Vegetation greenness for the whole cell (MODIS MOD13Q1, 250 m), from the most
recent 16-day composite that CLOSED before this date. Range 0.06-0.85. This is
the vegetation column to train on.
```

**ndvi_normal**

```
This cell's average NDVI for the same 16-day seasonal slot across 2000-2011 - a
fuel-type signature (scrub ~0.25, oak forest ~0.65). The baseline years are
deliberately disjoint from 2012-2025, so no normal contains a fire being
predicted.
```

**ndvi_anomaly**

```
ndvi minus ndvi_normal: departure from this cell's own seasonal norm. Provided
so the ablation is reproducible - it measured -0.0009 PR-AUC and is NOT in the
trained feature set. Use ndvi instead; subtracting the norm deletes the
fuel-type signal that makes NDVI useful.
```

**ndvi_change_32d**

```
Change in NDVI over the previous two composites (~32 days). Negative means the
vegetation is curing - drying out and becoming more flammable.
```

**ndvi_stale_days**

```
Age in days of the NDVI observation at this date. Never 0: a composite is used
only once its 16-day window has closed, so a same-day fire's burn scar cannot
leak into the features. Median 9, range 1-365.
```

---

## Upload checklist

- [ ] Regenerate `dataset.csv` (43.4 MB) and copy `data/processed/dataset.parquet` (12.6 MB)
- [ ] Dataset page -> **New Version** (never a second dataset)
- [ ] Upload BOTH files, paste the version notes above
- [ ] Wait for processing to finish
- [ ] **Data** tab -> fill the 5 column descriptions above
- [ ] Replace the dataset description with `kaggle_description.md`
- [ ] Notebook -> upload the new `kaggle_starter.ipynb`, run, republish
- [ ] Check the usability score is back to 10/10

The usability score will dip until the 5 descriptions are filled in - that is
expected, not a problem with the upload.
