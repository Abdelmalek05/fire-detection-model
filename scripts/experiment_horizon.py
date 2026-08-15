"""Does predicting 'fire within the next H days' beat 'fire today'?

Cloud cover makes the exact detection day partly random - a fire starting
Tuesday may only be seen on Thursday. That timing noise is baked into a
same-day label and caps achievable performance. A horizon absorbs it.

This needs a RE-SAMPLED panel, not a re-labelled one: the existing negatives
already sit >=4 days from any fire, so every pre-fire day the horizon would
turn positive was excluded at sampling time.
"""
import argparse
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from firerisk import firms, panel
from firerisk.build import _load_cached_weather
from firerisk.config import load_config
from firerisk.features import BASE_FEATURES, add_rolling
from firerisk.fwi import compute_series
from firerisk.grid import neighbours

warnings.filterwarnings("ignore")
SEED = 42


def build_samples(cfg, horizon):
    """Positives: a fire falls in [d, d+H-1]. Negatives: nothing within the
    horizon plus the buffer on either side."""
    det_all = firms.load_detections(cfg)
    qual = firms.qualifying(det_all, cfg)
    uni = pd.read_parquet("data/interim/universe.parquet")
    keep = set(uni.cell_id)

    fire_days = {c: set(g) for c, g in
                 qual[qual.cell_id.isin(keep)].groupby("cell_id")["date"]}
    all_days = {c: set(g) for c, g in det_all.groupby("cell_id")["date"]}
    season = panel.season_days(cfg.years, cfg.season_start, cfg.season_end)
    buf = cfg.temporal_buffer_days
    rng = np.random.default_rng(cfg.random_seed)

    rows = []
    for cell, days in fire_days.items():
        arr = np.array(sorted(days), dtype="datetime64[ns]")
        sd = np.array(season, dtype="datetime64[ns]")

        # positive if any fire day lies in [d, d+H-1]
        pos_mask = np.zeros(len(sd), bool)
        for h in range(horizon):
            pos_mask |= np.isin(sd + np.timedelta64(h, "D"), arr)

        # blocked if any own-cell fire within the horizon window +/- buffer
        blocked = np.zeros(len(sd), bool)
        own = np.array(sorted(all_days.get(cell, [])), dtype="datetime64[ns]")
        for f in own:
            delta = (sd - f) / np.timedelta64(1, "D")
            blocked |= (delta >= -(horizon - 1) - buf) & (delta <= buf)
        for nb in neighbours(cell):
            for f in all_days.get(nb, []):
                delta = (sd - np.datetime64(f)) / np.timedelta64(1, "D")
                blocked |= (delta >= -(horizon - 1)) & (delta <= 0)

        pos_days = sd[pos_mask]
        if len(pos_days) == 0:
            continue
        # cap per cell per season, as the main pipeline does
        pos_df = pd.DataFrame({"date": pos_days})
        pos_df["year"] = pd.to_datetime(pos_df.date).dt.year
        pos_df = pos_df.groupby("year", group_keys=False).head(
            cfg.max_positives_per_cell_season)
        rows += [{"cell_id": cell, "date": d, "label": 1} for d in pos_df.date]

        cand = sd[~blocked & ~pos_mask]
        need = min(len(pos_df) * cfg.k_negatives, len(cand))
        if need:
            for d in rng.choice(cand, size=need, replace=False):
                rows.append({"cell_id": cell, "date": d, "label": 0})

    s = pd.DataFrame(rows)
    s["date"] = pd.to_datetime(s.date)
    return s, uni


def assemble(cfg, samples, uni):
    wx = add_rolling(_load_cached_weather(cfg, uni))
    wx = compute_series(wx)
    d = samples.merge(wx, on=["cell_id", "date"], how="inner")
    d["year"] = d.date.dt.year
    return d


def evaluate(d, feats, label="model"):
    X, y, years = d[feats], d["label"], d["year"]
    rows = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, years):
        inner = np.sort(years.iloc[tr].unique())[-1]
        fit = (years.iloc[tr] != inner).values
        m = lgb.LGBMClassifier(objective="binary", n_estimators=600,
                               learning_rate=0.05, num_leaves=63,
                               min_child_samples=40, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8,
                               reg_lambda=1.0, n_jobs=-1, verbose=-1,
                               random_state=SEED)
        m.fit(X.iloc[tr][fit], y.iloc[tr][fit])
        pc = m.predict_proba(X.iloc[tr][~fit])[:, 1]
        ths = np.linspace(0.02, 0.98, 97)
        thr = ths[int(np.argmax([f1_score(y.iloc[tr][~fit], (pc >= t).astype(int),
                                          zero_division=0) for t in ths]))]
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        yhat = (p >= thr).astype(int)
        rows.append(dict(pr_auc=average_precision_score(y.iloc[te], p),
                         roc_auc=roc_auc_score(y.iloc[te], p),
                         precision=precision_score(y.iloc[te], yhat, zero_division=0),
                         recall=recall_score(y.iloc[te], yhat, zero_division=0),
                         f1=f1_score(y.iloc[te], yhat, zero_division=0)))
    r = pd.DataFrame(rows).mean()
    print(f"  {label:<12} PR-AUC {r.pr_auc:.4f}   ROC {r.roc_auc:.4f}   "
          f"P {r.precision:.3f}   R {r.recall:.3f}   F1 {r.f1:.4f}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="1,3,5")
    args = ap.parse_args()
    cfg = load_config()

    print("horizon = 'a fire occurs within the next H days'\n")
    for h in [int(x) for x in args.horizons.split(",")]:
        s, uni = build_samples(cfg, h)
        d = assemble(cfg, s, uni)
        print(f"H={h}: rows {len(d):,}  positives {d.label.mean() * 100:.1f}%")
        evaluate(d, BASE_FEATURES, f"H={h}")
        print()


if __name__ == "__main__":
    main()
