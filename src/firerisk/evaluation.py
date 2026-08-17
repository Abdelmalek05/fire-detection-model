"""The evaluation protocol, shared by every modelling notebook.

Not modelling - measurement. Which model and which features belong in a
notebook; how a fold is split and scored is the ruler, and two copies of a
ruler is not one ruler. Notebooks 02 and 03 previously each had their own CV
loop and reported two different headline numbers for the same model.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold

# From a 40-configuration search. Heavy regularisation - 29 leaves, 197
# samples per leaf - is what a model picks when it is fighting noise.
TUNED_PARAMS = dict(
    objective="binary", n_estimators=1275, learning_rate=0.0146, num_leaves=29,
    min_child_samples=197, subsample=0.5998, subsample_freq=1,
    colsample_bytree=0.7671, reg_alpha=0.0667, reg_lambda=3.2759,
    n_jobs=-1, verbose=-1, random_state=42,
)

THRESHOLD_GRID = np.linspace(0.02, 0.98, 97)


def positive_proba(model, X):
    """P(fire). Resolved via classes_, since a fold whose training years hold
    no fires returns a single column and [:, 1] would then be wrong."""
    p = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    return p[:, classes.index(1)] if 1 in classes else np.zeros(len(X))


def best_threshold(y_true, proba, min_recall=None):
    """Best-F1 cutoff, optionally restricted to those meeting a recall target."""
    best, best_f1 = 0.5, -1.0
    for t in THRESHOLD_GRID:
        yhat = (proba >= t).astype(int)
        if min_recall is not None and recall_score(y_true, yhat, zero_division=0) < min_recall:
            continue
        f = f1_score(y_true, yhat, zero_division=0)
        if f > best_f1:
            best, best_f1 = t, f
    return float(best)


def cross_validate(model, X, y, groups, folds=5, random_split=False):
    """Year-grouped CV. Returns (per-fold metrics, out-of-fold probabilities).

    The threshold is fitted on an inner year and never on the fold being
    scored. random_split=True exists only to demonstrate what leakage looks
    like - a fire spans days, so random folds put one fire on both sides.
    """
    splitter = (KFold(folds, shuffle=True, random_state=42).split(X)
                if random_split else GroupKFold(folds).split(X, y, groups))
    oof, rows = np.zeros(len(X)), []

    for tr, te in splitter:
        g = groups.iloc[tr]
        inner = np.sort(g.unique())[-1]
        fit = (g != inner).values

        cal = clone(model).fit(X.iloc[tr][fit], y.iloc[tr][fit])
        thr = best_threshold(y.iloc[tr][~fit], positive_proba(cal, X.iloc[tr][~fit]))

        final = clone(model).fit(X.iloc[tr], y.iloc[tr])
        p = positive_proba(final, X.iloc[te])
        oof[te] = p
        yhat = (p >= thr).astype(int)

        rows.append(dict(
            pr_auc=average_precision_score(y.iloc[te], p),
            roc_auc=roc_auc_score(y.iloc[te], p),
            precision=precision_score(y.iloc[te], yhat, zero_division=0),
            recall=recall_score(y.iloc[te], yhat, zero_division=0),
            f1=f1_score(y.iloc[te], yhat, zero_division=0),
            threshold=thr,
        ))
    return pd.DataFrame(rows), oof


def summarise(results):
    """{name: folds_df} -> one row per model, sorted by PR-AUC.

    Carries the fold spread: a mean without it hides that these differences
    are smaller than the noise between seasons.
    """
    out = pd.DataFrame({k: v.mean() for k, v in results.items()}).T
    out["pr_auc_sd"] = [v.pr_auc.std() for v in results.values()]
    return out.sort_values("pr_auc", ascending=False).round(4)
