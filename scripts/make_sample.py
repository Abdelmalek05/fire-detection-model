"""Regenerate the committed sample in data/sample/.

The sample exists so a fresh clone has real rows to open without an API key
and hours of downloading. It must therefore look like the real dataset:
same columns, same class balance, same year spread. Both files hold the
SAME rows - one parquet, one CSV - so they cannot disagree.

    $env:PYTHONPATH="src"; python scripts/make_sample.py
"""
import argparse
from pathlib import Path

import pandas as pd

N_ROWS = 3600
SEED = 42


def make_sample(d, n=N_ROWS, seed=SEED):
    """Proportional stratified sample over (year, label).

    Stratifying rather than sampling at random guarantees every season and
    both classes appear, while keeping the 24.9% positive rate intact - a
    sample that implied a different base rate would contradict the docs.
    """
    frac = n / len(d)
    out = (d.groupby(["year", "label"], group_keys=False)
             .apply(lambda g: g.sample(max(1, round(len(g) * frac)),
                                       random_state=seed)))
    return out.sort_values(["cell_id", "date"]).reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="data/processed/dataset.parquet")
    ap.add_argument("--out", default="data/sample")
    ap.add_argument("--rows", type=int, default=N_ROWS)
    args = ap.parse_args(argv)

    d = pd.read_parquet(args.source)
    s = make_sample(d, args.rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s.to_parquet(out / "dataset_sample.parquet", index=False)
    s.to_csv(out / "dataset_sample.csv", index=False)

    print(f"source  {len(d):>8,} rows x {d.shape[1]} cols  "
          f"positives {d.label.mean()*100:.1f}%")
    print(f"sample  {len(s):>8,} rows x {s.shape[1]} cols  "
          f"positives {s.label.mean()*100:.1f}%")
    print(f"cells   {s.cell_id.nunique():,}   years {s.year.min()}-{s.year.max()}")
    print(f"wrote   {out/'dataset_sample.parquet'}")
    print(f"        {out/'dataset_sample.csv'}")


if __name__ == "__main__":
    main()
