"""The committed sample must not drift from the dataset it represents.

It shipped for months as 31 columns at a 50% positive rate while the real
dataset was 32 columns at 24.9%, and the two files disagreed with each other.
Nothing failed, because nothing checked.
"""
from pathlib import Path

import pandas as pd
import pytest

from firerisk.features import MODEL_FEATURES

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"
REAL_POSITIVE_RATE = 0.249


@pytest.fixture(scope="module")
def files():
    return (pd.read_parquet(SAMPLE / "dataset_sample.parquet"),
            pd.read_csv(SAMPLE / "dataset_sample.csv"))


def test_both_files_hold_the_same_rows(files):
    """Same base name, same content. They used to be 3,600 and 2,000 rows
    with different class balances."""
    parquet, csv = files
    assert len(parquet) == len(csv)
    assert list(parquet.columns) == list(csv.columns)


def test_sample_carries_every_model_feature(files):
    """A sample missing a feature cannot demonstrate the model that uses it."""
    parquet, _ = files
    missing = [f for f in MODEL_FEATURES if f not in parquet.columns]
    assert missing == [], f"sample is missing model features: {missing}"


def test_sample_preserves_the_real_class_balance(files):
    """The docs explain at length that 24.9% is itself a sampling artefact.
    A sample implying a different base rate contradicts them."""
    parquet, _ = files
    assert parquet.label.mean() == pytest.approx(REAL_POSITIVE_RATE, abs=0.02)


def test_sample_spans_every_season(files):
    parquet, _ = files
    assert parquet.year.nunique() == 14
    assert parquet.year.min() == 2012 and parquet.year.max() == 2025


def test_fuel_feature_respects_the_lag(files):
    """Same invariant as the full dataset: no value below buffer_days + 1,
    or the feature encodes the sampling design rather than fire behaviour."""
    parquet, _ = files
    burned = parquet.days_since_last_fire[parquet.days_since_last_fire < 9999]
    assert burned.min() >= 4
