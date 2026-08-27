from firerisk.features import (
    MODEL_FEATURES, NDVI_FEATURES, TRAINING_FEATURES,
)
from firerisk.ndvi import NDVI_JOIN_COLS


def test_the_raw_level_is_a_feature_not_only_the_anomaly():
    """The level measured +0.0128 PR-AUC over 12/14 folds; the anomaly alone
    measured -0.0009. The original design had this backwards, so it is worth
    a test rather than a comment."""
    assert "ndvi" in NDVI_FEATURES
    assert "ndvi_normal" in NDVI_FEATURES


def test_the_anomaly_is_stored_but_not_trained_on():
    """Kept in the dataset because it costs nothing and someone will want to
    reproduce the ablation. Left out of training because raw NDVI already
    contains it - level = normal + anomaly - so it is redundant."""
    assert "ndvi_anomaly" in NDVI_JOIN_COLS
    assert "ndvi_anomaly" not in TRAINING_FEATURES


def test_training_features_include_the_ndvi_block():
    for f in NDVI_FEATURES:
        assert f in TRAINING_FEATURES


def test_training_features_have_no_duplicates():
    """doy is stored as a key column and also trained on. Appending it to the
    stored feature list instead would write it to the parquet twice."""
    assert len(TRAINING_FEATURES) == len(set(TRAINING_FEATURES))


def test_doy_is_trained_on_but_not_a_stored_model_feature():
    assert "doy" in TRAINING_FEATURES
    assert "doy" not in MODEL_FEATURES


def test_every_training_feature_is_available_at_inference():
    """Anything the model reads must be derivable from weather, fire history,
    the calendar or MODIS - never from the label or the sampling design."""
    leaks = {"sample_kind", "n_detections", "max_frp", "label", "split"}
    assert not (set(TRAINING_FEATURES) & leaks)
