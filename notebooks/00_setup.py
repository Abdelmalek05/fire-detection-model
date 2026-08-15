"""Shared bootstrap for the modelling notebooks.

Not modelling code - just the boilerplate every notebook would otherwise
repeat: find the project root from wherever Jupyter was launched, put `src`
on the path, and load config. Keeping it here means the notebooks open with
the analysis rather than five cells of setup.

    from setup import project_root, load_dataset, FEATURES
"""
import sys
from pathlib import Path


def project_root():
    """Walk up until we find the marker files, so the notebooks work whether
    Jupyter was started in notebooks/ or the repo root."""
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "config" / "config.yaml").exists():
            return candidate
    raise RuntimeError("could not locate the project root")


ROOT = project_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from firerisk import firms  # noqa: E402
from firerisk.config import load_config  # noqa: E402
from firerisk.features import (  # noqa: E402
    BASE_FEATURES, MODEL_FEATURES, add_days_since_last_fire,
)

import dataclasses  # noqa: E402

# config.yaml stores data_dir as a RELATIVE path ("data"), which resolves
# against the working directory. Jupyter runs with cwd=notebooks/, so it would
# point at notebooks/data/ - and load_detections returns an EMPTY frame for a
# missing directory rather than raising. The fuel feature then silently
# becomes a constant sentinel and every model scores identically. Anchor it to
# the project root so the notebooks cannot hit that.
CFG = dataclasses.replace(
    load_config(ROOT / "config" / "config.yaml"), data_dir=ROOT / "data"
)
FEATURES = MODEL_FEATURES


def load_dataset(with_fuel=True):
    """The built dataset, plus the fuel feature the pipeline does not store.

    days_since_last_fire is derived here rather than baked into the parquet
    because it depends on the sampling buffer - storing it would freeze that
    choice into the data file.
    """
    import pandas as pd

    d = pd.read_parquet(ROOT / "data" / "processed" / "dataset.parquet")
    if with_fuel:
        det = firms.load_detections(CFG)
        if det.empty:
            raise RuntimeError(
                f"no FIRMS detections under {CFG.data_dir}. Without them the "
                "fuel feature degrades to a constant and every model scores "
                "the same - fail loudly rather than report that."
            )
        qual = firms.qualifying(det, CFG)
        d["days_since_last_fire"] = add_days_since_last_fire(
            d, qual, CFG.temporal_buffer_days
        )
        if d["days_since_last_fire"].nunique() < 10:
            raise RuntimeError(
                "days_since_last_fire is near-constant - it carries no signal."
            )
    return d
