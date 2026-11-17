"""dp1_lc_reader.py: load the shared DP1 periodic-variable light curves.

    import sys; sys.path.insert(0, "/home/<owner>/<shared-dir>")
    import dp1_lc_reader as R
    cat = R.load_catalog()                    # one row per star
    lc  = R.load_lightcurve(cat.objectId[0])  # all epochs of one star
    ph  = R.phase_fold(lc.mjd, cat.period_final_d[0])
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load_catalog(directory=HERE):
    return pd.read_parquet(Path(directory) / "catalog.parquet")


def load_showcase(directory=HERE):
    return pd.read_csv(Path(directory) / "showcase.csv")


def load_lightcurves(directory=HERE, object_ids=None, bands=None):
    filters = []
    if object_ids is not None:
        filters.append(("objectId", "in", [int(i) for i in np.atleast_1d(object_ids)]))
    if bands is not None:
        filters.append(("band", "in", list(bands)))
    return pd.read_parquet(Path(directory) / "lightcurves.parquet", filters=filters or None)


def load_lightcurve(object_id, directory=HERE, bands=None):
    return load_lightcurves(directory, [object_id], bands).sort_values("mjd").reset_index(drop=True)


def phase_fold(mjd, period, t0=None):
    mjd = np.asarray(mjd, float)
    t0 = mjd.min() if t0 is None else t0
    return ((mjd - t0) / period) % 1.0
