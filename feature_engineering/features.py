"""
Point-in-time feature construction - Phase 2.

CRITICAL DISCIPLINE: every feature at row t must be computable using only
information available as of t. Forward/target returns are the one deliberate
exception and are clearly separated (prefixed `target_`) so they can never
accidentally be fed into a model as a feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = [1, 3, 5, 10]
TRAILING_WINDOWS = [20, 60, 120]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index.copy())
    close = df["Close"].astype(float)

    log_ret = np.log(close / close.shift(1))
    out["feature_log_return_1d"] = log_ret

    for w in TRAILING_WINDOWS:
        roll = log_ret.rolling(window=w, min_periods=w)
        out[f"feature_trailing_mean_{w}d"] = roll.mean()
        out[f"feature_trailing_std_{w}d"] = roll.std()
        out[f"feature_zscore_{w}d"] = (log_ret - out[f"feature_trailing_mean_{w}d"]) / out[f"feature_trailing_std_{w}d"]
        out[f"feature_rolling_vol_ann_{w}d"] = out[f"feature_trailing_std_{w}d"] * np.sqrt(252)

    out["feature_autocorr_lag1_60d"] = log_ret.rolling(60, min_periods=60).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False
    )

    for h in HORIZONS:
        fwd = np.log(close.shift(-h) / close)
        out[f"target_forward_return_{h}d"] = fwd
        out[f"target_forward_positive_{h}d"] = (fwd > 0).astype(float)
        out.loc[fwd.isna(), f"target_forward_positive_{h}d"] = np.nan

    for h in HORIZONS:
        out[f"feature_resolved_forward_return_{h}d"] = out[f"target_forward_return_{h}d"].shift(h)

    return out