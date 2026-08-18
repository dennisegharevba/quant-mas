"""
Model 6 - Seasonality (commodity-specific) - Phase 10.

Conditions Model 1's forecast on the CALENDAR MONTH a trade would start
in, using only historical instances that started in the same month. Same
fallback pattern as Model 2: falls back to Model 1's unconditional
forecast when there is not enough same-month evidence. CI now included
(reused from Model 1's block bootstrap) so this can feed the ranking
engine's significance gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = [1, 3, 5, 10]
MIN_SAMPLES = 60
MIN_MONTH_SAMPLES = 20


def forecast_series(features: pd.DataFrame, horizon: int,
                     min_month_samples: int = MIN_MONTH_SAMPLES) -> pd.DataFrame:
    from models.statistical.baseline import _bootstrap_ci

    resolved_col = f"feature_resolved_forward_return_{horizon}d"
    if resolved_col not in features.columns:
        raise KeyError(f"{resolved_col} not found - did you run build_features first?")

    resolved = features[resolved_col].to_numpy()
    dates = features.index
    n = len(features)

    origination_month = np.full(n, -1, dtype=int)
    for i in range(horizon, n):
        origination_month[i] = dates[i - horizon].month

    exp_ret = np.full(n, np.nan)
    prob_pos = np.full(n, np.nan)
    ci_lo = np.full(n, np.nan)
    ci_hi = np.full(n, np.nan)
    n_samples = np.zeros(n, dtype=int)
    seasonal_conditioned = np.zeros(n, dtype=bool)

    for i in range(n):
        window_resolved = resolved[: i + 1]
        window_month = origination_month[: i + 1]
        valid = ~np.isnan(window_resolved)
        window_resolved = window_resolved[valid]
        window_month = window_month[valid]
        n_samples[i] = len(window_resolved)
        if len(window_resolved) == 0:
            continue

        exp_ret[i] = window_resolved.mean()
        prob_pos[i] = (window_resolved > 0).mean()
        basis = window_resolved

        current_month = dates[i].month
        same_month = window_month == current_month
        if same_month.sum() >= min_month_samples:
            seasonal_resolved = window_resolved[same_month]
            exp_ret[i] = seasonal_resolved.mean()
            prob_pos[i] = (seasonal_resolved > 0).mean()
            seasonal_conditioned[i] = True
            basis = seasonal_resolved

        if len(basis) >= 5:
            lo, hi = _bootstrap_ci(basis, block_len=horizon)
            ci_lo[i], ci_hi[i] = lo, hi

    return pd.DataFrame(
        {
            "expected_return": exp_ret,
            "prob_positive": prob_pos,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "n_samples": n_samples,
            "sufficient_sample": n_samples >= MIN_SAMPLES,
            "seasonal_conditioned": seasonal_conditioned,
        },
        index=features.index,
    )