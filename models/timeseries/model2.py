"""
Model 2 - Time-Series - Phase 3.

Design: conditions Model 1's forecast on the CURRENT momentum regime (sign
of the most recent 1-day return), using only historical instances that
were in the same regime. Regime-conditioning is only applied when a
Ljung-Box test finds statistically significant autocorrelation in the
trailing window - otherwise Model 2 falls back to Model 1's unconditional
forecast exactly, so it can never do worse than Model 1 by construction
when there's no evidence of momentum/reversion.

Conditioning variable is the most recent 1-day return sign, not a smoothed
trailing average - this matches the short lags the Ljung-Box test operates
on. (An earlier version conditioned on a 20-day trailing mean; synthetic
validation with a known injected 1-day AR(1) signal showed it failed to
detect that signal, exactly because of this mismatch - caught before
reaching real data.)

All conditioning uses only point-in-time-safe data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

HORIZONS = [1, 3, 5, 10]
DEFAULT_LOOKBACK = 250
MIN_SAMPLES = 60
MIN_REGIME_SAMPLES = 30
LJUNGBOX_LAGS = 5
LJUNGBOX_ALPHA = 0.05


def _ljungbox_significant(returns: np.ndarray, lags: int = LJUNGBOX_LAGS,
                            alpha: float = LJUNGBOX_ALPHA) -> bool:
    clean = returns[~np.isnan(returns)]
    if len(clean) < lags + 10:
        return False
    try:
        result = acorr_ljungbox(clean, lags=[lags], return_df=True)
        p_value = result["lb_pvalue"].iloc[0]
        return bool(p_value < alpha)
    except Exception:
        return False


def forecast_series(features: pd.DataFrame, horizon: int,
                     lookback: int = DEFAULT_LOOKBACK) -> pd.DataFrame:
    resolved_col = f"feature_resolved_forward_return_{horizon}d"
    return_col = "feature_log_return_1d"
    for col in (resolved_col, return_col):
        if col not in features.columns:
            raise KeyError(f"{col} not found - did you run build_features first?")

    resolved = features[resolved_col].to_numpy()
    hist_momentum = features[return_col].shift(horizon).to_numpy()
    current_momentum = features[return_col].to_numpy()
    raw_returns = features[return_col].to_numpy()

    n = len(features)
    exp_ret = np.full(n, np.nan)
    prob_pos = np.full(n, np.nan)
    n_samples = np.zeros(n, dtype=int)
    regime_conditioned = np.zeros(n, dtype=bool)
    autocorr_significant = np.zeros(n, dtype=bool)

    for i in range(n):
        lo = max(0, i - lookback + 1)
        window_resolved = resolved[lo: i + 1]
        window_hist_momentum = hist_momentum[lo: i + 1]
        window_raw_returns = raw_returns[lo: i + 1]

        valid = ~np.isnan(window_resolved)
        window_resolved = window_resolved[valid]
        window_hist_momentum = window_hist_momentum[valid]
        n_samples[i] = len(window_resolved)
        if len(window_resolved) == 0:
            continue

        exp_ret[i] = window_resolved.mean()
        prob_pos[i] = (window_resolved > 0).mean()

        sig = _ljungbox_significant(window_raw_returns)
        autocorr_significant[i] = sig
        if not sig or np.isnan(current_momentum[i]):
            continue

        same_regime = (np.sign(window_hist_momentum) == np.sign(current_momentum[i])) & ~np.isnan(window_hist_momentum)
        if same_regime.sum() < MIN_REGIME_SAMPLES:
            continue

        regime_resolved = window_resolved[same_regime]
        exp_ret[i] = regime_resolved.mean()
        prob_pos[i] = (regime_resolved > 0).mean()
        regime_conditioned[i] = True

    return pd.DataFrame(
        {
            "expected_return": exp_ret,
            "prob_positive": prob_pos,
            "n_samples": n_samples,
            "sufficient_sample": n_samples >= MIN_SAMPLES,
            "regime_conditioned": regime_conditioned,
            "autocorr_significant": autocorr_significant,
        },
        index=features.index,
    )