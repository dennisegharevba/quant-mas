"""
Model 7 - Equity Momentum Factor - Phase 12.

Classic Jegadeesh-Titman "12-1 month" momentum - cumulative return from
252 trading days ago to 21 trading days ago (skipping the most recent
month, a separately-documented short-term reversal effect).

Scoped to equities only (AAPL, MSFT, NVDA) per the architecture
correction.

Uses an EXPANDING window over all available history for momentum-sign
conditioning, not a fixed short trailing lookback - momentum is a slow-
moving, 231-day-smoothed measure, so a short window (e.g. 250 days)
rarely contains more than one sign regime, making the conditioning a
silent no-op (confirmed via direct testing before this was fixed). Same
fix pattern as Model 6's seasonality conditioning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = [1, 3, 5, 10]
DEFAULT_LOOKBACK = 250
MIN_SAMPLES = 60
MIN_REGIME_SAMPLES = 30
MOMENTUM_LONG_WINDOW = 252
MOMENTUM_SKIP_WINDOW = 21


def compute_momentum_12_1(close: pd.Series) -> pd.Series:
    return np.log(close.shift(MOMENTUM_SKIP_WINDOW) / close.shift(MOMENTUM_LONG_WINDOW))


def forecast_series(features: pd.DataFrame, close: pd.Series, horizon: int) -> pd.DataFrame:
    from models.statistical.baseline import _bootstrap_ci

    resolved_col = f"feature_resolved_forward_return_{horizon}d"
    if resolved_col not in features.columns:
        raise KeyError(f"{resolved_col} not found - did you run build_features first?")

    resolved = features[resolved_col].to_numpy()
    momentum = compute_momentum_12_1(close).reindex(features.index).to_numpy()
    hist_momentum = pd.Series(momentum, index=features.index).shift(horizon).to_numpy()
    current_momentum = momentum

    n = len(features)
    exp_ret = np.full(n, np.nan)
    prob_pos = np.full(n, np.nan)
    ci_lo = np.full(n, np.nan)
    ci_hi = np.full(n, np.nan)
    n_samples = np.zeros(n, dtype=int)
    momentum_conditioned = np.zeros(n, dtype=bool)

    for i in range(n):
        window_resolved = resolved[: i + 1]
        window_hist_momentum = hist_momentum[: i + 1]

        valid = ~np.isnan(window_resolved)
        window_resolved = window_resolved[valid]
        window_hist_momentum = window_hist_momentum[valid]
        n_samples[i] = len(window_resolved)
        if len(window_resolved) == 0:
            continue

        exp_ret[i] = window_resolved.mean()
        prob_pos[i] = (window_resolved > 0).mean()
        basis = window_resolved

        if not np.isnan(current_momentum[i]):
            same_sign = (np.sign(window_hist_momentum) == np.sign(current_momentum[i])) & ~np.isnan(window_hist_momentum)
            if same_sign.sum() >= MIN_REGIME_SAMPLES:
                regime_resolved = window_resolved[same_sign]
                exp_ret[i] = regime_resolved.mean()
                prob_pos[i] = (regime_resolved > 0).mean()
                momentum_conditioned[i] = True
                basis = regime_resolved

        if len(basis) >= 5:
            lo_ci, hi_ci = _bootstrap_ci(basis, block_len=horizon)
            ci_lo[i], ci_hi[i] = lo_ci, hi_ci

    return pd.DataFrame(
        {
            "expected_return": exp_ret,
            "prob_positive": prob_pos,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "n_samples": n_samples,
            "sufficient_sample": n_samples >= MIN_SAMPLES,
            "momentum_conditioned": momentum_conditioned,
        },
        index=features.index,
    )