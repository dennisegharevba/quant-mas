"""
Model 1 - Statistical Baseline - Phase 2.

The "null hypothesis" model: the simplest statistically defensible forecast,
using only the trailing empirical distribution of ALREADY-RESOLVED forward
returns. Every later model must beat this out of sample, after costs, or it
does not get added to the system.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZONS = [1, 3, 5, 10]
DEFAULT_LOOKBACK = 250
MIN_SAMPLES = 60
N_BOOTSTRAP = 2000


@dataclass
class BaselineForecast:
    date: pd.Timestamp
    horizon: int
    expected_return: float
    prob_positive: float
    ci_low: float
    ci_high: float
    n_samples: int
    sufficient_sample: bool


_RNG = np.random.default_rng(0)


def _bootstrap_ci(samples: np.ndarray, n_boot: int = N_BOOTSTRAP, alpha: float = 0.10) -> tuple[float, float]:
    n = len(samples)
    if n < 5:
        return (np.nan, np.nan)
    idx = _RNG.integers(0, n, size=(n_boot, n))
    boot_means = samples[idx].mean(axis=1)
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return (lo, hi)


def forecast_series(features: pd.DataFrame, horizon: int,
                     lookback: int = DEFAULT_LOOKBACK) -> pd.DataFrame:
    col = f"feature_resolved_forward_return_{horizon}d"
    if col not in features.columns:
        raise KeyError(f"{col} not found - did you run build_features first?")

    resolved = features[col]
    n = len(resolved)

    exp_ret = np.full(n, np.nan)
    prob_pos = np.full(n, np.nan)
    ci_lo = np.full(n, np.nan)
    ci_hi = np.full(n, np.nan)
    n_samples = np.zeros(n, dtype=int)

    values = resolved.to_numpy()
    for i in range(n):
        window = values[max(0, i - lookback + 1): i + 1]
        window = window[~np.isnan(window)]
        n_samples[i] = len(window)
        if len(window) == 0:
            continue
        exp_ret[i] = window.mean()
        prob_pos[i] = (window > 0).mean()
        if len(window) >= 5:
            lo, hi = _bootstrap_ci(window)
            ci_lo[i], ci_hi[i] = lo, hi

    out = pd.DataFrame(
        {
            "expected_return": exp_ret,
            "prob_positive": prob_pos,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "n_samples": n_samples,
            "sufficient_sample": n_samples >= MIN_SAMPLES,
        },
        index=features.index,
    )
    return out