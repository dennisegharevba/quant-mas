"""
Corwin-Schultz Spread Estimator - Phase 17.

Estimates the bid-ask spread purely from OHLC data (Corwin & Schultz,
2012), replacing the documented-typical cost placeholder with an actual
measured estimate per instrument.

CAVEAT: developed and validated on equities. Applying it to FX,
commodities, and crypto is a reasonable extension but not validated by
the original paper for those classes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

K = 3 - 2 * np.sqrt(2)
ROLLING_WINDOW = 20


def _corwin_schultz_pair_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    log_hl = np.log(high / low)
    beta = log_hl ** 2 + log_hl.shift(-1) ** 2

    high2 = pd.concat([high, high.shift(-1)], axis=1).max(axis=1)
    low2 = pd.concat([low, low.shift(-1)], axis=1).min(axis=1)
    gamma = np.log(high2 / low2) ** 2

    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / K - np.sqrt(gamma / K)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

    return spread.shift(1)


def estimate_spread_bps(high: pd.Series, low: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """
    IMPORTANT: average raw pair estimates FIRST, then floor the smoothed
    average at 0 - not the reverse. Flooring each noisy pair estimate
    before averaging introduces a real upward bias (verified: on a pure
    random walk with zero true spread, floor-then-average inflated the
    estimate to ~43bps vs the correct ~4-10bps from average-then-floor).
    """
    pair_spread = _corwin_schultz_pair_spread(high, low)
    rolling_raw = pair_spread.rolling(window, min_periods=max(5, window // 2)).mean()
    rolling_floored = rolling_raw.clip(lower=0)
    return rolling_floored * 10000


def latest_spread_estimate_bps(high: pd.Series, low: pd.Series, window: int = ROLLING_WINDOW) -> float | None:
    series = estimate_spread_bps(high, low, window)
    valid = series.dropna()
    if len(valid) == 0:
        return None
    return float(valid.iloc[-1])