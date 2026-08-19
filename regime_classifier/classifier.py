"""
Regime Classifier - Phase 15.

Two independent regime dimensions, both computable from existing data:
  - volatility regime: current EWMA vol relative to its own trailing
    distribution - high/normal/low
  - trend regime: Ljung-Box test on trailing daily returns - trending,
    mean_reverting, or neutral

KNOWN CHARACTERISTIC: because both classifications judge "today" relative
to a trailing window of recent history, they detect regime SHIFTS well
(validated: 100% correct right at an injected transition, still ~100%
correct 100 days later) but "acclimate" once a regime has been sustained
long enough to dominate the trailing window itself (~250+ days) - at that
point it can no longer distinguish "objectively high volatility" from
"the same volatility as the past year." A real limitation of a purely
relative classifier, not a bug.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

VOL_LOOKBACK = 250
VOL_HIGH_PCTILE = 66
VOL_LOW_PCTILE = 33
TREND_WINDOW = 60
LJUNGBOX_LAGS = 5
LJUNGBOX_ALPHA = 0.05


def classify_volatility_regime(ewma_vol: pd.Series, lookback: int = VOL_LOOKBACK) -> pd.Series:
    values = ewma_vol.to_numpy()
    n = len(values)
    out = [None] * n
    for i in range(n):
        lo = max(0, i - lookback + 1)
        window = values[lo:i + 1]
        window = window[~np.isnan(window)]
        if len(window) < 30 or np.isnan(values[i]):
            continue
        hi_thresh = np.percentile(window, VOL_HIGH_PCTILE)
        lo_thresh = np.percentile(window, VOL_LOW_PCTILE)
        if values[i] >= hi_thresh:
            out[i] = "high_vol"
        elif values[i] <= lo_thresh:
            out[i] = "low_vol"
        else:
            out[i] = "normal_vol"
    return pd.Series(out, index=ewma_vol.index, dtype=object)


def classify_trend_regime(daily_returns: pd.Series, window: int = TREND_WINDOW) -> pd.Series:
    values = daily_returns.to_numpy()
    n = len(values)
    out = [None] * n
    for i in range(n):
        lo = max(0, i - window + 1)
        w = values[lo:i + 1]
        w = w[~np.isnan(w)]
        if len(w) < window:
            continue
        try:
            result = acorr_ljungbox(w, lags=[LJUNGBOX_LAGS], return_df=True)
            pval = result["lb_pvalue"].iloc[0]
            if pval < LJUNGBOX_ALPHA:
                ac1 = pd.Series(w).autocorr(lag=1)
                out[i] = "trending" if ac1 > 0 else "mean_reverting"
            else:
                out[i] = "neutral"
        except Exception:
            pass
    return pd.Series(out, index=daily_returns.index, dtype=object)


def classify_regime(close: pd.Series, ewma_vol: pd.Series) -> pd.DataFrame:
    daily_ret = np.log(close / close.shift(1))
    vol_regime = classify_volatility_regime(ewma_vol)
    trend_regime = classify_trend_regime(daily_ret)
    return pd.DataFrame({"vol_regime": vol_regime, "trend_regime": trend_regime}, index=close.index)