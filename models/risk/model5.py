"""
Model 5 - Volatility / Risk Model - Phase 6.

Tests whether EWMA volatility clustering-aware weighting beats a naive
flat rolling-window estimate at predicting forward realized volatility.
Also provides downside deviation and max drawdown utilities (treated as
a FLOOR on future drawdown risk, never a ceiling).

EWMA formula (RiskMetrics-style, lambda=0.94 for daily data):
sigma_t^2 = lambda * sigma_{t-1}^2 + (1-lambda) * r_t^2
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = [3, 5, 10]  # h=1 excluded: a 1-day forward "volatility" (std of a
                         # single return) is mathematically undefined
EWMA_LAMBDA = 0.94
NAIVE_WINDOW = 20
DD_WINDOW = 250


def compute_ewma_vol(daily_returns: pd.Series, lam: float = EWMA_LAMBDA) -> pd.Series:
    r2 = daily_returns.fillna(0.0) ** 2
    ewma_var = r2.ewm(alpha=(1 - lam), adjust=False).mean()
    ewma_var[daily_returns.isna()] = np.nan
    return np.sqrt(ewma_var) * np.sqrt(252)


def compute_naive_vol(daily_returns: pd.Series, window: int = NAIVE_WINDOW) -> pd.Series:
    return daily_returns.rolling(window, min_periods=window).std() * np.sqrt(252)


def compute_downside_deviation(daily_returns: pd.Series, window: int = DD_WINDOW) -> pd.Series:
    downside_sq = daily_returns.clip(upper=0) ** 2
    dd = downside_sq.rolling(window, min_periods=window).mean().pow(0.5)
    return dd * np.sqrt(252)


def compute_max_drawdown(close: pd.Series, window: int = DD_WINDOW) -> pd.Series:
    def _mdd(x: np.ndarray) -> float:
        cummax = np.maximum.accumulate(x)
        drawdown = (x - cummax) / cummax
        return float(drawdown.min())
    return close.rolling(window, min_periods=window).apply(_mdd, raw=True)


def build_forward_realized_vol(daily_returns: pd.Series, horizons: list[int] = HORIZONS) -> pd.DataFrame:
    out = pd.DataFrame(index=daily_returns.index)
    for h in horizons:
        fwd_std = daily_returns.shift(-1).rolling(h).std().shift(-(h - 1))
        out[f"target_forward_vol_{h}d"] = fwd_std * np.sqrt(252)
    return out