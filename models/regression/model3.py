"""
Model 3 - Regression / Factor Model - Phase 4.

Factors: SPY (equity market risk), GC_F (Gold, safe-haven), CL_F (Crude
oil, commodity/inflation) - all already in the universe, avoiding a new
data source before this component earns its place. An asset is never
regressed on itself if it happens to be one of the three factor tickers.

Point-in-time discipline: rolling OLS coefficients at time t are fit
using data through t-1 only (window ending at t, then shifted forward by
1 day before being applied to day t's factor realization).

Model 3's forecast strips out the factor-explained component and applies
the same "trailing mean of already-resolved outcomes" logic as Model 1,
but to the IDIOSYNCRATIC RESIDUAL rather than the raw return.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tools import add_constant

HORIZONS = [1, 3, 5, 10]
DEFAULT_LOOKBACK = 250
MIN_SAMPLES = 60
REGRESSION_WINDOW = 120
FACTOR_TICKERS = ["SPY", "GC_F", "CL_F"]


def load_factor_returns(processed_dir: Path, factor_tickers: list[str] = FACTOR_TICKERS) -> pd.DataFrame:
    series = {}
    for t in factor_tickers:
        path = processed_dir / f"{t}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        close = df["Close"].astype(float)
        series[t] = np.log(close / close.shift(1))
    if not series:
        raise FileNotFoundError(f"None of {factor_tickers} found in {processed_dir}")
    factors = pd.DataFrame(series).dropna(how="all")
    return factors


def compute_residuals(ticker: str, asset_close: pd.Series, factors: pd.DataFrame,
                       window: int = REGRESSION_WINDOW) -> pd.DataFrame:
    asset_ret = np.log(asset_close / asset_close.shift(1))
    use_factors = [c for c in factors.columns if c != ticker]
    if not use_factors:
        return pd.DataFrame(columns=["residual", "predicted", "r_squared"], index=asset_close.index)

    aligned = pd.concat([asset_ret.rename("asset"), factors[use_factors]], axis=1, join="inner").dropna()
    if len(aligned) < window + 5:
        return pd.DataFrame(columns=["residual", "predicted", "r_squared"], index=asset_close.index)

    X = add_constant(aligned[use_factors])
    y = aligned["asset"]

    model = RollingOLS(y, X, window=window, min_nobs=window, missing="drop")
    fit = model.fit()

    params = fit.params.shift(1)
    r_squared = fit.rsquared.shift(1)

    predicted = (params * X).sum(axis=1)
    residual = y - predicted

    out = pd.DataFrame({"residual": residual, "predicted": predicted, "r_squared": r_squared})
    out = out.reindex(asset_close.index)
    return out


def build_residual_features(ticker: str, df: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].astype(float)
    resid_df = compute_residuals(ticker, close, factors)
    residual = resid_df["residual"]

    out = pd.DataFrame(index=df.index)
    out["residual_1d"] = residual
    out["r_squared"] = resid_df["r_squared"]

    for h in HORIZONS:
        fwd_sum = residual.shift(-1).rolling(h).sum().shift(-(h - 1))
        out[f"target_forward_residual_{h}d"] = fwd_sum
        out[f"feature_resolved_forward_residual_{h}d"] = fwd_sum.shift(h)

    return out


def forecast_series(residual_features: pd.DataFrame, horizon: int,
                     lookback: int = DEFAULT_LOOKBACK) -> pd.DataFrame:
    col = f"feature_resolved_forward_residual_{horizon}d"
    if col not in residual_features.columns:
        raise KeyError(f"{col} not found - did you run build_residual_features first?")

    resolved = residual_features[col].to_numpy()
    n = len(resolved)

    exp_ret = np.full(n, np.nan)
    prob_pos = np.full(n, np.nan)
    n_samples = np.zeros(n, dtype=int)

    for i in range(n):
        window = resolved[max(0, i - lookback + 1): i + 1]
        window = window[~np.isnan(window)]
        n_samples[i] = len(window)
        if len(window) == 0:
            continue
        exp_ret[i] = window.mean()
        prob_pos[i] = (window > 0).mean()

    return pd.DataFrame(
        {
            "expected_return": exp_ret,
            "prob_positive": prob_pos,
            "n_samples": n_samples,
            "sufficient_sample": n_samples >= MIN_SAMPLES,
        },
        index=residual_features.index,
    )