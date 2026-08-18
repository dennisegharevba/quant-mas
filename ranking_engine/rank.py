"""
Ranking Engine - Phase 7.

Combines Model 1 (the only return-estimate remaining - not because it
passed its own ablation test, but because nothing tested against it beat
it) with Model 5 EWMA volatility (accepted) into a per-asset composite
score, ranked cross-sectionally across the universe on a given day.

Model 1 did NOT clear its own bar against a naive-zero forecast in
backtesting (~20-27% RMSE win rate). It remains the return estimate here
only because Models 2-4 were tested against it and none demonstrated a
genuine improvement. This means the NO-TRADE filter is expected to reject
most candidates most of the time right now - correct behavior, not a bug.

Composite weighting: equal-weighted z-score combination of EV and
risk-adjusted return - a stated placeholder, not the regression-calibrated
weighting the blueprint calls for (no validated signal exists yet to
calibrate those weights against).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_engineering.features import build_features
from models.statistical.baseline import forecast_series as model1_forecast
from models.risk.model5 import compute_ewma_vol


def build_scan_row(ticker: str, df: pd.DataFrame, horizon: int) -> dict:
    features = build_features(df)
    fc1 = model1_forecast(features, horizon=horizon)

    close = df["Close"].astype(float)
    daily_ret = np.log(close / close.shift(1))
    ewma_vol = compute_ewma_vol(daily_ret)

    valid_idx = fc1.index[fc1["sufficient_sample"] & ewma_vol.notna()]
    if len(valid_idx) == 0:
        return {
            "ticker": ticker, "horizon": horizon, "date": None,
            "expected_return": np.nan, "prob_positive": np.nan,
            "ci_low": np.nan, "ci_high": np.nan, "n_samples": 0,
            "ewma_vol": np.nan, "risk_adjusted_score": np.nan,
            "ci_excludes_zero": False,
        }

    latest = valid_idx[-1]
    row = fc1.loc[latest]
    vol = float(ewma_vol.loc[latest])
    exp_ret = float(row["expected_return"])
    risk_adj = exp_ret / vol if vol > 0 else np.nan
    ci_excludes_zero = bool(row["ci_low"] > 0 or row["ci_high"] < 0)

    return {
        "ticker": ticker, "horizon": horizon, "date": str(latest.date()),
        "expected_return": exp_ret, "prob_positive": float(row["prob_positive"]),
        "ci_low": float(row["ci_low"]), "ci_high": float(row["ci_high"]),
        "n_samples": int(row["n_samples"]), "ewma_vol": vol,
        "risk_adjusted_score": risk_adj, "ci_excludes_zero": ci_excludes_zero,
    }


def rank_universe(processed_dir: Path, horizon: int) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(processed_dir.glob("*.csv")):
        ticker = csv_path.stem
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        try:
            rows.append(build_scan_row(ticker, df, horizon))
        except Exception as e:
            print(f"  [skip] {ticker}: {e}")

    scan = pd.DataFrame(rows)
    if len(scan) == 0:
        return scan

    valid = scan["expected_return"].notna() & scan["risk_adjusted_score"].notna()

    for col, z_col in [("expected_return", "ev_zscore"), ("risk_adjusted_score", "risk_adj_zscore")]:
        scan[z_col] = np.nan
        vals = scan.loc[valid, col]
        if vals.std() > 0:
            scan.loc[valid, z_col] = (vals - vals.mean()) / vals.std()

    scan["composite_score"] = np.nan
    scan.loc[valid, "composite_score"] = 0.5 * scan.loc[valid, "ev_zscore"] + 0.5 * scan.loc[valid, "risk_adj_zscore"]

    scan = scan.sort_values("composite_score", ascending=False, na_position="last").reset_index(drop=True)
    scan["rank"] = np.where(scan["composite_score"].notna(), scan.index + 1, np.nan)
    return scan