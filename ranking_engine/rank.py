"""
Ranking Engine - Phase 7, revised per the architecture correction and
Phase 10's seasonality finding.

Asset-class-appropriate model selection:
  - index -> Model 1, crypto/fx -> Model 3, commodity/stock -> Model 1
    (see stratified evidence in prior commits for rationale)

Ticker-level override: CL_F (crude oil) -> Model 6 (seasonality),
justified by experiment #12 - Model 6 beat Model 1 at all 4 horizons for
CL_F specifically, with a plausible economic rationale (documented
calendar-driven demand: driving season, heating season, refinery
maintenance) distinct from gold/silver/copper which did NOT show the
effect. This is a single-instrument finding, not generalized to the
commodity class as a whole - hence a ticker override, not a class default
change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_engineering.features import build_features
from models.statistical.baseline import forecast_series as model1_forecast, DEFAULT_LOOKBACK
from models.regression.model3 import load_factor_returns, build_residual_features, forecast_series as model3_forecast
from models.seasonality.model6 import forecast_series as model6_forecast
from models.risk.model5 import compute_ewma_vol
from position_sizer.sizer import get_win_loss_magnitudes
from config.universe import get as get_instrument

ASSET_CLASS_MODEL_MAP = {
    "index": "model1",
    "crypto": "model3",
    "fx": "model3",
    "commodity": "model1",
    "stock": "model1",
}

TICKER_MODEL_OVERRIDE = {
    "CL_F": "model6",
}


def _ticker_to_universe_ticker(file_ticker: str) -> str:
    for suffix in ("_X",):
        if file_ticker.endswith(suffix):
            return file_ticker[: -len(suffix)] + "=X"
    if file_ticker in ("GC_F", "SI_F", "CL_F", "HG_F"):
        return file_ticker.replace("_F", "=F")
    return file_ticker


def build_scan_row_model1(ticker: str, df: pd.DataFrame, horizon: int, asset_class: str = 'unknown') -> dict:
    features = build_features(df)
    fc1 = model1_forecast(features, horizon=horizon)

    close = df["Close"].astype(float)
    daily_ret = np.log(close / close.shift(1))
    ewma_vol = compute_ewma_vol(daily_ret)

    valid_idx = fc1.index[fc1["sufficient_sample"] & ewma_vol.notna()]
    if len(valid_idx) == 0:
        return _empty_row(ticker, horizon, "model1", asset_class)

    latest = valid_idx[-1]
    row = fc1.loc[latest]
    vol = float(ewma_vol.loc[latest])
    exp_ret = float(row["expected_return"])
    risk_adj = exp_ret / vol if vol > 0 else np.nan
    ci_excludes_zero = bool(row["ci_low"] > 0 or row["ci_high"] < 0)

    resolved_col = f"feature_resolved_forward_return_{horizon}d"
    loc = features.index.get_loc(latest)
    lo = max(0, loc - DEFAULT_LOOKBACK + 1)
    window = features[resolved_col].iloc[lo: loc + 1].to_numpy()
    mean_win, mean_loss = get_win_loss_magnitudes(window)

    return {
        "ticker": ticker, "horizon": horizon, "date": str(latest.date()), "model_used": "model1",
        "asset_class": asset_class,
        "expected_return": exp_ret, "prob_positive": float(row["prob_positive"]),
        "ci_low": float(row["ci_low"]), "ci_high": float(row["ci_high"]),
        "n_samples": int(row["n_samples"]), "ewma_vol": vol,
        "risk_adjusted_score": risk_adj, "ci_excludes_zero": ci_excludes_zero,
        "mean_win": mean_win, "mean_loss": mean_loss,
    }


def build_scan_row_model3(ticker: str, df: pd.DataFrame, horizon: int, factors: pd.DataFrame, asset_class: str = 'unknown') -> dict:
    universe_ticker = _ticker_to_universe_ticker(ticker)
    resid_features = build_residual_features(universe_ticker, df, factors)
    fc3 = model3_forecast(resid_features, horizon=horizon)

    close = df["Close"].astype(float)
    daily_ret = np.log(close / close.shift(1))
    ewma_vol = compute_ewma_vol(daily_ret)

    valid_idx = fc3.index[fc3["sufficient_sample"] & ewma_vol.notna()]
    if len(valid_idx) == 0:
        return _empty_row(ticker, horizon, "model3", asset_class)

    latest = valid_idx[-1]
    row = fc3.loc[latest]
    vol = float(ewma_vol.loc[latest])
    exp_ret = float(row["expected_return"])
    risk_adj = exp_ret / vol if vol > 0 else np.nan
    ci_excludes_zero = bool(row["ci_low"] > 0 or row["ci_high"] < 0)

    resolved_col = f"feature_resolved_forward_residual_{horizon}d"
    loc = resid_features.index.get_loc(latest)
    lo = max(0, loc - DEFAULT_LOOKBACK + 1)
    window = resid_features[resolved_col].iloc[lo: loc + 1].to_numpy()
    mean_win, mean_loss = get_win_loss_magnitudes(window)

    return {
        "ticker": ticker, "horizon": horizon, "date": str(latest.date()), "model_used": "model3",
        "asset_class": asset_class,
        "expected_return": exp_ret, "prob_positive": float(row["prob_positive"]),
        "ci_low": float(row["ci_low"]), "ci_high": float(row["ci_high"]),
        "n_samples": int(row["n_samples"]), "ewma_vol": vol,
        "risk_adjusted_score": risk_adj, "ci_excludes_zero": ci_excludes_zero,
        "mean_win": mean_win, "mean_loss": mean_loss,
    }


def build_scan_row_model6(ticker: str, df: pd.DataFrame, horizon: int, asset_class: str = 'unknown') -> dict:
    features = build_features(df)
    fc6 = model6_forecast(features, horizon=horizon)

    close = df["Close"].astype(float)
    daily_ret = np.log(close / close.shift(1))
    ewma_vol = compute_ewma_vol(daily_ret)

    valid_idx = fc6.index[fc6["sufficient_sample"] & ewma_vol.notna()]
    if len(valid_idx) == 0:
        return _empty_row(ticker, horizon, "model6", asset_class)

    latest = valid_idx[-1]
    row = fc6.loc[latest]
    vol = float(ewma_vol.loc[latest])
    exp_ret = float(row["expected_return"])
    risk_adj = exp_ret / vol if vol > 0 else np.nan
    ci_excludes_zero = bool(row["ci_low"] > 0 or row["ci_high"] < 0)

    resolved_col = f"feature_resolved_forward_return_{horizon}d"
    loc = features.index.get_loc(latest)
    window = features[resolved_col].iloc[: loc + 1].to_numpy()
    mean_win, mean_loss = get_win_loss_magnitudes(window)

    return {
        "ticker": ticker, "horizon": horizon, "date": str(latest.date()), "model_used": "model6",
        "asset_class": asset_class,
        "expected_return": exp_ret, "prob_positive": float(row["prob_positive"]),
        "ci_low": float(row["ci_low"]), "ci_high": float(row["ci_high"]),
        "n_samples": int(row["n_samples"]), "ewma_vol": vol,
        "risk_adjusted_score": risk_adj, "ci_excludes_zero": ci_excludes_zero,
        "mean_win": mean_win, "mean_loss": mean_loss,
    }


def _empty_row(ticker: str, horizon: int, model_used: str, asset_class: str = "unknown") -> dict:
    return {
        "ticker": ticker, "horizon": horizon, "date": None, "model_used": model_used,
        "asset_class": asset_class,
        "expected_return": np.nan, "prob_positive": np.nan,
        "ci_low": np.nan, "ci_high": np.nan, "n_samples": 0,
        "ewma_vol": np.nan, "risk_adjusted_score": np.nan,
        "ci_excludes_zero": False, "mean_win": np.nan, "mean_loss": np.nan,
    }


def build_scan_row(ticker: str, df: pd.DataFrame, horizon: int, factors: pd.DataFrame) -> dict:
    try:
        inst = get_instrument(_ticker_to_universe_ticker(ticker))
        asset_class = inst.asset_class
    except KeyError:
        asset_class = "stock"

    if ticker in TICKER_MODEL_OVERRIDE:
        selected_model = TICKER_MODEL_OVERRIDE[ticker]
    else:
        selected_model = ASSET_CLASS_MODEL_MAP.get(asset_class, "model1")

    if selected_model == "model6":
        try:
            row = build_scan_row_model6(ticker, df, horizon, asset_class)
            if row["date"] is not None:
                return row
        except Exception:
            pass
        row = build_scan_row_model1(ticker, df, horizon, asset_class)
        row["model_used"] = "model1 (fallback - model6 unavailable)"
        return row

    if selected_model == "model3":
        try:
            row = build_scan_row_model3(ticker, df, horizon, factors, asset_class)
            if row["date"] is not None:
                return row
        except Exception:
            pass
        row = build_scan_row_model1(ticker, df, horizon, asset_class)
        row["model_used"] = "model1 (fallback - model3 unavailable)"
        return row

    return build_scan_row_model1(ticker, df, horizon, asset_class)


def rank_universe(processed_dir: Path, horizon: int) -> pd.DataFrame:
    factors = load_factor_returns(processed_dir)

    rows = []
    for csv_path in sorted(processed_dir.glob("*.csv")):
        ticker = csv_path.stem
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        try:
            rows.append(build_scan_row(ticker, df, horizon, factors))
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