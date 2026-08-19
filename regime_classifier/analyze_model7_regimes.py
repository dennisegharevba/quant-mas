"""
Phase 15 analysis: does Model 7's (equity momentum) edge concentrate in
a particular regime, the same way it concentrated in particular sectors?

Uses PER-DAY win/loss (was Model 7's error smaller than Model 1's error
on THIS SPECIFIC day) so each day can be tagged with its own regime.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_engineering.features import build_features, HORIZONS
from models.statistical.baseline import forecast_series as model1_forecast
from models.equity_momentum.model7 import forecast_series as model7_forecast
from models.risk.model5 import compute_ewma_vol
from regime_classifier.classifier import classify_regime
from backtest_engine.splitting import get_test_mask
from config.universe import by_asset_class

TEST_FRACTION = 0.30
EMBARGO_DAYS = max(HORIZONS)
ANALYSIS_HORIZON = 5

STOCK_TICKERS = [inst.ticker.replace("=", "_") for inst in by_asset_class("stock")]


def analyze_instrument(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    features = build_features(df)
    close = df["Close"].astype(float)
    ewma_vol = compute_ewma_vol(np.log(close / close.shift(1)))
    regimes = classify_regime(close, ewma_vol)

    fc1 = model1_forecast(features, horizon=ANALYSIS_HORIZON)
    fc7 = model7_forecast(features, close, horizon=ANALYSIS_HORIZON)
    actual = features[f"target_forward_return_{ANALYSIS_HORIZON}d"]

    test_mask_base, split_idx, embargo_end_idx = get_test_mask(features.index, TEST_FRACTION, EMBARGO_DAYS)
    test_mask = (
        test_mask_base.to_numpy()
        & fc1["sufficient_sample"].to_numpy() & fc7["sufficient_sample"].to_numpy()
        & actual.notna().to_numpy()
    )

    if test_mask.sum() < 10:
        return pd.DataFrame()

    idx = features.index[test_mask]
    f1 = fc1.loc[test_mask, "expected_return"].to_numpy()
    f7 = fc7.loc[test_mask, "expected_return"].to_numpy()
    a = actual.loc[test_mask].to_numpy()

    sq_err_1 = (f1 - a) ** 2
    sq_err_7 = (f7 - a) ** 2
    model7_wins_today = sq_err_7 < sq_err_1

    vol_regime = regimes.loc[idx, "vol_regime"].to_numpy()
    trend_regime = regimes.loc[idx, "trend_regime"].to_numpy()

    return pd.DataFrame({
        "ticker": ticker, "date": idx, "vol_regime": vol_regime, "trend_regime": trend_regime,
        "model7_wins_today": model7_wins_today,
    })


def run_all(processed_dir: Path) -> pd.DataFrame:
    all_results = []
    for ticker in STOCK_TICKERS:
        csv_path = processed_dir / f"{ticker}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        try:
            res = analyze_instrument(df, ticker)
            if len(res):
                all_results.append(res)
        except Exception as e:
            print(f"  [skip] {ticker}: {e}")
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


if __name__ == "__main__":
    processed = Path(__file__).resolve().parents[1] / "data" / "processed"
    results = run_all(processed)

    if len(results) == 0:
        print("No results.")
        sys.exit(0)

    results = results.dropna(subset=["vol_regime", "trend_regime"])

    print(f"\n{'='*70}")
    print("Model 7 win rate by VOLATILITY regime (all stocks pooled, day-level)")
    print(f"{'='*70}")
    vol_summary = results.groupby("vol_regime")["model7_wins_today"].agg(["mean", "count"])
    vol_summary.columns = ["win_rate", "n_days"]
    print(vol_summary.round(3))

    print(f"\n{'='*70}")
    print("Model 7 win rate by TREND regime (all stocks pooled, day-level)")
    print(f"{'='*70}")
    trend_summary = results.groupby("trend_regime")["model7_wins_today"].agg(["mean", "count"])
    trend_summary.columns = ["win_rate", "n_days"]
    print(trend_summary.round(3))

    print(f"\n{'='*70}")
    print("Model 7 win rate by ticker x volatility regime")
    print(f"{'='*70}")
    cross = results.groupby(["ticker", "vol_regime"])["model7_wins_today"].agg(["mean", "count"])
    cross.columns = ["win_rate", "n_days"]
    print(cross.round(3))