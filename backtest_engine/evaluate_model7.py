"""
Phase 12 evaluation - ablation test: does Model 7 (equity 12-1 month
momentum) beat Model 1 out-of-sample? Restricted to stocks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_engineering.features import build_features, HORIZONS
from models.statistical.baseline import forecast_series as model1_forecast, DEFAULT_LOOKBACK
from models.equity_momentum.model7 import forecast_series as model7_forecast
from research_ledger.ledger import Experiment, log_experiment
from backtest_engine.splitting import get_test_mask
from config.universe import by_asset_class

TEST_FRACTION = 0.30
EMBARGO_DAYS = max(HORIZONS)

STOCK_TICKERS = [inst.ticker.replace("=", "_") for inst in by_asset_class("stock")]


def evaluate_instrument(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    features = build_features(df)
    close = df["Close"].astype(float)
    test_mask_base, split_idx, embargo_end_idx = get_test_mask(features.index, TEST_FRACTION, EMBARGO_DAYS)

    rows = []
    for h in HORIZONS:
        fc1 = model1_forecast(features, horizon=h)
        fc7 = model7_forecast(features, close, horizon=h)
        actual = features[f"target_forward_return_{h}d"]

        test_mask = (
            test_mask_base.to_numpy()
            & fc1["sufficient_sample"].to_numpy() & fc7["sufficient_sample"].to_numpy()
            & actual.notna().to_numpy()
        )
        if test_mask.sum() < 10:
            rows.append({
                "ticker": ticker, "horizon": h, "n_test": int(test_mask.sum()),
                "note": "insufficient out-of-sample observations",
                "pct_momentum_conditioned": np.nan,
                "rmse_model1": np.nan, "rmse_model7": np.nan,
                "sign_agree_model1": np.nan, "sign_agree_model7": np.nan,
                "model7_beats_model1_rmse": np.nan,
            })
            continue

        f1 = fc1.loc[test_mask, "expected_return"]
        f7 = fc7.loc[test_mask, "expected_return"]
        a_ret = actual.loc[test_mask]
        pct_conditioned = float(fc7.loc[test_mask, "momentum_conditioned"].mean())

        rmse1 = float(np.sqrt(((f1 - a_ret) ** 2).mean()))
        rmse7 = float(np.sqrt(((f7 - a_ret) ** 2).mean()))
        sign1 = float((np.sign(f1) == np.sign(a_ret)).mean())
        sign7 = float((np.sign(f7) == np.sign(a_ret)).mean())

        rows.append({
            "ticker": ticker, "horizon": h, "n_test": int(test_mask.sum()),
            "note": "", "pct_momentum_conditioned": round(pct_conditioned, 3),
            "rmse_model1": round(rmse1, 5), "rmse_model7": round(rmse7, 5),
            "sign_agree_model1": round(sign1, 3), "sign_agree_model7": round(sign7, 3),
            "model7_beats_model1_rmse": rmse7 < rmse1,
        })

    return pd.DataFrame(rows)


def run_all(processed_dir: Path) -> pd.DataFrame:
    all_results = []
    for ticker in STOCK_TICKERS:
        csv_path = processed_dir / f"{ticker}.csv"
        if not csv_path.exists():
            print(f"  [skip] {ticker}: not found in {processed_dir}")
            continue
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        try:
            res = evaluate_instrument(df, ticker)
            all_results.append(res)
        except Exception as e:
            print(f"  [skip] {ticker}: {e}")
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


if __name__ == "__main__":
    processed = Path(__file__).resolve().parents[1] / "data" / "processed"
    results = run_all(processed)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 200)
    print(results.to_string(index=False))

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    results.to_csv(results_dir / "model7_results.csv", index=False)

    valid = results.dropna(subset=["rmse_model1", "rmse_model7"])
    oos_metrics = {}
    if len(valid):
        print("\n--- Ablation summary: Model 7 (equity momentum) vs Model 1 ---")
        frac_better = float(valid["model7_beats_model1_rmse"].mean())
        mean_conditioned = float(valid["pct_momentum_conditioned"].mean())
        mean_sign_gain = float((valid["sign_agree_model7"] - valid["sign_agree_model1"]).mean())
        print(f"Fraction of instrument-horizons where Model 7 beats Model 1 on RMSE: {frac_better:.1%}")
        print(f"Mean fraction of test days where momentum-conditioning fired: {mean_conditioned:.1%}")
        print(f"Mean change in sign agreement (Model 7 - Model 1): {mean_sign_gain:+.3f}")
        oos_metrics = {
            "fraction_model7_beats_model1_rmse": round(frac_better, 4),
            "mean_pct_momentum_conditioned": round(mean_conditioned, 4),
            "mean_sign_agreement_gain": round(mean_sign_gain, 4),
        }

    tickers_tested = sorted(results["ticker"].unique().tolist()) if len(results) else []
    exp = Experiment(
        model_name="Model 7 - Equity Momentum (12-1 month; vs Model 1)",
        script="backtest_engine/evaluate_model7.py",
        universe=tickers_tested,
        horizons=HORIZONS,
        test_fraction=TEST_FRACTION,
        lookback_days=DEFAULT_LOOKBACK,
        hyperparameters={"momentum_long_window": 252, "momentum_skip_window": 21,
                          "min_regime_samples": 30, "embargo_days": EMBARGO_DAYS, "scope": "stock only"},
        n_combinations_tested=len(tickers_tested) * len(HORIZONS),
        oos_metrics=oos_metrics,
        costs_included=False,
        notes=("Asset-class-specific model - equities only. Uses an expanding window for "
               "momentum-sign conditioning, not a fixed trailing lookback - a fixed 250-day "
               "window was tested and found to be a silent no-op (momentum barely changes "
               "sign within 250 days), caught before real-data testing. Multi-seed testing "
               "(n=25+15) confirmed no systematic fabricated edge on pure noise."),
        decision="",
        influenced_later_decisions="",
    )
    exp_id = log_experiment(exp)
    print(f"\nLogged to research ledger as experiment #{exp_id}.")
    print("NOTE: decision field left blank - review the summary above and update the ledger row manually.")