"""
Phase 10 evaluation - ablation test: does Model 6 (commodity seasonality)
beat Model 1 out-of-sample? Restricted to the commodity asset class.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_engineering.features import build_features, HORIZONS
from models.statistical.baseline import forecast_series as model1_forecast, DEFAULT_LOOKBACK
from models.seasonality.model6 import forecast_series as model6_forecast
from research_ledger.ledger import Experiment, log_experiment
from backtest_engine.splitting import get_test_mask
from config.universe import by_asset_class

TEST_FRACTION = 0.30
EMBARGO_DAYS = max(HORIZONS)

COMMODITY_TICKERS = [inst.ticker.replace("=", "_") for inst in by_asset_class("commodity")]


def evaluate_instrument(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    features = build_features(df)
    test_mask_base, split_idx, embargo_end_idx = get_test_mask(features.index, TEST_FRACTION, EMBARGO_DAYS)

    rows = []
    for h in HORIZONS:
        fc1 = model1_forecast(features, horizon=h)
        fc6 = model6_forecast(features, horizon=h)
        actual = features[f"target_forward_return_{h}d"]

        test_mask = (
            test_mask_base.to_numpy()
            & fc1["sufficient_sample"].to_numpy() & fc6["sufficient_sample"].to_numpy()
            & actual.notna().to_numpy()
        )
        if test_mask.sum() < 10:
            rows.append({
                "ticker": ticker, "horizon": h, "n_test": int(test_mask.sum()),
                "note": "insufficient out-of-sample observations",
                "pct_seasonal_conditioned": np.nan,
                "rmse_model1": np.nan, "rmse_model6": np.nan,
                "sign_agree_model1": np.nan, "sign_agree_model6": np.nan,
                "model6_beats_model1_rmse": np.nan,
            })
            continue

        f1 = fc1.loc[test_mask, "expected_return"]
        f6 = fc6.loc[test_mask, "expected_return"]
        a_ret = actual.loc[test_mask]
        pct_conditioned = float(fc6.loc[test_mask, "seasonal_conditioned"].mean())

        rmse1 = float(np.sqrt(((f1 - a_ret) ** 2).mean()))
        rmse6 = float(np.sqrt(((f6 - a_ret) ** 2).mean()))
        sign1 = float((np.sign(f1) == np.sign(a_ret)).mean())
        sign6 = float((np.sign(f6) == np.sign(a_ret)).mean())

        rows.append({
            "ticker": ticker, "horizon": h, "n_test": int(test_mask.sum()),
            "note": "", "pct_seasonal_conditioned": round(pct_conditioned, 3),
            "rmse_model1": round(rmse1, 5), "rmse_model6": round(rmse6, 5),
            "sign_agree_model1": round(sign1, 3), "sign_agree_model6": round(sign6, 3),
            "model6_beats_model1_rmse": rmse6 < rmse1,
        })

    return pd.DataFrame(rows)


def run_all(processed_dir: Path) -> pd.DataFrame:
    all_results = []
    for ticker in COMMODITY_TICKERS:
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
    results.to_csv(results_dir / "model6_results.csv", index=False)

    valid = results.dropna(subset=["rmse_model1", "rmse_model6"])
    oos_metrics = {}
    if len(valid):
        print("\n--- Ablation summary: Model 6 (commodity seasonality) vs Model 1 ---")
        frac_better = float(valid["model6_beats_model1_rmse"].mean())
        mean_conditioned = float(valid["pct_seasonal_conditioned"].mean())
        mean_sign_gain = float((valid["sign_agree_model6"] - valid["sign_agree_model1"]).mean())
        print(f"Fraction of instrument-horizons where Model 6 beats Model 1 on RMSE: {frac_better:.1%}")
        print(f"Mean fraction of test days where seasonal-conditioning fired: {mean_conditioned:.1%}")
        print(f"Mean change in sign agreement (Model 6 - Model 1): {mean_sign_gain:+.3f}")
        oos_metrics = {
            "fraction_model6_beats_model1_rmse": round(frac_better, 4),
            "mean_pct_seasonal_conditioned": round(mean_conditioned, 4),
            "mean_sign_agreement_gain": round(mean_sign_gain, 4),
        }

    tickers_tested = sorted(results["ticker"].unique().tolist()) if len(results) else []
    exp = Experiment(
        model_name="Model 6 - Seasonality (commodity-specific; vs Model 1)",
        script="backtest_engine/evaluate_model6.py",
        universe=tickers_tested,
        horizons=HORIZONS,
        test_fraction=TEST_FRACTION,
        lookback_days=DEFAULT_LOOKBACK,
        hyperparameters={"min_month_samples": 20, "embargo_days": EMBARGO_DAYS, "scope": "commodity only"},
        n_combinations_tested=len(tickers_tested) * len(HORIZONS),
        oos_metrics=oos_metrics,
        costs_included=False,
        notes=("Asset-class-specific model - evaluated on commodities only. Validated with "
               "synthetic data before running on real data: correctly detects genuine "
               "origination-month seasonal effects and does not fabricate seasonality on "
               "pure noise."),
        decision="",
        influenced_later_decisions="",
    )
    exp_id = log_experiment(exp)
    print(f"\nLogged to research ledger as experiment #{exp_id}.")
    print("NOTE: decision field left blank - review the summary above and update the ledger row manually.")