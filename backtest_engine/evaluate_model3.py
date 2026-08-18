"""
Phase 4 evaluation - ablation test: does Model 3 (regression/factor) beat
Model 1 (statistical baseline) out-of-sample?
Uses an embargo gap around the train/test boundary (see splitting.py).

Model 3's forecast assumes zero expected factor-driven return (no
directional view on SPY/Gold/Oil themselves) - forecast is purely the
idiosyncratic residual. Mean R2 reported to show how much variance the
factors actually explain per instrument.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_engineering.features import build_features, HORIZONS
from models.statistical.baseline import forecast_series as model1_forecast, DEFAULT_LOOKBACK
from models.regression.model3 import (
    load_factor_returns, build_residual_features, forecast_series as model3_forecast,
    FACTOR_TICKERS, REGRESSION_WINDOW,
)
from research_ledger.ledger import Experiment, log_experiment
from backtest_engine.splitting import get_test_mask

TEST_FRACTION = 0.30
EMBARGO_DAYS = max(HORIZONS)


def evaluate_instrument(ticker: str, df: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    features = build_features(df)
    resid_features = build_residual_features(ticker, df, factors)
    test_mask_base, split_idx, embargo_end_idx = get_test_mask(features.index, TEST_FRACTION, EMBARGO_DAYS)

    mean_r2 = float(resid_features["r_squared"].mean(skipna=True))

    rows = []
    for h in HORIZONS:
        fc1 = model1_forecast(features, horizon=h)
        fc3 = model3_forecast(resid_features, horizon=h)
        actual = features[f"target_forward_return_{h}d"]

        test_mask = (
            test_mask_base.to_numpy()
            & fc1["sufficient_sample"].to_numpy() & fc3["sufficient_sample"].to_numpy()
            & actual.notna().to_numpy()
        )
        if test_mask.sum() < 10:
            rows.append({
                "ticker": ticker, "horizon": h, "n_test": int(test_mask.sum()),
                "note": "insufficient out-of-sample observations", "mean_r2": mean_r2,
                "rmse_model1": np.nan, "rmse_model3": np.nan,
                "sign_agree_model1": np.nan, "sign_agree_model3": np.nan,
                "model3_beats_model1_rmse": np.nan,
            })
            continue

        f1 = fc1.loc[test_mask, "expected_return"]
        f3 = fc3.loc[test_mask, "expected_return"]
        a_ret = actual.loc[test_mask]

        rmse1 = float(np.sqrt(((f1 - a_ret) ** 2).mean()))
        rmse3 = float(np.sqrt(((f3 - a_ret) ** 2).mean()))
        sign1 = float((np.sign(f1) == np.sign(a_ret)).mean())
        sign3 = float((np.sign(f3) == np.sign(a_ret)).mean())

        rows.append({
            "ticker": ticker, "horizon": h, "n_test": int(test_mask.sum()),
            "note": "", "mean_r2": round(mean_r2, 3),
            "rmse_model1": round(rmse1, 5), "rmse_model3": round(rmse3, 5),
            "sign_agree_model1": round(sign1, 3), "sign_agree_model3": round(sign3, 3),
            "model3_beats_model1_rmse": rmse3 < rmse1,
        })

    return pd.DataFrame(rows)


def run_all(processed_dir: Path) -> pd.DataFrame:
    factors = load_factor_returns(processed_dir)
    all_results = []
    for csv_path in sorted(processed_dir.glob("*.csv")):
        ticker = csv_path.stem
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        try:
            res = evaluate_instrument(ticker, df, factors)
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

    valid = results.dropna(subset=["rmse_model1", "rmse_model3"])
    oos_metrics = {}
    if len(valid):
        print("\n--- Ablation summary: Model 3 vs Model 1 ---")
        frac_better = float(valid["model3_beats_model1_rmse"].mean())
        mean_r2_all = float(valid["mean_r2"].mean())
        mean_sign_gain = float((valid["sign_agree_model3"] - valid["sign_agree_model1"]).mean())
        print(f"Fraction of instrument-horizons where Model 3 beats Model 1 on RMSE: {frac_better:.1%}")
        print(f"Mean R2 of the factor regression across instruments: {mean_r2_all:.3f}")
        print(f"Mean change in sign agreement (Model 3 - Model 1): {mean_sign_gain:+.3f}")
        oos_metrics = {
            "fraction_model3_beats_model1_rmse": round(frac_better, 4),
            "mean_factor_r_squared": round(mean_r2_all, 4),
            "mean_sign_agreement_gain": round(mean_sign_gain, 4),
        }

    tickers_tested = sorted(results["ticker"].unique().tolist()) if len(results) else []
    exp = Experiment(
        model_name="Model 3 - Regression/Factor (SPY, Gold, Oil; residual vs Model 1)",
        script="backtest_engine/evaluate_model3.py",
        universe=tickers_tested,
        horizons=HORIZONS,
        test_fraction=TEST_FRACTION,
        lookback_days=DEFAULT_LOOKBACK,
        hyperparameters={"factors": FACTOR_TICKERS, "regression_window": REGRESSION_WINDOW, "embargo_days": EMBARGO_DAYS},
        n_combinations_tested=len(tickers_tested) * len(HORIZONS),
        oos_metrics=oos_metrics,
        costs_included=False,
        notes=("Assumes zero expected factor-driven return - forecast is purely the "
               "idiosyncratic residual. Mean R2 reported to show how much variance the "
               "factors actually explain per instrument. Embargo-adjusted split."),
        decision="",
        influenced_later_decisions="",
    )
    exp_id = log_experiment(exp)
    print(f"\nLogged to research ledger as experiment #{exp_id}.")
    print("NOTE: decision field left blank - review the summary above and update the ledger row manually.")