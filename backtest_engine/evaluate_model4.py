"""
Phase 5 evaluation - ablation test: does Model 4 (probability model) beat
Model 1 (statistical baseline) out-of-sample? Also reports calibration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_engineering.features import build_features, HORIZONS
from models.statistical.baseline import forecast_series as model1_forecast, DEFAULT_LOOKBACK
from models.probability.model4 import forecast_series as model4_forecast
from research_ledger.ledger import Experiment, log_experiment
from backtest_engine.splitting import get_test_mask

TEST_FRACTION = 0.30
EMBARGO_DAYS = max(HORIZONS)


def evaluate_instrument(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    features = build_features(df)
    test_mask_base, split_idx, embargo_end_idx = get_test_mask(features.index, TEST_FRACTION, EMBARGO_DAYS)

    rows = []
    for h in HORIZONS:
        fc1 = model1_forecast(features, horizon=h)
        fc4 = model4_forecast(features, horizon=h)
        actual = features[f"target_forward_return_{h}d"]
        actual_pos = features[f"target_forward_positive_{h}d"]

        test_mask = (
            test_mask_base.to_numpy()
            & fc1["sufficient_sample"].to_numpy() & fc4["sufficient_sample"].to_numpy()
            & actual.notna().to_numpy()
        )
        if test_mask.sum() < 10:
            rows.append({
                "ticker": ticker, "horizon": h, "n_test": int(test_mask.sum()),
                "note": "insufficient out-of-sample observations",
                "rmse_model1": np.nan, "rmse_model4": np.nan,
                "sign_agree_model1": np.nan, "sign_agree_model4": np.nan,
                "model4_beats_model1_rmse": np.nan, "calibration_gap_model4": np.nan,
            })
            continue

        f1 = fc1.loc[test_mask, "expected_return"]
        f4 = fc4.loc[test_mask, "expected_return"]
        f4_prob = fc4.loc[test_mask, "prob_positive"]
        a_ret = actual.loc[test_mask]
        a_pos = actual_pos.loc[test_mask]

        rmse1 = float(np.sqrt(((f1 - a_ret) ** 2).mean()))
        rmse4 = float(np.sqrt(((f4 - a_ret) ** 2).mean()))
        sign1 = float((np.sign(f1) == np.sign(a_ret)).mean())
        sign4 = float((np.sign(f4) == np.sign(a_ret)).mean())

        try:
            buckets = pd.qcut(f4_prob, q=5, duplicates="drop")
            cal = pd.DataFrame({"pred": f4_prob, "actual": a_pos, "bucket": buckets})
            grp = cal.groupby("bucket", observed=True).agg(pred_mean=("pred", "mean"), actual_rate=("actual", "mean"))
            calibration_gap = float((grp["pred_mean"] - grp["actual_rate"]).abs().mean())
        except (ValueError, IndexError):
            calibration_gap = np.nan

        rows.append({
            "ticker": ticker, "horizon": h, "n_test": int(test_mask.sum()),
            "note": "", "rmse_model1": round(rmse1, 5), "rmse_model4": round(rmse4, 5),
            "sign_agree_model1": round(sign1, 3), "sign_agree_model4": round(sign4, 3),
            "model4_beats_model1_rmse": rmse4 < rmse1,
            "calibration_gap_model4": round(calibration_gap, 3) if not np.isnan(calibration_gap) else np.nan,
        })

    return pd.DataFrame(rows)


def run_all(processed_dir: Path) -> pd.DataFrame:
    all_results = []
    for csv_path in sorted(processed_dir.glob("*.csv")):
        ticker = csv_path.stem
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

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    results.to_csv(results_dir / "model4_results.csv", index=False)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 200)
    print(results.to_string(index=False))

    valid = results.dropna(subset=["rmse_model1", "rmse_model4"])
    oos_metrics = {}
    if len(valid):
        print("\n--- Ablation summary: Model 4 vs Model 1 ---")
        frac_better = float(valid["model4_beats_model1_rmse"].mean())
        mean_sign_gain = float((valid["sign_agree_model4"] - valid["sign_agree_model1"]).mean())
        mean_calib = float(valid["calibration_gap_model4"].mean())
        print(f"Fraction of instrument-horizons where Model 4 beats Model 1 on RMSE: {frac_better:.1%}")
        print(f"Mean change in sign agreement (Model 4 - Model 1): {mean_sign_gain:+.3f}")
        print(f"Mean calibration gap (Model 4): {mean_calib:.3f}  (lower = better calibrated)")
        oos_metrics = {
            "fraction_model4_beats_model1_rmse": round(frac_better, 4),
            "mean_sign_agreement_gain": round(mean_sign_gain, 4),
            "mean_calibration_gap": round(mean_calib, 4),
        }

    tickers_tested = sorted(results["ticker"].unique().tolist()) if len(results) else []
    exp = Experiment(
        model_name="Model 4 - Probability (logistic regression; vs Model 1)",
        script="backtest_engine/evaluate_model4.py",
        universe=tickers_tested,
        horizons=HORIZONS,
        test_fraction=TEST_FRACTION,
        lookback_days=DEFAULT_LOOKBACK,
        hyperparameters={"train_window": 400, "refit_every": 20, "embargo_days": EMBARGO_DAYS,
                          "features": "trailing mean/std/zscore/vol at 20/60/120d + autocorr_lag1_60d"},
        n_combinations_tested=len(tickers_tested) * len(HORIZONS),
        oos_metrics=oos_metrics,
        costs_included=False,
        notes=("Probability (logistic regression) and magnitude (stratified trailing mean, "
               "same mechanism as Model 1) modeled and combined separately per the blueprint. "
               "Point-in-time alignment for classifier training was caught and fixed during "
               "development before this ran on real data."),
        decision="",
        influenced_later_decisions="",
    )
    exp_id = log_experiment(exp)
    print(f"\nLogged to research ledger as experiment #{exp_id}.")
    print("NOTE: decision field left blank - review the summary above and update the ledger row manually.")