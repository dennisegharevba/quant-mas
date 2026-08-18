"""
Phase 2 evaluation - out-of-sample test of Model 1 (statistical baseline).
Logs itself to the research ledger automatically on every run.
Uses an embargo gap around the train/test boundary (see splitting.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_engineering.features import build_features, HORIZONS
from models.statistical.baseline import forecast_series, DEFAULT_LOOKBACK
from research_ledger.ledger import Experiment, log_experiment
from backtest_engine.splitting import get_test_mask

TEST_FRACTION = 0.30
EMBARGO_DAYS = max(HORIZONS)


def evaluate_instrument(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    features = build_features(df)
    test_mask_base, split_idx, embargo_end_idx = get_test_mask(features.index, TEST_FRACTION, EMBARGO_DAYS)

    rows = []
    for h in HORIZONS:
        fc = forecast_series(features, horizon=h)
        actual = features[f"target_forward_return_{h}d"]
        actual_pos = features[f"target_forward_positive_{h}d"]

        test_mask = test_mask_base.to_numpy() & fc["sufficient_sample"].to_numpy() & actual.notna().to_numpy()
        if test_mask.sum() < 10:
            rows.append({
                "ticker": ticker, "horizon": h, "n_test": int(test_mask.sum()),
                "note": "insufficient out-of-sample observations", "sign_agreement": np.nan,
                "corr_forecast_actual": np.nan, "rmse_model": np.nan, "rmse_naive_zero": np.nan,
                "calibration_gap": np.nan,
            })
            continue

        f_exp = fc.loc[test_mask, "expected_return"]
        f_prob = fc.loc[test_mask, "prob_positive"]
        a_ret = actual.loc[test_mask]
        a_pos = actual_pos.loc[test_mask]

        sign_agree = float((np.sign(f_exp) == np.sign(a_ret)).mean())
        corr = float(np.corrcoef(f_exp, a_ret)[0, 1]) if f_exp.std() > 0 else np.nan
        rmse_model = float(np.sqrt(((f_exp - a_ret) ** 2).mean()))
        rmse_naive = float(np.sqrt((a_ret ** 2).mean()))

        try:
            buckets = pd.qcut(f_prob, q=5, duplicates="drop")
            cal = pd.DataFrame({"pred": f_prob, "actual": a_pos, "bucket": buckets})
            grp = cal.groupby("bucket", observed=True).agg(pred_mean=("pred", "mean"), actual_rate=("actual", "mean"))
            calibration_gap = float((grp["pred_mean"] - grp["actual_rate"]).abs().mean())
        except (ValueError, IndexError):
            calibration_gap = np.nan

        rows.append({
            "ticker": ticker, "horizon": h, "n_test": int(test_mask.sum()),
            "note": "", "sign_agreement": round(sign_agree, 3),
            "corr_forecast_actual": round(corr, 3) if not np.isnan(corr) else np.nan,
            "rmse_model": round(rmse_model, 5), "rmse_naive_zero": round(rmse_naive, 5),
            "calibration_gap": round(calibration_gap, 3) if not np.isnan(calibration_gap) else np.nan,
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
    pd.set_option("display.width", 140)
    pd.set_option("display.max_rows", 200)
    print(results.to_string(index=False))

    valid = results.dropna(subset=["rmse_model"])
    oos_metrics = {}
    if len(valid):
        print("\n--- Summary across all instruments/horizons ---")
        mean_sign = float(valid["sign_agreement"].mean())
        beats_naive = float((valid["rmse_model"] < valid["rmse_naive_zero"]).mean())
        mean_calib = float(valid["calibration_gap"].mean())
        print(f"Mean sign agreement: {mean_sign:.3f}  (>0.50 = better than coin flip)")
        print(f"Fraction beating naive-zero RMSE: {beats_naive:.1%}")
        print(f"Mean calibration gap: {mean_calib:.3f}  (lower = better calibrated)")
        oos_metrics = {
            "mean_sign_agreement": round(mean_sign, 4),
            "fraction_beats_naive_rmse": round(beats_naive, 4),
            "mean_calibration_gap": round(mean_calib, 4),
        }

    tickers_tested = sorted(results["ticker"].unique().tolist()) if len(results) else []
    exp = Experiment(
        model_name="Model 1 - Statistical Baseline",
        script="backtest_engine/evaluate_model1.py",
        universe=tickers_tested,
        horizons=HORIZONS,
        test_fraction=TEST_FRACTION,
        lookback_days=DEFAULT_LOOKBACK,
        hyperparameters={"min_samples": 60, "n_bootstrap": 2000},
        n_combinations_tested=len(tickers_tested) * len(HORIZONS),
        oos_metrics=oos_metrics,
        costs_included=False,
        notes=("Correlation between forecast and realized return was negative across "
               "nearly all instruments/horizons. RESOLVED via Monte Carlo simulation "
               "(300 sims, pure i.i.d. random walk): confirmed this is a known structural "
               "artifact of overlapping-window trailing-mean estimators (mean correlation "
               "-0.02 to -0.06, growing more negative with horizon, even under zero true "
               "autocorrelation) - not a real market finding. Embargo added to the "
               "evaluation split (see backtest_engine/splitting.py) as a separate, "
               "unrelated methodology improvement."),
        decision=("Does not clear the bar to build on. RMSE beats naive-zero in only ~20% "
                   "of instrument-horizon combinations; sign agreement is near coin-flip on "
                   "average and inconsistent across asset classes (near-zero on FX, "
                   "mildly positive on indices/metals, negative on crypto). Treated as the "
                   "expected null-hypothesis result, not a failure."),
        influenced_later_decisions=("Establishes the baseline Model 2 (time-series) must beat "
                                     "out of sample, after costs, to be added."),
    )
    exp_id = log_experiment(exp)
    print(f"\nLogged to research ledger as experiment #{exp_id}.")