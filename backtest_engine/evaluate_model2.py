"""
Phase 3 evaluation - ablation test: does Model 2 (time-series) beat
Model 1 (statistical baseline) out-of-sample?

Not "does Model 2 beat naive-zero" but "does adding this specific
component improve on what we already had" - the actual ablation test the
blueprint calls for. Same chronological train/test split as Model 1's
evaluation, both models scored on identical rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_engineering.features import build_features, HORIZONS
from models.statistical.baseline import forecast_series as model1_forecast, DEFAULT_LOOKBACK
from models.timeseries.model2 import forecast_series as model2_forecast
from research_ledger.ledger import Experiment, log_experiment

TEST_FRACTION = 0.30


def evaluate_instrument(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    features = build_features(df)
    n = len(features)
    split_idx = int(n * (1 - TEST_FRACTION))
    split_date = features.index[split_idx]

    rows = []
    for h in HORIZONS:
        fc1 = model1_forecast(features, horizon=h)
        fc2 = model2_forecast(features, horizon=h)
        actual = features[f"target_forward_return_{h}d"]

        test_mask = (
            (features.index >= split_date)
            & fc1["sufficient_sample"] & fc2["sufficient_sample"]
            & actual.notna()
        )
        if test_mask.sum() < 10:
            rows.append({
                "ticker": ticker, "horizon": h, "n_test": int(test_mask.sum()),
                "note": "insufficient out-of-sample observations",
                "pct_regime_conditioned": np.nan,
                "rmse_model1": np.nan, "rmse_model2": np.nan,
                "sign_agree_model1": np.nan, "sign_agree_model2": np.nan,
                "model2_beats_model1_rmse": np.nan,
            })
            continue

        f1 = fc1.loc[test_mask, "expected_return"]
        f2 = fc2.loc[test_mask, "expected_return"]
        a_ret = actual.loc[test_mask]
        pct_conditioned = float(fc2.loc[test_mask, "regime_conditioned"].mean())

        rmse1 = float(np.sqrt(((f1 - a_ret) ** 2).mean()))
        rmse2 = float(np.sqrt(((f2 - a_ret) ** 2).mean()))
        sign1 = float((np.sign(f1) == np.sign(a_ret)).mean())
        sign2 = float((np.sign(f2) == np.sign(a_ret)).mean())

        rows.append({
            "ticker": ticker, "horizon": h, "n_test": int(test_mask.sum()),
            "note": "", "pct_regime_conditioned": round(pct_conditioned, 3),
            "rmse_model1": round(rmse1, 5), "rmse_model2": round(rmse2, 5),
            "sign_agree_model1": round(sign1, 3), "sign_agree_model2": round(sign2, 3),
            "model2_beats_model1_rmse": rmse2 < rmse1,
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
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 200)
    print(results.to_string(index=False))

    valid = results.dropna(subset=["rmse_model1", "rmse_model2"])
    oos_metrics = {}
    if len(valid):
        print("\n--- Ablation summary: Model 2 vs Model 1 ---")
        frac_better = float(valid["model2_beats_model1_rmse"].mean())
        mean_conditioned = float(valid["pct_regime_conditioned"].mean())
        mean_sign_gain = float((valid["sign_agree_model2"] - valid["sign_agree_model1"]).mean())
        print(f"Fraction of instrument-horizons where Model 2 beats Model 1 on RMSE: {frac_better:.1%}")
        print(f"Mean fraction of test days where regime-conditioning actually fired: {mean_conditioned:.1%}")
        print(f"Mean change in sign agreement (Model 2 - Model 1): {mean_sign_gain:+.3f}")
        oos_metrics = {
            "fraction_model2_beats_model1_rmse": round(frac_better, 4),
            "mean_pct_regime_conditioned": round(mean_conditioned, 4),
            "mean_sign_agreement_gain": round(mean_sign_gain, 4),
        }

    tickers_tested = sorted(results["ticker"].unique().tolist()) if len(results) else []
    exp = Experiment(
        model_name="Model 2 - Time-Series (regime-conditioned on Model 1)",
        script="backtest_engine/evaluate_model2.py",
        universe=tickers_tested,
        horizons=HORIZONS,
        test_fraction=TEST_FRACTION,
        lookback_days=DEFAULT_LOOKBACK,
        hyperparameters={"ljungbox_alpha": 0.05, "min_regime_samples": 30, "momentum_window": 20},
        n_combinations_tested=len(tickers_tested) * len(HORIZONS),
        oos_metrics=oos_metrics,
        costs_included=False,
        notes="Ablation test against Model 1 on identical out-of-sample rows, not against naive-zero.",
        decision="",
        influenced_later_decisions="",
    )
    exp_id = log_experiment(exp)
    print(f"\nLogged to research ledger as experiment #{exp_id}.")
    print("NOTE: decision field left blank - review the summary above and update the ledger row manually.")