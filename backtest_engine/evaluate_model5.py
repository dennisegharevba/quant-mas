"""
Phase 6 evaluation - does EWMA volatility forecasting beat a naive flat
rolling-window estimate at predicting forward realized volatility?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.risk.model5 import (
    compute_ewma_vol, compute_naive_vol, build_forward_realized_vol, HORIZONS,
)
from research_ledger.ledger import Experiment, log_experiment
from backtest_engine.splitting import get_test_mask

TEST_FRACTION = 0.30
EMBARGO_DAYS = 10


def evaluate_instrument(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    close = df["Close"].astype(float)
    daily_ret = np.log(close / close.shift(1))

    ewma_vol = compute_ewma_vol(daily_ret)
    naive_vol = compute_naive_vol(daily_ret)
    forward_vol = build_forward_realized_vol(daily_ret)

    test_mask_base, split_idx, embargo_end_idx = get_test_mask(close.index, TEST_FRACTION, EMBARGO_DAYS)

    rows = []
    for h in HORIZONS:
        target = forward_vol[f"target_forward_vol_{h}d"]
        test_mask = (
            test_mask_base.to_numpy()
            & ewma_vol.notna().to_numpy() & naive_vol.notna().to_numpy() & target.notna().to_numpy()
        )
        if test_mask.sum() < 10:
            rows.append({
                "ticker": ticker, "horizon": h, "n_test": int(test_mask.sum()),
                "note": "insufficient out-of-sample observations",
                "rmse_ewma": np.nan, "rmse_naive": np.nan, "ewma_beats_naive": np.nan,
                "corr_ewma_actual": np.nan, "corr_naive_actual": np.nan,
            })
            continue

        e = ewma_vol.loc[test_mask]
        nv = naive_vol.loc[test_mask]
        a = target.loc[test_mask]

        rmse_ewma = float(np.sqrt(((e - a) ** 2).mean()))
        rmse_naive = float(np.sqrt(((nv - a) ** 2).mean()))
        corr_ewma = float(np.corrcoef(e, a)[0, 1]) if e.std() > 0 else np.nan
        corr_naive = float(np.corrcoef(nv, a)[0, 1]) if nv.std() > 0 else np.nan

        rows.append({
            "ticker": ticker, "horizon": h, "n_test": int(test_mask.sum()),
            "note": "", "rmse_ewma": round(rmse_ewma, 5), "rmse_naive": round(rmse_naive, 5),
            "ewma_beats_naive": rmse_ewma < rmse_naive,
            "corr_ewma_actual": round(corr_ewma, 3) if not np.isnan(corr_ewma) else np.nan,
            "corr_naive_actual": round(corr_naive, 3) if not np.isnan(corr_naive) else np.nan,
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
    results.to_csv(results_dir / "model5_results.csv", index=False)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 200)
    print(results.to_string(index=False))

    valid = results.dropna(subset=["rmse_ewma", "rmse_naive"])
    oos_metrics = {}
    if len(valid):
        print("\n--- Ablation summary: EWMA vs naive rolling-window volatility forecast ---")
        frac_better = float(valid["ewma_beats_naive"].mean())
        mean_corr_ewma = float(valid["corr_ewma_actual"].mean())
        mean_corr_naive = float(valid["corr_naive_actual"].mean())
        print(f"Fraction of instrument-horizons where EWMA beats naive on RMSE: {frac_better:.1%}")
        print(f"Mean correlation with actual forward vol - EWMA: {mean_corr_ewma:.3f}, naive: {mean_corr_naive:.3f}")
        oos_metrics = {
            "fraction_ewma_beats_naive_rmse": round(frac_better, 4),
            "mean_corr_ewma": round(mean_corr_ewma, 4),
            "mean_corr_naive": round(mean_corr_naive, 4),
        }

    tickers_tested = sorted(results["ticker"].unique().tolist()) if len(results) else []
    exp = Experiment(
        model_name="Model 5 - Volatility/Risk (EWMA vs naive rolling window)",
        script="backtest_engine/evaluate_model5.py",
        universe=tickers_tested,
        horizons=HORIZONS,
        test_fraction=TEST_FRACTION,
        lookback_days=250,
        hyperparameters={"ewma_lambda": 0.94, "naive_window": 20, "embargo_days": EMBARGO_DAYS},
        n_combinations_tested=len(tickers_tested) * len(HORIZONS),
        oos_metrics=oos_metrics,
        costs_included=False,
        notes=("Different ablation shape than Models 1-4: tests whether EWMA volatility "
               "clustering-aware weighting beats a naive flat rolling window at predicting "
               "forward realized volatility, not return direction. h=1 excluded (undefined)."),
        decision="",
        influenced_later_decisions="",
    )
    exp_id = log_experiment(exp)
    print(f"\nLogged to research ledger as experiment #{exp_id}.")
    print("NOTE: decision field left blank - review the summary above and update the ledger row manually.")