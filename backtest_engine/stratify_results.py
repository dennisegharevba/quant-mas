"""
Asset-Class Stratification - Architecture correction addendum.

Re-slices each evaluate_modelN.py's saved results by asset class, using
the class labels from config/universe.py. Requires each evaluate_modelN.py
to have been re-run at least once after the results-saving patch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.universe import UNIVERSE

RESULTS_DIR = Path(__file__).resolve().parents[1] / "backtest_engine" / "results"

TICKER_TO_CLASS = {inst.ticker.replace("=", "_"): inst.asset_class for inst in UNIVERSE}

MODEL_METRIC_SPECS = {
    "model1": {
        "file": "model1_results.csv",
        "rmse_model_col": "rmse_model", "rmse_ref_col": "rmse_naive_zero",
        "extra_cols": ["sign_agreement", "corr_forecast_actual", "calibration_gap"],
    },
    "model2": {
        "file": "model2_results.csv",
        "rmse_model_col": "rmse_model2", "rmse_ref_col": "rmse_model1",
        "extra_cols": ["sign_agree_model2", "pct_regime_conditioned"],
    },
    "model3": {
        "file": "model3_results.csv",
        "rmse_model_col": "rmse_model3", "rmse_ref_col": "rmse_model1",
        "extra_cols": ["sign_agree_model3", "mean_r2"],
    },
    "model4": {
        "file": "model4_results.csv",
        "rmse_model_col": "rmse_model4", "rmse_ref_col": "rmse_model1",
        "extra_cols": ["sign_agree_model4", "calibration_gap_model4"],
    },
    "model5": {
        "file": "model5_results.csv",
        "rmse_model_col": "rmse_ewma", "rmse_ref_col": "rmse_naive",
        "extra_cols": ["corr_ewma_actual", "corr_naive_actual"],
    },
}


def load_and_tag(model_key: str) -> pd.DataFrame:
    spec = MODEL_METRIC_SPECS[model_key]
    path = RESULTS_DIR / spec["file"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - re-run backtest_engine/evaluate_{model_key}.py "
            f"after applying the results-saving patch."
        )
    df = pd.read_csv(path)
    df["asset_class"] = df["ticker"].map(TICKER_TO_CLASS)
    unmapped = df[df["asset_class"].isna()]["ticker"].unique()
    if len(unmapped) > 0:
        print(f"  [warning] tickers not found in universe map: {list(unmapped)}")
    return df


def stratified_summary(model_key: str) -> pd.DataFrame:
    spec = MODEL_METRIC_SPECS[model_key]
    df = load_and_tag(model_key)
    df = df.dropna(subset=[spec["rmse_model_col"], spec["rmse_ref_col"]])

    rows = []
    for asset_class, grp in df.groupby("asset_class"):
        n_tickers = grp["ticker"].nunique()
        n_combos = len(grp)
        win_rate = float((grp[spec["rmse_model_col"]] < grp[spec["rmse_ref_col"]]).mean())
        row = {
            "asset_class": asset_class, "n_tickers": n_tickers, "n_combinations": n_combos,
            "rmse_win_rate": round(win_rate, 3),
        }
        for col in spec["extra_cols"]:
            if col in grp.columns:
                row[f"mean_{col}"] = round(float(grp[col].mean()), 4)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("asset_class").reset_index(drop=True)


def print_all_stratified() -> None:
    for model_key in MODEL_METRIC_SPECS:
        print(f"\n{'='*70}")
        print(f"  {model_key.upper()} - stratified by asset class")
        print(f"{'='*70}")
        try:
            summary = stratified_summary(model_key)
            pd.set_option("display.width", 140)
            print(summary.to_string(index=False))
        except FileNotFoundError as e:
            print(f"  [skip] {e}")


if __name__ == "__main__":
    print_all_stratified()