"""
Run Monte Carlo - Phase 9.

Runs the scanner, and for every instrument that currently passes all
NO-TRADE gates, stress-tests the recommended position size via block-
bootstrap Monte Carlo simulation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scanner import run_scan, print_scan_report, PRIMARY_HORIZON
from position_sizer.sizer import recommend_position_size
from montecarlo_engine.simulate import run_monte_carlo, print_mc_report


if __name__ == "__main__":
    processed = Path(__file__).resolve().parent / "data" / "processed"
    scan = run_scan(processed, horizon=PRIMARY_HORIZON)

    if len(scan) == 0:
        print("No instruments found in data/processed - run data_ingestion and data_cleaning first.")
        sys.exit(0)

    print_scan_report(scan, PRIMARY_HORIZON)

    traded = scan[scan["decision"] == "TRADE"]
    if len(traded) == 0:
        print("\nNo TRADE candidates - nothing to stress test.")
        sys.exit(0)

    for _, r in traded.iterrows():
        ticker = r["ticker"]
        sizing = recommend_position_size(
            prob_win=r["prob_positive"], mean_win=r["mean_win"], mean_loss=r["mean_loss"],
        )
        position_fraction = sizing["recommended_fraction"]
        if position_fraction <= 0:
            continue

        df = pd.read_csv(processed / f"{ticker}.csv", index_col=0, parse_dates=True)
        close = df["Close"].astype(float)
        daily_ret = np.log(close / close.shift(1)).dropna().to_numpy()

        direction_sign = 1.0 if r["expected_return"] > 0 else -1.0
        mc_result = run_monte_carlo(daily_ret * direction_sign, position_fraction=position_fraction)
        print_mc_report(mc_result, ticker)