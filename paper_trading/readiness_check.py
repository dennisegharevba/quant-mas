"""
Live-Readiness Check - Phase 16.

Tracks the OBJECTIVE, checkable criteria for whether there is enough real
paper-trading evidence to seriously evaluate live deployment. Does NOT
tell you to go live - that is a judgment call about your own risk
tolerance and capital, and this is not financial advice. Reports the
facts plainly so the decision is based on accumulated evidence.

Run any time (e.g. alongside the daily paper trading cycle):
    python paper_trading\readiness_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_trading.tracker import load_positions

MIN_CLOSED_TRADES = 20


def check_closed_trade_evidence() -> dict:
    df = load_positions()
    closed = df[df["status"] == "CLOSED"].copy()
    n_closed = len(closed)

    result = {"n_closed": n_closed, "meets_minimum": n_closed >= MIN_CLOSED_TRADES}

    if n_closed == 0:
        return result

    closed["realized_return"] = pd.to_numeric(closed["realized_return"], errors="coerce")
    closed["realized_pnl_pct_equity"] = pd.to_numeric(closed["realized_pnl_pct_equity"], errors="coerce")

    result["win_rate"] = round(float((closed["realized_return"] > 0).mean()), 3)
    result["mean_return"] = round(float(closed["realized_return"].mean()), 5)
    result["total_pnl_pct_equity"] = round(float(closed["realized_pnl_pct_equity"].sum()), 5)

    by_model = closed.groupby("model_used").agg(
        n=("ticker", "count"),
        win_rate=("realized_return", lambda x: (x > 0).mean()),
        mean_return=("realized_return", "mean"),
    ).round(3)
    result["by_model"] = by_model

    return result


def print_readiness_report() -> None:
    print(f"\n{'='*78}")
    print("LIVE-READINESS CHECK")
    print(f"{'='*78}")
    print("\nThis reports objective facts only - it does not recommend going live.")
    print("Going live with real capital is a decision about your own risk")
    print("tolerance and capital that only you can make. This isn't financial advice.\n")

    evidence = check_closed_trade_evidence()
    n_closed = evidence["n_closed"]

    print(f"--- 1. Closed-trade sample size ---")
    print(f"Closed paper trades so far: {n_closed} (reference floor: {MIN_CLOSED_TRADES}+)")
    if n_closed == 0:
        print("No closed trades yet - too early to evaluate anything. Keep running the")
        print("daily paper trading cycle.")
    elif not evidence["meets_minimum"]:
        print(f"Below the reference floor - {MIN_CLOSED_TRADES - n_closed} more closed trades")
        print("needed before this sample says much of anything statistically.")
    else:
        print(f"Meets the reference floor of {MIN_CLOSED_TRADES}+ closed trades.")

    if n_closed > 0:
        print(f"\n--- 2. Aggregate paper-trading performance ---")
        print(f"Win rate: {evidence['win_rate']:.1%}  |  Mean return per trade: {evidence['mean_return']*100:.2f}%  |  "
              f"Total P&L: {evidence['total_pnl_pct_equity']*100:.3f}% of equity")
        print(f"\nBy model:")
        print(evidence["by_model"].to_string())

    print(f"\n--- 3. Other criteria (not automatically checkable - manual review) ---")
    print("[ ] Real transaction cost data (spread/slippage) in place - currently using")
    print("    documented TYPICAL cost assumptions (fx=1.5bps, stock=5bps, etc.), not")
    print("    measured data specific to your actual broker/execution.")
    print("[ ] No new bugs found for a sustained stretch of continued use.")
    print("[ ] Execution/broker integration built and tested - this system currently")
    print("    only produces signals and paper-tracks them; it does not place real")
    print("    orders. That is separate infrastructure, not yet built.")
    print("[ ] You are personally comfortable with the position sizing (2% single-trade")
    print("    cap, 3% correlated-cluster cap) at your actual account size.")

    print(f"\n{'='*78}\n")


if __name__ == "__main__":
    print_readiness_report()