"""
Quant Market Scanner - Phase 7-10.

Ties the ranking engine, NO-TRADE filter, and position sizer together
into the actual per-scan output. Position sizing is now direction-aware
(a bug fix): for a SHORT trade, win probability and win/loss magnitudes
are flipped from the raw long-convention values, since "winning" for a
short means the price goes down, not up.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ranking_engine.rank import rank_universe
from no_trade_filter.filter import evaluate_no_trade, ASSET_CLASS_COST_BPS
from position_sizer.sizer import recommend_position_size

PRIMARY_HORIZON = 5


def run_scan(processed_dir: Path, horizon: int = PRIMARY_HORIZON) -> pd.DataFrame:
    scan = rank_universe(processed_dir, horizon)
    if len(scan) == 0:
        return scan

    decisions = []
    all_reasons = []
    for _, row in scan.iterrows():
        result = evaluate_no_trade(row.to_dict())
        decisions.append(result.decision)
        all_reasons.append("; ".join(result.reasons) if result.reasons else "")

    scan["decision"] = decisions
    scan["no_trade_reasons"] = all_reasons
    return scan


def print_scan_report(scan: pd.DataFrame, horizon: int) -> None:
    print(f"\n{'='*78}")
    print(f"QUANT MARKET SCANNER  -  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Horizon: {horizon}D  |  Universe scanned: {len(scan)} instruments")
    cost_str = ", ".join(f"{cls}={bps}bps" for cls, bps in ASSET_CLASS_COST_BPS.items())
    print(f"Assumed round-trip costs (documented typical, not measured): {cost_str}")
    print(f"{'='*78}")

    n_trade = (scan["decision"] == "TRADE").sum()
    n_no_trade = (scan["decision"] == "NO TRADE").sum()
    print(f"\nDecision: {n_trade} TRADE / {n_no_trade} NO TRADE\n")

    scan["decision_display"] = scan.apply(
        lambda r: (f"TRADE (LONG)" if r["decision"] == "TRADE" and r["expected_return"] > 0
                    else f"TRADE (SHORT)" if r["decision"] == "TRADE" and r["expected_return"] < 0
                    else r["decision"]),
        axis=1,
    )

    cols = ["rank", "ticker", "model_used", "date", "expected_return", "prob_positive", "ewma_vol",
            "risk_adjusted_score", "composite_score", "decision_display", "no_trade_reasons"]
    display = scan[cols].copy()
    display["expected_return"] = (display["expected_return"] * 100).round(3)
    display["prob_positive"] = (display["prob_positive"] * 100).round(1)
    display["ewma_vol"] = (display["ewma_vol"] * 100).round(1)
    display = display.rename(columns={
        "expected_return": "E[R]%", "prob_positive": "P(win)%", "ewma_vol": "vol%(ann)",
        "risk_adjusted_score": "risk_adj", "composite_score": "score", "decision_display": "decision",
    })
    pd.set_option("display.width", 180)
    pd.set_option("display.max_rows", 200)
    print(display.to_string(index=False))

    traded = scan[scan["decision"] == "TRADE"]
    if len(traded) > 0:
        print(f"\n--- {len(traded)} instrument(s) passed all current gates ---")
        for _, r in traded.iterrows():
            direction = "LONG" if r["expected_return"] > 0 else "SHORT"
            print(f"  {r['ticker']} ({direction}): E[R]={r['expected_return']*100:.2f}%, "
                  f"P(win)={r['prob_positive']*100:.1f}%, vol={r['ewma_vol']*100:.1f}%, "
                  f"CI=[{r['ci_low']*100:.2f}%, {r['ci_high']*100:.2f}%]")
            if direction == "SHORT":
                sizing = recommend_position_size(
                    prob_win=1 - r["prob_positive"], mean_win=-r["mean_loss"], mean_loss=-r["mean_win"],
                )
            else:
                sizing = recommend_position_size(
                    prob_win=r["prob_positive"], mean_win=r["mean_win"], mean_loss=r["mean_loss"],
                )
            print(f"    Position sizing (fractional Kelly, k=0.25, max 2% cap): "
                  f"{sizing['recommended_fraction']:.1%} of equity  "
                  f"(full Kelly={sizing['kelly_full']}, {sizing['note']})")
    else:
        print("\n--- NO TRADE across the entire universe at this horizon ---")
        print("(This is the expected, honest result given experiments 1-5: the surviving")
        print(" return estimate did not itself demonstrate a robust edge in backtesting.)")


if __name__ == "__main__":
    processed = Path(__file__).resolve().parent / "data" / "processed"
    scan = run_scan(processed, horizon=PRIMARY_HORIZON)
    if len(scan) == 0:
        print("No instruments found in data/processed - run data_ingestion and data_cleaning first.")
    else:
        print_scan_report(scan, PRIMARY_HORIZON)