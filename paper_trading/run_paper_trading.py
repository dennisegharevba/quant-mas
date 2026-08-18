"""
Run Paper Trading - Phase 11.

Run this periodically (e.g. daily, after refreshing data via
data_ingestion + data_cleaning):

    python paper_trading\run_paper_trading.py

Each run closes any position whose holding period has elapsed using real
subsequent price movement, then opens new positions for fresh TRADE
signals (skipping tickers that already have an open position).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scanner import run_scan, PRIMARY_HORIZON
from paper_trading.tracker import (
    close_eligible_positions, open_position, has_open_position, summary_stats,
)


def run_paper_trading_cycle(processed_dir: Path, horizon: int = PRIMARY_HORIZON) -> None:
    print("=" * 78)
    print("PAPER TRADING CYCLE")
    print("=" * 78)

    closed = close_eligible_positions(processed_dir)
    if closed:
        print(f"\nClosed {len(closed)} position(s) this run:")
        for c in closed:
            print(f"  {c['ticker']} ({c['direction']}): entered {c['entry_date']}, "
                  f"exited {c['exit_date']}, raw_return={c['raw_return']*100:.2f}%, "
                  f"pnl={c['pnl_pct_equity']*100:.3f}% of equity")
    else:
        print("\nNo positions eligible to close this run.")

    print("\nRunning scanner for new signals...")
    scan = run_scan(processed_dir, horizon=horizon)
    if len(scan) == 0:
        print("No instruments found - run data_ingestion and data_cleaning first.")
        return

    traded = scan[scan["decision"] == "TRADE"]
    opened_count = 0
    for _, r in traded.iterrows():
        ticker = r["ticker"]
        if has_open_position(ticker):
            print(f"  [skip] {ticker}: already has an open paper position")
            continue

        direction = "LONG" if r["expected_return"] > 0 else "SHORT"
        from position_sizer.sizer import recommend_position_size
        if direction == "SHORT":
            sizing = recommend_position_size(
                prob_win=1 - r["prob_positive"], mean_win=-r["mean_loss"], mean_loss=-r["mean_win"],
            )
        else:
            sizing = recommend_position_size(
                prob_win=r["prob_positive"], mean_win=r["mean_win"], mean_loss=r["mean_loss"],
            )
        size_fraction = sizing["recommended_fraction"]
        if size_fraction <= 0:
            print(f"  [skip] {ticker}: position sizer recommended 0% - not opening")
            continue

        import pandas as pd
        entry_date = pd.Timestamp(r["date"])
        entry_price = None
        df = pd.read_csv(processed_dir / f"{ticker}.csv", index_col=0, parse_dates=True)
        if entry_date in df.index:
            entry_price = float(df.loc[entry_date, "Close"])
        if entry_price is None:
            print(f"  [skip] {ticker}: could not resolve entry price")
            continue

        open_position(ticker, r["model_used"], direction, horizon, entry_date, entry_price, size_fraction)
        opened_count += 1
        print(f"  Opened {ticker} ({direction}), model={r['model_used']}, "
              f"size={size_fraction:.1%}, entry_price={entry_price:.4f}, entry_date={entry_date.date()}")

    if opened_count == 0:
        print("  No new positions opened this run.")

    stats = summary_stats()
    print(f"\n--- Paper trading summary ---")
    print(f"Open positions: {stats.get('n_open', 0)}")
    if stats.get("n_closed", 0) > 0:
        print(f"Closed positions: {stats['n_closed']}  |  Win rate: {stats['win_rate']:.1%}  |  "
              f"Mean return: {stats['mean_return']*100:.2f}%  |  "
              f"Total P&L: {stats['total_pnl_pct_equity']*100:.3f}% of equity")
    else:
        print("Closed positions: 0 (nothing has completed its holding period yet)")


if __name__ == "__main__":
    processed = Path(__file__).resolve().parents[1] / "data" / "processed"
    run_paper_trading_cycle(processed)