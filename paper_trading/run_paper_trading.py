"""
Run Paper Trading - Phase 11-13.

Run this periodically (e.g. daily, after refreshing data via
data_ingestion + data_cleaning):

    python paper_trading\run_paper_trading.py

Each run closes any position whose holding period has elapsed using real
subsequent price movement, then opens new positions for fresh TRADE
signals - now with correlation-aware sizing that accounts for BOTH other
new candidates AND already-open positions, so correlated risk can't
silently stack across separate days.
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
    if len(traded) == 0:
        print("  No new positions opened this run.")
    else:
        candidates = []
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
            if sizing["recommended_fraction"] <= 0:
                print(f"  [skip] {ticker}: position sizer recommended 0% - not opening")
                continue
            candidates.append({
                "ticker": ticker, "direction": direction, "model_used": r["model_used"],
                "date": r["date"], "recommended_fraction": sizing["recommended_fraction"],
            })

        adjusted_by_ticker = {c["ticker"]: c["recommended_fraction"] for c in candidates}
        if candidates:
            try:
                from portfolio_optimizer.optimizer import build_correlation_matrix
                from paper_trading.tracker import load_positions

                open_positions = load_positions()
                open_positions = open_positions[open_positions["status"] == "OPEN"]
                open_tickers = open_positions["ticker"].tolist()
                new_tickers = [c["ticker"] for c in candidates]
                all_tickers = list(set(open_tickers) | set(new_tickers))

                if len(all_tickers) >= 2:
                    corr = build_correlation_matrix(processed_dir, all_tickers)
                    from portfolio_optimizer.optimizer import identify_clusters, MAX_CLUSTER_EXPOSURE
                    clusters = identify_clusters(corr)

                    for cluster in clusters:
                        already_committed = float(
                            open_positions[open_positions["ticker"].isin(cluster)]["size_fraction"].sum()
                        )
                        new_in_cluster = [c for c in candidates if c["ticker"] in cluster]
                        if not new_in_cluster:
                            continue
                        available_room = max(0.0, MAX_CLUSTER_EXPOSURE - already_committed)
                        total_requested = sum(c["recommended_fraction"] for c in new_in_cluster)
                        scale = min(1.0, available_room / total_requested) if total_requested > 0 else 0.0
                        for c in new_in_cluster:
                            adjusted_by_ticker[c["ticker"]] = c["recommended_fraction"] * scale
                            if scale < 1.0:
                                other_members = [t for t in cluster if t != c["ticker"]]
                                print(f"  [correlation] {c['ticker']} scaled from {c['recommended_fraction']:.1%} "
                                      f"to {adjusted_by_ticker[c['ticker']]:.1%} - correlated with {other_members} "
                                      f"({already_committed:.1%} already committed to this cluster)")
            except Exception as e:
                print(f"  [correlation adjustment unavailable: {e}]")

        opened_count = 0
        for c in candidates:
            size_fraction = adjusted_by_ticker[c["ticker"]]
            if size_fraction <= 0:
                print(f"  [skip] {c['ticker']}: correlation adjustment reduced size to 0% - cluster already fully committed")
                continue

            import pandas as pd
            entry_date = pd.Timestamp(c["date"])
            df = pd.read_csv(processed_dir / f"{c['ticker']}.csv", index_col=0, parse_dates=True)
            if entry_date not in df.index:
                print(f"  [skip] {c['ticker']}: could not resolve entry price")
                continue
            entry_price = float(df.loc[entry_date, "Close"])

            open_position(c["ticker"], c["model_used"], c["direction"], horizon, entry_date, entry_price, size_fraction)
            opened_count += 1
            print(f"  Opened {c['ticker']} ({c['direction']}), model={c['model_used']}, "
                  f"size={size_fraction:.1%}, entry_price={entry_price:.4f}, entry_date={entry_date.date()}")

        if opened_count == 0 and candidates:
            print("  No new positions opened this run (all reduced to 0% by correlation limits or already open).")

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