"""
Paper Trading Tracker - Phase 11 (final phase per the original roadmap).

No capital at risk - tracks what WOULD have happened if signals from the
scanner had actually been traded, using real (not backtested) subsequent
price movement.

INTENDED USE: run paper_trading/run_paper_trading.py periodically (e.g.
daily, after refreshing data via data_ingestion + data_cleaning). Each
run closes any position whose holding period has elapsed using real
subsequent price movement, then opens new positions for fresh TRADE
signals.

State is persisted in paper_trading/positions.csv.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

POSITIONS_PATH = Path(__file__).resolve().parent / "positions.csv"

POSITIONS_COLUMNS = [
    "position_id", "ticker", "model_used", "direction", "horizon_days",
    "entry_date", "entry_price", "size_fraction",
    "status", "exit_date", "exit_price", "realized_return", "realized_pnl_pct_equity",
    "opened_at_utc", "closed_at_utc",
]


def load_positions() -> pd.DataFrame:
    if not POSITIONS_PATH.exists():
        return pd.DataFrame(columns=POSITIONS_COLUMNS)
    df = pd.read_csv(POSITIONS_PATH, parse_dates=["entry_date", "exit_date"])
    for col in ["position_id", "ticker", "model_used", "direction", "status",
                "opened_at_utc", "closed_at_utc"]:
        if col in df.columns:
            df[col] = df[col].astype(object)
    return df


def save_positions(df: pd.DataFrame) -> None:
    df.to_csv(POSITIONS_PATH, index=False)


def open_position(ticker: str, model_used: str, direction: str, horizon_days: int,
                   entry_date: pd.Timestamp, entry_price: float, size_fraction: float) -> pd.DataFrame:
    df = load_positions()
    new_row = {
        "position_id": str(uuid.uuid4())[:8],
        "ticker": ticker, "model_used": model_used, "direction": direction,
        "horizon_days": horizon_days, "entry_date": entry_date, "entry_price": entry_price,
        "size_fraction": size_fraction, "status": "OPEN",
        "exit_date": pd.NaT, "exit_price": np.nan, "realized_return": np.nan, "realized_pnl_pct_equity": np.nan,
        "opened_at_utc": datetime.now(timezone.utc).isoformat(), "closed_at_utc": "",
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_positions(df)
    return df


def has_open_position(ticker: str) -> bool:
    df = load_positions()
    if len(df) == 0:
        return False
    return bool(((df["ticker"] == ticker) & (df["status"] == "OPEN")).any())


def close_eligible_positions(processed_dir: Path) -> list[dict]:
    df = load_positions()
    if len(df) == 0:
        return []

    closed_summaries = []
    for idx, pos in df[df["status"] == "OPEN"].iterrows():
        csv_path = processed_dir / f"{pos['ticker']}.csv"
        if not csv_path.exists():
            continue
        price_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        close = price_df["Close"].astype(float)

        if pos["entry_date"] not in close.index:
            continue

        entry_loc = close.index.get_loc(pos["entry_date"])
        exit_loc = entry_loc + int(pos["horizon_days"])
        if exit_loc >= len(close):
            continue

        exit_date = close.index[exit_loc]
        exit_price = float(close.iloc[exit_loc])
        entry_price = float(pos["entry_price"])
        direction_sign = 1.0 if pos["direction"] == "LONG" else -1.0
        raw_return = direction_sign * np.log(exit_price / entry_price)
        pnl_pct_equity = raw_return * float(pos["size_fraction"])

        df.loc[idx, "status"] = "CLOSED"
        df.loc[idx, "exit_date"] = exit_date
        df.loc[idx, "exit_price"] = exit_price
        df.loc[idx, "realized_return"] = raw_return
        df.loc[idx, "realized_pnl_pct_equity"] = pnl_pct_equity
        df.loc[idx, "closed_at_utc"] = datetime.now(timezone.utc).isoformat()

        closed_summaries.append({
            "ticker": pos["ticker"], "direction": pos["direction"],
            "entry_date": str(pos["entry_date"].date()), "exit_date": str(exit_date.date()),
            "raw_return": raw_return, "pnl_pct_equity": pnl_pct_equity,
        })

    save_positions(df)
    return closed_summaries


def summary_stats() -> dict:
    df = load_positions()
    closed = df[df["status"] == "CLOSED"]
    open_count = int((df["status"] == "OPEN").sum())
    if len(closed) == 0:
        return {"n_open": open_count, "n_closed": 0}
    return {
        "n_open": open_count,
        "n_closed": len(closed),
        "win_rate": round(float((closed["realized_return"] > 0).mean()), 3),
        "mean_return": round(float(closed["realized_return"].mean()), 5),
        "total_pnl_pct_equity": round(float(closed["realized_pnl_pct_equity"].sum()), 5),
    }