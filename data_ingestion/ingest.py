"""
Data ingestion - Phase 1.
Pulls daily OHLCV for every instrument in the universe and writes raw,
untouched data to data/raw/{ticker}.csv.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.universe import UNIVERSE, Instrument

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class IngestResult:
    ticker: str
    success: bool
    rows: int
    start: str | None
    end: str | None
    error: str | None = None


def fetch_one(instrument: Instrument, period: str = "5y") -> IngestResult:
    try:
        df = yf.download(
            instrument.ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if df is None or df.empty:
            return IngestResult(instrument.ticker, False, 0, None, None,
                                 error="empty response from source")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index.name = "date"
        out_path = RAW_DIR / f"{instrument.ticker.replace('=', '_')}.csv"
        df.to_csv(out_path)

        return IngestResult(
            ticker=instrument.ticker,
            success=True,
            rows=len(df),
            start=str(df.index.min().date()),
            end=str(df.index.max().date()),
        )
    except Exception as e:
        return IngestResult(instrument.ticker, False, 0, None, None, error=str(e))


def fetch_universe(period: str = "5y") -> list[IngestResult]:
    return [fetch_one(inst, period=period) for inst in UNIVERSE]


def print_report(results: list[IngestResult]) -> None:
    print(f"\nIngestion run: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'ticker':<12}{'status':<10}{'rows':<8}{'start':<12}{'end':<12}error")
    for r in results:
        status = "OK" if r.success else "FAIL"
        print(f"{r.ticker:<12}{status:<10}{r.rows:<8}{r.start or '-':<12}{r.end or '-':<12}{r.error or ''}")
    n_ok = sum(r.success for r in results)
    print(f"\n{n_ok}/{len(results)} instruments ingested successfully.")


if __name__ == "__main__":
    results = fetch_universe()
    print_report(results)