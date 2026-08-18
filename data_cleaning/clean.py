"""
Data cleaning + quality scoring - Phase 1.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.universe import Instrument, UNIVERSE

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]

# Major low-liquidity holidays where FX feeds are known to produce unreliable
# quotes (thin trading -> bad prints, not genuine data corruption). Dropped
# from FX series specifically before scoring, rather than loosening the
# violation tolerance globally.
FX_LOW_LIQUIDITY_HOLIDAYS = {(1, 1), (12, 25), (12, 26)}  # (month, day)


def _filter_fx_holidays(df: pd.DataFrame, asset_class: str) -> pd.DataFrame:
    if asset_class != "fx":
        return df
    mask = ~df.index.map(lambda d: (d.month, d.day) in FX_LOW_LIQUIDITY_HOLIDAYS)
    return df[mask]


@dataclass
class QualityReport:
    ticker: str
    total_rows: int
    date_gaps: int
    zero_volume_days: int
    price_outlier_days: int
    ohlc_violations: int
    duplicate_dates: int
    history_days: int
    meets_min_history: bool
    quality_score: float
    eligible: bool
    holiday_days_dropped: int = 0


def _detect_ohlc_violations(df: pd.DataFrame, tolerance: float = 0.0005) -> pd.Series:
    """
    Flag genuine OHLC violations, allowing a small relative tolerance for
    floating-point/rounding noise in the source feed (observed at ~0.003-0.01%
    magnitude - not real market violations).
    """
    tol = df["Close"] * tolerance
    return (
        (df["High"] < df["Low"] - tol)
        | (df["Close"] > df["High"] + tol)
        | (df["Close"] < df["Low"] - tol)
        | (df["Open"] > df["High"] + tol)
        | (df["Open"] < df["Low"] - tol)
    )


def _detect_date_gaps(df: pd.DataFrame, asset_class: str) -> int:
    if len(df) < 2:
        return 0
    freq = "C" if asset_class == "crypto" else "B"
    full_range = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    return max(0, len(full_range) - len(df))


def _detect_price_outliers(df: pd.DataFrame, threshold: float = 0.20) -> int:
    close = df["Close"].astype(float)
    log_ret = np.log(close / close.shift(1))
    return int((log_ret.abs() > threshold).sum())


def clean_one(instrument: Instrument) -> tuple[pd.DataFrame | None, QualityReport]:
    path = RAW_DIR / f"{instrument.ticker.replace('=', '_')}.csv"
    if not path.exists():
        return None, QualityReport(
            ticker=instrument.ticker, total_rows=0, date_gaps=0, zero_volume_days=0,
            price_outlier_days=0, ohlc_violations=0, duplicate_dates=0, history_days=0,
            meets_min_history=False, quality_score=0.0, eligible=False,
        )

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{instrument.ticker}: missing columns {missing_cols}")

    duplicate_dates = int(df.index.duplicated().sum())
    df = df[~df.index.duplicated(keep="first")].sort_index()

    rows_before_holiday_filter = len(df)
    df = _filter_fx_holidays(df, instrument.asset_class)
    holiday_days_dropped = rows_before_holiday_filter - len(df)

    ohlc_violations = int(_detect_ohlc_violations(df).sum())
    zero_volume_days = int((df["Volume"].fillna(0) == 0).sum()) if instrument.asset_class not in ("fx",) else 0
    date_gaps = _detect_date_gaps(df, instrument.asset_class)
    price_outlier_days = _detect_price_outliers(df)

    history_days = (df.index.max() - df.index.min()).days if len(df) > 0 else 0
    meets_min_history = len(df) >= instrument.min_history_days

    total_rows = len(df)
    score = 100.0
    if total_rows > 0:
        score -= min(30, 100 * date_gaps / max(total_rows, 1))
        score -= min(20, 100 * ohlc_violations / max(total_rows, 1))
        score -= min(15, 100 * zero_volume_days / max(total_rows, 1))
        score -= min(15, 100 * price_outlier_days / max(total_rows, 1))
        score -= 10 * min(1, duplicate_dates)
    else:
        score = 0.0
    score = max(0.0, round(score, 1))

    eligible = meets_min_history and score >= 70.0 and ohlc_violations == 0

    report = QualityReport(
        ticker=instrument.ticker,
        total_rows=total_rows,
        date_gaps=date_gaps,
        zero_volume_days=zero_volume_days,
        price_outlier_days=price_outlier_days,
        ohlc_violations=ohlc_violations,
        duplicate_dates=duplicate_dates,
        history_days=history_days,
        meets_min_history=meets_min_history,
        quality_score=score,
        eligible=eligible,
        holiday_days_dropped=holiday_days_dropped,
    )

    if eligible:
        out_path = PROCESSED_DIR / f"{instrument.ticker.replace('=', '_')}.csv"
        df.to_csv(out_path)

    return df, report


def clean_universe() -> list[QualityReport]:
    return [clean_one(inst)[1] for inst in UNIVERSE]


def print_quality_report(reports: list[QualityReport]) -> None:
    print(f"\n{'ticker':<12}{'rows':<8}{'gaps':<8}{'ohlc_v':<8}{'outl':<8}{'holiday':<9}{'score':<8}{'eligible'}")
    for r in reports:
        print(f"{r.ticker:<12}{r.total_rows:<8}{r.date_gaps:<8}{r.ohlc_violations:<8}"
              f"{r.price_outlier_days:<8}{r.holiday_days_dropped:<9}{r.quality_score:<8}{r.eligible}")
    n_eligible = sum(r.eligible for r in reports)
    print(f"\n{n_eligible}/{len(reports)} instruments eligible for modeling.")


if __name__ == "__main__":
    reports = clean_universe()
    print_quality_report(reports)