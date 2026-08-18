"""
Smoke test for the Phase 1 pipeline. Generates synthetic OHLCV with
deliberately injected problems (gap, outlier, OHLC violation, zero-volume
day) to validate the cleaning logic without needing live network access.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.universe import get
from data_cleaning.clean import clean_one, print_quality_report

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def make_synthetic_ohlcv(ticker: str, n_days: int = 600, seed: int = 42,
                          inject_gap: bool = True, inject_outlier: bool = True,
                          inject_ohlc_violation: bool = True,
                          inject_zero_volume: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    if inject_gap:
        dates = dates.delete(slice(200, 210))

    n = len(dates)
    rets = rng.normal(0.0003, 0.015, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0.001, 0.01, n))
    low = close * (1 - rng.uniform(0.001, 0.01, n))
    open_ = low + (high - low) * rng.uniform(0.3, 0.7, n)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)

    if inject_outlier:
        close[300] *= 1.35

    if inject_ohlc_violation:
        high[350] = low[350] - 1

    if inject_zero_volume:
        volume[400] = 0

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    df.index.name = "date"
    return df


def run_smoke_test():
    inst = get("AAPL")
    synthetic = make_synthetic_ohlcv(inst.ticker)
    out_path = RAW_DIR / f"{inst.ticker}.csv"
    synthetic.to_csv(out_path)

    _, report = clean_one(inst)
    print_quality_report([report])

    assert report.date_gaps >= 9, "should detect the injected ~10-day gap"
    assert report.ohlc_violations >= 1, "should detect the injected OHLC violation"
    assert report.price_outlier_days >= 1, "should detect the injected 35% jump"
    assert report.zero_volume_days == 1, "should detect the injected zero-volume day"
    assert report.eligible is False, "record with an OHLC violation must not be marked eligible"

    print("\nAll smoke-test assertions passed.")


if __name__ == "__main__":
    run_smoke_test()