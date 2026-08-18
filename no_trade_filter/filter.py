"""
NO-TRADE Filter - Phase 7, revised per the architecture correction.

TRANSACTION COST, stated honestly: the per-class figures below are
DOCUMENTED TYPICAL round-trip cost assumptions from well-known market
convention, NOT measured from this system's own data. A genuine data-
driven estimate would be a further, separately-validated upgrade.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ASSET_CLASS_COST_BPS = {
    "fx": 1.5,
    "index": 3.0,
    "stock": 5.0,
    "commodity": 7.0,
    "crypto": 12.0,
}
DEFAULT_COST_BPS = 5.0


@dataclass
class NoTradeResult:
    decision: str
    reasons: list[str] = field(default_factory=list)
    cost_bps_used: float = 0.0


def get_cost_bps(asset_class: str | None) -> float:
    if asset_class is None:
        return DEFAULT_COST_BPS
    return ASSET_CLASS_COST_BPS.get(asset_class, DEFAULT_COST_BPS)


def evaluate_no_trade(row: dict, cost_bps: float | None = None) -> NoTradeResult:
    reasons = []

    if row.get("date") is None or row.get("n_samples", 0) == 0:
        return NoTradeResult("NO TRADE", ["no valid forecast available for this instrument/horizon"], 0.0)

    if cost_bps is None:
        cost_bps = get_cost_bps(row.get("asset_class"))

    if row["n_samples"] < 60:
        reasons.append(f"insufficient sample size ({row['n_samples']} < 60)")

    if not row.get("ci_excludes_zero", False):
        reasons.append("statistically insignificant - bootstrap CI includes zero")

    cost_frac = cost_bps / 10000.0
    ev_after_cost = row["expected_return"] - cost_frac if row["expected_return"] > 0 else row["expected_return"] + cost_frac
    if row["expected_return"] > 0 and ev_after_cost <= 0:
        reasons.append(f"expected value does not survive assumed {cost_bps}bps cost ({row.get('asset_class', 'unknown')})")
    elif row["expected_return"] < 0 and ev_after_cost >= 0:
        reasons.append(f"expected value does not survive assumed {cost_bps}bps cost ({row.get('asset_class', 'unknown')})")

    decision = "NO TRADE" if reasons else "TRADE"
    return NoTradeResult(decision, reasons, cost_bps)