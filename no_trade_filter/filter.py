"""
NO-TRADE Filter - Phase 7, revised per the architecture correction and
Phase 17's measured transaction costs.

TRANSACTION COST: cost used is max(documented-typical asset-class figure,
Corwin-Schultz MEASURED spread estimate from the instrument's own OHLC
history) - never below the documented floor, but using the instrument's
own measured spread when it's wider than the class-level assumption. The
measured estimator has a small, known upward bias at very low true
spreads (validated via synthetic testing) - since that bias runs toward
OVERstating cost, it's the safe direction for a cost gate to be biased in.
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


def get_cost_bps(asset_class: str | None, measured_spread_bps: float | None = None) -> float:
    documented = DEFAULT_COST_BPS if asset_class is None else ASSET_CLASS_COST_BPS.get(asset_class, DEFAULT_COST_BPS)
    if measured_spread_bps is None:
        return documented
    return max(documented, measured_spread_bps)


def evaluate_no_trade(row: dict, cost_bps: float | None = None) -> NoTradeResult:
    reasons = []

    if row.get("date") is None or row.get("n_samples", 0) == 0:
        return NoTradeResult("NO TRADE", ["no valid forecast available for this instrument/horizon"], 0.0)

    if cost_bps is None:
        cost_bps = get_cost_bps(row.get("asset_class"), row.get("measured_spread_bps"))

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