"""
NO-TRADE Filter - Phase 7.

Mandatory per the blueprint: the system must be able to reject every
candidate. Given experiments 1-5 (Model 1 - the surviving return
estimate - did not itself beat naive-zero in backtesting), this filter is
EXPECTED to reject most or all candidates most of the time. That is
correct behavior, not a bug to be tuned away.

Gates implemented: insufficient sample size, statistical insignificance
(bootstrap CI from Model 1 includes zero), EV after a default assumed
transaction cost <= 0.

Gates NOT implemented (explicitly, not faked): real bid/ask spread and
slippage data (a flat placeholder cost is used instead), liquidity
screening, correlation-exposure limits (needs the portfolio optimizer,
not yet built), regime-instability screening (needs the regime
classifier, not yet built).
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_COST_BPS = 5.0  # flat placeholder round-trip cost - NOT real spread/slippage data


@dataclass
class NoTradeResult:
    decision: str
    reasons: list[str] = field(default_factory=list)


def evaluate_no_trade(row: dict, cost_bps: float = DEFAULT_COST_BPS) -> NoTradeResult:
    reasons = []

    if row.get("date") is None or row.get("n_samples", 0) == 0:
        return NoTradeResult("NO TRADE", ["no valid forecast available for this instrument/horizon"])

    if row["n_samples"] < 60:
        reasons.append(f"insufficient sample size ({row['n_samples']} < 60)")

    if not row.get("ci_excludes_zero", False):
        reasons.append("statistically insignificant - bootstrap CI includes zero")

    cost_frac = cost_bps / 10000.0
    ev_after_cost = row["expected_return"] - cost_frac if row["expected_return"] > 0 else row["expected_return"] + cost_frac
    if row["expected_return"] > 0 and ev_after_cost <= 0:
        reasons.append(f"expected value does not survive assumed {cost_bps}bps cost")
    elif row["expected_return"] < 0 and ev_after_cost >= 0:
        reasons.append(f"expected value does not survive assumed {cost_bps}bps cost")

    decision = "NO TRADE" if reasons else "TRADE"
    return NoTradeResult(decision, reasons)