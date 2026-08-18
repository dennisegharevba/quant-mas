"""
Position Sizer - Phase 8.

Several sizing modes for research (fixed fractional, volatility targeting,
Kelly, fractional Kelly), defaulting to FRACTIONAL Kelly - never full
Kelly - for anything actually sized.

WHY fractional Kelly, not full Kelly:
  1. Estimation error - P(win) and win/loss magnitudes are noisy finite-
     sample estimates (and per experiments 1-4, often not statistically
     distinguishable from zero edge at all). Full Kelly assumes these are
     known exactly.
  2. Fat tails - Kelly assumes a well-behaved return distribution;
     financial returns are not.
  3. Drawdown tolerance - full Kelly produces drawdowns most retail
     accounts cannot survive even when the edge is real. Fractional Kelly
     trades some growth-rate optimality for a large reduction in outcome
     variance - the right trade for low capital.

Kelly fraction: f* = (p*b - (1-p)) / b, where p = P(win),
b = mean_win / |mean_loss|
"""

from __future__ import annotations

import numpy as np

DEFAULT_KELLY_FRACTION = 0.25
DEFAULT_MAX_RISK_PER_TRADE = 0.02
MIN_WIN_LOSS_SAMPLES = 5


def get_win_loss_magnitudes(resolved_returns: np.ndarray) -> tuple[float, float]:
    clean = resolved_returns[~np.isnan(resolved_returns)]
    wins = clean[clean > 0]
    losses = clean[clean < 0]
    mean_win = wins.mean() if len(wins) >= MIN_WIN_LOSS_SAMPLES else np.nan
    mean_loss = losses.mean() if len(losses) >= MIN_WIN_LOSS_SAMPLES else np.nan
    return float(mean_win), float(mean_loss)


def kelly_fraction(prob_win: float, mean_win: float, mean_loss: float) -> float:
    if np.isnan(prob_win) or np.isnan(mean_win) or np.isnan(mean_loss) or mean_loss >= 0 or mean_win <= 0:
        return np.nan
    b = mean_win / abs(mean_loss)
    f_star = (prob_win * b - (1 - prob_win)) / b
    return f_star


def fractional_kelly(f_star: float, k: float = DEFAULT_KELLY_FRACTION) -> float:
    if np.isnan(f_star):
        return np.nan
    return max(0.0, k * f_star)


def fixed_fractional(risk_per_trade: float = DEFAULT_MAX_RISK_PER_TRADE) -> float:
    return risk_per_trade


def volatility_target(target_vol: float, asset_vol: float, max_leverage: float = 1.0) -> float:
    if asset_vol <= 0 or np.isnan(asset_vol):
        return 0.0
    return min(max_leverage, target_vol / asset_vol)


def recommend_position_size(
    prob_win: float, mean_win: float, mean_loss: float,
    kelly_k: float = DEFAULT_KELLY_FRACTION,
    max_risk_per_trade: float = DEFAULT_MAX_RISK_PER_TRADE,
) -> dict:
    f_star = kelly_fraction(prob_win, mean_win, mean_loss)
    f_fractional = fractional_kelly(f_star, k=kelly_k)

    if np.isnan(f_fractional):
        return {
            "kelly_full": np.nan, "kelly_fractional": np.nan,
            "recommended_fraction": 0.0,
            "capped_by_max_risk": False,
            "note": "insufficient win/loss sample to compute Kelly - defaulting to 0",
        }

    capped = f_fractional > max_risk_per_trade
    recommended = min(f_fractional, max_risk_per_trade)

    return {
        "kelly_full": round(f_star, 4),
        "kelly_fractional": round(f_fractional, 4),
        "recommended_fraction": round(recommended, 4),
        "capped_by_max_risk": capped,
        "note": (f"capped at max_risk_per_trade={max_risk_per_trade:.1%}"
                 if capped else "fractional Kelly not capped"),
    }