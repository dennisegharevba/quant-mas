"""
Monte Carlo Engine - Phase 9.

Simulate thousands of possible future paths, report probability of
profit/loss, drawdown distribution, losing-streak distribution,
probability of ruin, terminal equity distribution.

SCOPE, stated honestly: simulates holding a position sized at a given
fraction of equity in ONE instrument, using block-bootstrap resampling of
historical daily returns (preserving volatility clustering - a naive
i.i.d. resample would understate risk). Does NOT simulate a full multi-
trade strategy with entries/exits, or a multi-asset portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_SIMULATIONS = 5000
DEFAULT_SIM_DAYS = 60
DEFAULT_BLOCK_LEN = 20
RUIN_THRESHOLD = 0.50


@dataclass
class MonteCarloResult:
    n_simulations: int
    n_days: int
    position_fraction: float
    prob_profit: float
    prob_loss: float
    terminal_equity_pct: dict
    max_drawdown_pct: dict
    losing_streak_days_pct: dict
    prob_of_ruin: float


def _block_bootstrap_path(daily_returns: np.ndarray, n_days: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    n = len(daily_returns)
    block_len = min(block_len, n)
    n_blocks_needed = int(np.ceil(n_days / block_len))
    starts = rng.integers(0, n - block_len + 1, size=n_blocks_needed)
    path = np.concatenate([daily_returns[s:s + block_len] for s in starts])[:n_days]
    return path


def run_monte_carlo(
    daily_returns: np.ndarray,
    position_fraction: float,
    n_days: int = DEFAULT_SIM_DAYS,
    n_simulations: int = N_SIMULATIONS,
    block_len: int = DEFAULT_BLOCK_LEN,
    ruin_threshold: float = RUIN_THRESHOLD,
    seed: int = 0,
) -> MonteCarloResult:
    rng = np.random.default_rng(seed)
    clean = daily_returns[~np.isnan(daily_returns)]

    terminal_equities = np.empty(n_simulations)
    max_drawdowns = np.empty(n_simulations)
    losing_streaks = np.empty(n_simulations, dtype=int)

    for s in range(n_simulations):
        path_returns = _block_bootstrap_path(clean, n_days, block_len, rng)
        equity_curve = np.cumprod(1 + position_fraction * path_returns)

        terminal_equities[s] = equity_curve[-1]

        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdowns[s] = drawdown.min()

        is_losing_day = path_returns < 0
        max_run = 0
        current_run = 0
        for is_loss in is_losing_day:
            if is_loss:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        losing_streaks[s] = max_run

    prob_profit = float((terminal_equities > 1.0).mean())
    prob_loss = float((terminal_equities < 1.0).mean())
    prob_of_ruin = float((terminal_equities <= ruin_threshold).mean())

    def _pctiles(arr: np.ndarray) -> dict:
        return {f"p{p}": round(float(np.percentile(arr, p)), 5) for p in (5, 25, 50, 75, 95)}

    return MonteCarloResult(
        n_simulations=n_simulations,
        n_days=n_days,
        position_fraction=position_fraction,
        prob_profit=round(prob_profit, 4),
        prob_loss=round(prob_loss, 4),
        terminal_equity_pct=_pctiles(terminal_equities),
        max_drawdown_pct=_pctiles(max_drawdowns),
        losing_streak_days_pct=_pctiles(losing_streaks.astype(float)),
        prob_of_ruin=round(prob_of_ruin, 4),
    )


def print_mc_report(result: MonteCarloResult, ticker: str) -> None:
    print(f"\n--- Monte Carlo stress test: {ticker} ---")
    print(f"Position size: {result.position_fraction:.1%} of equity | "
          f"Horizon: {result.n_days} trading days | Simulations: {result.n_simulations}")
    print(f"P(profit): {result.prob_profit:.1%}  |  P(loss): {result.prob_loss:.1%}  |  "
          f"P(ruin, equity<={RUIN_THRESHOLD:.0%}): {result.prob_of_ruin:.2%}")
    print(f"Terminal equity (1.00 = breakeven) percentiles: {result.terminal_equity_pct}")
    print(f"Max drawdown percentiles (within the {result.n_days}-day window): {result.max_drawdown_pct}")
    print(f"Longest losing-day-streak percentiles: {result.losing_streak_days_pct}")