"""
Multiple-Testing Diagnostics - Phase 14.

PBO (Probability of Backtest Overfitting) and the Deflated Sharpe Ratio
exist to address selection bias when many models/features are tried.
These are DIAGNOSTICS, not pass/fail gates.

APPROACH: a direct, conservative multiple-testing correction on the
win-rate metrics already used throughout every ablation test - a
binomial test against the null (50% win rate = no real edge), Bonferroni-
adjusted two ways: by the FULL ledger search budget (most conservative -
treats every combination ever tested, including unrelated tests, as
competing to produce this result) and by a SCOPED budget of only the
genuinely comparable alternatives actually tried on the same asset class
(more honest for a single, principled, hypothesis-driven test).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ledger import load_ledger


def total_search_budget() -> int:
    df = load_ledger()
    if len(df) == 0 or "n_combinations_tested" not in df.columns:
        return 0
    return int(df["n_combinations_tested"].fillna(0).sum())


def evaluate_finding(n_wins: int, n_trials: int, finding_name: str = "",
                      null_rate: float = 0.5, scoped_budget: int | None = None) -> dict:
    if n_trials == 0:
        return {"finding": finding_name, "error": "n_trials is 0"}

    win_rate = n_wins / n_trials
    raw_pvalue = binomtest(n_wins, n_trials, null_rate, alternative="greater").pvalue

    full_budget = total_search_budget()
    full_adjusted = min(1.0, raw_pvalue * max(full_budget, 1))

    def _verdict(p: float) -> str:
        if p < 0.01:
            return "Survives convincingly - very unlikely to be selection luck."
        elif p < 0.05:
            return "Survives, but not by a wide margin - worth continued monitoring."
        elif p < 0.20:
            return "Borderline - plausible this is partly or wholly selection luck."
        else:
            return "Does NOT survive - consistent with selection luck given the search size."

    result = {
        "finding": finding_name,
        "n_wins": n_wins, "n_trials": n_trials, "win_rate": round(win_rate, 3),
        "raw_pvalue": round(raw_pvalue, 5),
        "full_ledger_budget": full_budget,
        "full_ledger_adjusted_pvalue": round(full_adjusted, 5),
        "full_ledger_verdict": _verdict(full_adjusted),
    }

    if scoped_budget is not None:
        scoped_adjusted = min(1.0, raw_pvalue * max(scoped_budget, 1))
        result["scoped_budget"] = scoped_budget
        result["scoped_adjusted_pvalue"] = round(scoped_adjusted, 5)
        result["scoped_verdict"] = _verdict(scoped_adjusted)

    return result


def print_diagnostic_report(findings: list[dict]) -> None:
    budget = total_search_budget()
    print(f"\n{'='*90}")
    print(f"MULTIPLE-TESTING DIAGNOSTIC REPORT")
    print(f"Full-ledger search budget (sum of n_combinations_tested, entire project): {budget}")
    print(f"{'='*90}\n")
    for f in findings:
        if "error" in f:
            print(f"{f['finding']}: ERROR - {f['error']}")
            continue
        print(f"{f['finding']}")
        print(f"  {f['n_wins']}/{f['n_trials']} wins ({f['win_rate']:.1%})  |  raw p-value: {f['raw_pvalue']:.5f}")
        print(f"  Full-ledger correction (x{f['full_ledger_budget']}): p={f['full_ledger_adjusted_pvalue']:.5f} "
              f"-> {f['full_ledger_verdict']}")
        if "scoped_budget" in f:
            print(f"  Scoped correction (x{f['scoped_budget']}, comparable alternatives only): "
                  f"p={f['scoped_adjusted_pvalue']:.5f} -> {f['scoped_verdict']}")
        print()


if __name__ == "__main__":
    findings = [
        evaluate_finding(45, 45, "Model 5 (EWMA volatility) - universal, all asset classes",
                          scoped_budget=45),
        evaluate_finding(8, 8, "Model 1 (baseline) - index asset class only",
                          scoped_budget=8),
        evaluate_finding(11, 12, "Model 7 (equity momentum) - stock asset class",
                          scoped_budget=12 * 2),
        evaluate_finding(4, 4, "Model 6 (seasonality) - CL_F specifically",
                          scoped_budget=4 * 2),
    ]
    print_diagnostic_report(findings)