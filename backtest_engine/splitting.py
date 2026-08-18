"""
Purging/embargo evaluation split.

Addresses genuine cross-boundary information leakage between train and
test periods - a separate issue from the estimator bias found via Monte
Carlo simulation (that bias is a within-series artifact of overlapping-
window smoothing, not fixed by this).

What this fixes specifically: Model 3's rolling factor regression re-fits
coefficients continuously across the reporting train/test boundary with
no interruption. Each day's fit is causal (only data through t-1, no
future leak) - but the earliest test-period dates use coefficients fit
almost entirely from training-region data, so their forecasts aren't a
genuinely fresh out-of-sample start. The embargo gap - a buffer right
after the boundary, excluded from both training use and test scoring -
is the standard fix (Lopez de Prado, Advances in Financial Machine
Learning).
"""

from __future__ import annotations

import pandas as pd


def get_test_mask(index: pd.DatetimeIndex, test_fraction: float,
                   embargo_days: int) -> tuple[pd.Series, int, int]:
    n = len(index)
    split_idx = int(n * (1 - test_fraction))
    embargo_end_idx = min(split_idx + embargo_days, n)

    test_mask = pd.Series(False, index=index)
    test_mask.iloc[embargo_end_idx:] = True
    return test_mask, split_idx, embargo_end_idx