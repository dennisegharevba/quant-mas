"""
Portfolio Optimizer - Phase 13.

If multiple assets have attractive signals, do not simply sum their
individual position sizes - account for correlation. Uses Ledoit-Wolf
shrinkage covariance rather than raw sample covariance (noisy with
limited history).

SCOPE: adjusts position sizes AFTER the position sizer has already
produced individual recommendations - caps GROUP exposure for clusters
of correlated instruments rather than replacing per-instrument sizing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CORRELATION_LOOKBACK = 250
CORRELATION_CLUSTER_THRESHOLD = 0.70
MAX_CLUSTER_EXPOSURE = 0.03


def build_correlation_matrix(processed_dir: Path, tickers: list[str],
                              lookback: int = CORRELATION_LOOKBACK) -> pd.DataFrame:
    returns = {}
    for ticker in tickers:
        path = processed_dir / f"{ticker}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        close = df["Close"].astype(float)
        returns[ticker] = np.log(close / close.shift(1))

    if len(returns) < 2:
        return pd.DataFrame()

    panel = pd.DataFrame(returns).dropna(how="any").tail(lookback)
    if len(panel) < 30:
        return pd.DataFrame()

    lw = LedoitWolf().fit(panel.to_numpy())
    cov = lw.covariance_
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)

    return pd.DataFrame(corr, index=panel.columns, columns=panel.columns)


def identify_clusters(corr_matrix: pd.DataFrame, threshold: float = CORRELATION_CLUSTER_THRESHOLD) -> list[list[str]]:
    if corr_matrix.empty:
        return []

    tickers = list(corr_matrix.columns)
    visited = set()
    clusters = []

    def _neighbors(t: str) -> list[str]:
        return [other for other in tickers if other != t and abs(corr_matrix.loc[t, other]) > threshold]

    for t in tickers:
        if t in visited:
            continue
        stack = [t]
        component = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(_neighbors(node))
        visited |= component
        if len(component) > 1:
            clusters.append(sorted(component))

    return clusters


def adjust_for_correlation(candidates: list[dict], corr_matrix: pd.DataFrame,
                            max_cluster_exposure: float = MAX_CLUSTER_EXPOSURE) -> list[dict]:
    clusters = identify_clusters(corr_matrix)

    out = []
    adjusted_tickers = set()

    for cluster in clusters:
        cluster_candidates = [c for c in candidates if c["ticker"] in cluster]
        if len(cluster_candidates) < 2:
            continue

        total = sum(c["recommended_fraction"] for c in cluster_candidates)
        scale = min(1.0, max_cluster_exposure / total) if total > 0 else 1.0

        for c in cluster_candidates:
            adjusted = dict(c)
            adjusted["adjusted_fraction"] = c["recommended_fraction"] * scale
            adjusted["cluster"] = cluster
            adjusted["cluster_scaled"] = scale < 1.0
            out.append(adjusted)
            adjusted_tickers.add(c["ticker"])

    for c in candidates:
        if c["ticker"] not in adjusted_tickers:
            adjusted = dict(c)
            adjusted["adjusted_fraction"] = c["recommended_fraction"]
            adjusted["cluster"] = None
            adjusted["cluster_scaled"] = False
            out.append(adjusted)

    return out