"""
Model 4 - Probability Model - Phase 5.

Regularized logistic regression estimating calibrated P(R_h > 0 | X_t),
combined with separate magnitude sub-models (trailing mean of resolved
positive/negative outcomes - same mechanism as Model 1, stratified by
realized sign). Probability and magnitude are modeled and combined
separately, never conflated into one figure.

Walk-forward: the classifier is refit every REFIT_EVERY trading days using
only training rows whose label has already resolved as of the refit date.
A training example at date t' pairs X_all[t'] with label_all[t'] directly
(today's features -> future outcome) - the point-in-time safety comes from
which t' are ELIGIBLE at "today" (row i): only t' + horizon <= i, i.e.
where that outcome has already happened and is knowable as of today.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HORIZONS = [1, 3, 5, 10]
DEFAULT_LOOKBACK = 250
MIN_SAMPLES = 60
TRAIN_WINDOW = 400
REFIT_EVERY = 20
MIN_TRAIN_SAMPLES = 100

FEATURE_COLS = [
    "feature_trailing_mean_20d", "feature_trailing_std_20d", "feature_zscore_20d", "feature_rolling_vol_ann_20d",
    "feature_trailing_mean_60d", "feature_trailing_std_60d", "feature_zscore_60d", "feature_rolling_vol_ann_60d",
    "feature_trailing_mean_120d", "feature_trailing_std_120d", "feature_zscore_120d", "feature_rolling_vol_ann_120d",
    "feature_autocorr_lag1_60d",
]


def forecast_series(features: pd.DataFrame, horizon: int,
                     lookback: int = DEFAULT_LOOKBACK) -> pd.DataFrame:
    resolved_col = f"feature_resolved_forward_return_{horizon}d"
    label_col = f"target_forward_positive_{horizon}d"
    for col in FEATURE_COLS + [resolved_col, label_col]:
        if col not in features.columns:
            raise KeyError(f"{col} not found - did you run build_features first?")

    n = len(features)
    X_all = features[FEATURE_COLS].to_numpy()
    resolved = features[resolved_col].to_numpy()
    label_all = features[label_col].to_numpy()

    exp_ret = np.full(n, np.nan)
    prob_pos = np.full(n, np.nan)
    n_samples = np.zeros(n, dtype=int)

    model = None
    scaler = None
    last_refit = -REFIT_EVERY - 1

    for i in range(n):
        lo = max(0, i - lookback + 1)
        window_resolved = resolved[lo: i + 1]
        window_resolved = window_resolved[~np.isnan(window_resolved)]
        n_samples[i] = len(window_resolved)
        if len(window_resolved) == 0:
            continue

        pos_mask = window_resolved > 0
        neg_mask = window_resolved < 0
        mean_win = window_resolved[pos_mask].mean() if pos_mask.sum() >= 5 else np.nan
        mean_loss = window_resolved[neg_mask].mean() if neg_mask.sum() >= 5 else np.nan
        empirical_prob = pos_mask.mean()

        if i - last_refit >= REFIT_EVERY:
            train_hi = i - horizon
            train_lo = max(0, train_hi - TRAIN_WINDOW + 1)
            if train_hi >= train_lo:
                X_train = X_all[train_lo: train_hi + 1]
                y_train = label_all[train_lo: train_hi + 1]
                valid_train = ~np.isnan(y_train) & ~np.isnan(X_train).any(axis=1)
                if valid_train.sum() >= MIN_TRAIN_SAMPLES and len(np.unique(y_train[valid_train])) == 2:
                    try:
                        scaler = StandardScaler()
                        Xs = scaler.fit_transform(X_train[valid_train])
                        model = LogisticRegression(C=1.0, max_iter=1000)
                        model.fit(Xs, y_train[valid_train])
                    except Exception:
                        model = None
                        scaler = None
                else:
                    model = None
                    scaler = None
            else:
                model = None
                scaler = None
            last_refit = i

        if model is not None and not np.isnan(X_all[i]).any():
            try:
                p = float(model.predict_proba(scaler.transform(X_all[i:i + 1]))[0, 1])
            except Exception:
                p = empirical_prob
        else:
            p = empirical_prob

        prob_pos[i] = p

        if not np.isnan(mean_win) and not np.isnan(mean_loss):
            exp_ret[i] = p * mean_win + (1 - p) * mean_loss
        elif not np.isnan(mean_win):
            exp_ret[i] = p * mean_win
        elif not np.isnan(mean_loss):
            exp_ret[i] = (1 - p) * mean_loss

    return pd.DataFrame(
        {
            "expected_return": exp_ret,
            "prob_positive": prob_pos,
            "n_samples": n_samples,
            "sufficient_sample": n_samples >= MIN_SAMPLES,
        },
        index=features.index,
    )