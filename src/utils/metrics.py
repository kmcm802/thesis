"""Basic regression metrics used throughout the analysis scripts."""

from __future__ import annotations

import numpy as np


def compute_rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_nrmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    rng = float(np.max(y_true) - np.min(y_true))
    return compute_rmse(y_true, y_pred) / rng if rng > 1e-12 else float("nan")


def compute_r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def compute_mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))
