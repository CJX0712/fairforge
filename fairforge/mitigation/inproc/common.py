"""Shared helpers for in-processing backends."""

import numpy as np


def dp_gap(y_pred, A) -> float:
    """Signed DP gap: selection_rate(A=0) - selection_rate(A=1)."""
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    A = np.asarray(A).astype(int).ravel()
    r0 = float(np.mean(y_pred[A == 0])) if (A == 0).any() else np.nan
    r1 = float(np.mean(y_pred[A == 1])) if (A == 1).any() else np.nan
    return float(r0 - r1)


def tpr_gap(y_true, y_pred, A) -> float:
    """Signed TPR gap: TPR(A=0) - TPR(A=1) (NaN when a group has no positives)."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    A = np.asarray(A).astype(int).ravel()
    tprs = []
    for g in (0, 1):
        pos = (A == g) & (y_true == 1)
        tprs.append(float(np.mean(y_pred[pos])) if pos.any() else np.nan)
    return float(tprs[0] - tprs[1])


def signed_gap(constraint: str, y_true, y_pred, A) -> float:
    """Signed gap for the requested constraint (dp: selection-rate, eo: TPR)."""
    if constraint == "equal_opportunity_diff":
        return tpr_gap(y_true, y_pred, A)
    return dp_gap(y_pred, A)


def violation(constraint: str, y_true, y_pred, A) -> float:
    """Violation (absolute gap) in [0, 1] for the requested constraint."""
    g = signed_gap(constraint, y_true, y_pred, A)
    return float("nan") if np.isnan(g) else abs(g)
