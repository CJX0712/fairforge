"""Unified fairness metrics.

CONVENTIONS (single source of truth, used by diagnostics, constraint targets,
HPO objectives and benchmark columns alike):
  * privileged group: A == 1, unprivileged: A == 0.
  * every metric is a *violation*: an absolute difference, in [0, 1]
    (or NaN when undefined), SMALLER = FAIRER, 0 = perfectly fair.
  * division by zero yields NaN; aggregations skip NaN.

Metrics:
  demographic_parity_diff  |P(yhat=1|A=0) - P(yhat=1|A=1)|
  equal_opportunity_diff   |TPR(A=0) - TPR(A=1)|
  equalized_odds_diff      max(|dTPR|, |dFPR|)
  average_odds_diff        (|dTPR| + |dFPR|) / 2
  disparate_impact         reported as 1 - DI where DI is made symmetric
                           (min(ratio, 1/ratio)) so violation stays in [0,1]
  theil_index              Theil T of the predicted favorable outcomes
  calibration_by_group     |mean calibration gap (A=1) - (A=0)|, needs y_prob,
                           NaN when a group is empty

Author: 晨星
"""

import numpy as np

FAIRNESS_METRICS = (
    "demographic_parity_diff",
    "equal_opportunity_diff",
    "equalized_odds_diff",
    "average_odds_diff",
    "disparate_impact",
    "theil_index",
    "calibration_by_group",
)


def _rate(y_pred: np.ndarray, mask: np.ndarray) -> float:
    n = int(mask.sum())
    if n == 0:
        return np.nan
    return float(np.mean(np.asarray(y_pred, dtype=float)[mask]))


def _safe_abs_diff(a: float, b: float) -> float:
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return float(abs(a - b))


def demographic_parity_diff(y_pred, A) -> float:
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    A = np.asarray(A).astype(int).ravel()
    return _safe_abs_diff(_rate(y_pred, A == 0), _rate(y_pred, A == 1))


def _tpr_fpr(y_true, y_pred, A):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    out = []
    for g in (0, 1):
        mg = A == g
        pos, neg = mg & (y_true == 1), mg & (y_true == 0)
        tpr = _rate(y_pred, pos) if pos.sum() else np.nan
        fpr = _rate(y_pred, neg) if neg.sum() else np.nan
        out.append((tpr, fpr))
    return out


def equal_opportunity_diff(y_true, y_pred, A) -> float:
    A = np.asarray(A).astype(int).ravel()
    (tpr0, _), (tpr1, _) = _tpr_fpr(y_true, y_pred, A)
    return _safe_abs_diff(tpr0, tpr1)


def equalized_odds_diff(y_true, y_pred, A) -> float:
    A = np.asarray(A).astype(int).ravel()
    (tpr0, fpr0), (tpr1, fpr1) = _tpr_fpr(y_true, y_pred, A)
    d_tpr = _safe_abs_diff(tpr0, tpr1)
    d_fpr = _safe_abs_diff(fpr0, fpr1)
    vals = [v for v in (d_tpr, d_fpr) if not np.isnan(v)]
    return float(max(vals)) if vals else np.nan


def average_odds_diff(y_true, y_pred, A) -> float:
    A = np.asarray(A).astype(int).ravel()
    (tpr0, fpr0), (tpr1, fpr1) = _tpr_fpr(y_true, y_pred, A)
    d_tpr = _safe_abs_diff(tpr0, tpr1)
    d_fpr = _safe_abs_diff(fpr0, fpr1)
    vals = [v for v in (d_tpr, d_fpr) if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def disparate_impact(y_pred, A) -> float:
    """Violation reported as ``1 - DI`` with DI symmetrized into [0, 1]."""
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    A = np.asarray(A).astype(int).ravel()
    r0, r1 = _rate(y_pred, A == 0), _rate(y_pred, A == 1)
    if np.isnan(r0) or np.isnan(r1) or r1 == 0.0:
        return np.nan
    ratio = r0 / r1
    sym = min(ratio, 1.0 / ratio)  # symmetrize so the violation lives in [0, 1]
    return float(1.0 - sym)


def theil_index(y_pred, A) -> float:
    """Between-group Theil T over predicted favorable outcomes.

    ``T = sum_g (n_g/n) * (mu_g/mu) * ln(mu_g/mu)`` with group means mu_g and
    overall mean mu. 0 when both groups have identical selection rates
    (perfect fairness); also 0 for constant (all-0) vectors. NaN-safe: empty
    groups are skipped.
    """
    p = np.asarray(y_pred, dtype=float).ravel()
    A = np.asarray(A).astype(int).ravel()
    n = len(p)
    if n == 0:
        return np.nan
    mu = float(np.mean(p))
    if mu <= 0.0:
        return 0.0
    total = 0.0
    for g in (0, 1):
        mg = A == g
        n_g = int(mg.sum())
        if n_g == 0:
            continue
        mu_g = float(np.mean(p[mg]))
        if mu_g > 0.0:
            r = mu_g / mu
            total += (n_g / n) * r * float(np.log(r))
    return float(total)


def calibration_by_group(y_true, y_prob, A) -> float:
    """|calibration gap difference| between groups; needs probabilistic scores."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    A = np.asarray(A).astype(int).ravel()
    gaps = []
    for g in (0, 1):
        mg = A == g
        if mg.sum() == 0:
            return np.nan
        gaps.append(abs(float(np.mean(y_prob[mg])) - float(np.mean(y_true[mg]))))
    return float(abs(gaps[0] - gaps[1]))


def fairness_report(y_true, y_pred, A, y_prob=None) -> dict[str, float]:
    """Compute ALL fairness metrics with the unified convention."""
    report = {
        "demographic_parity_diff": demographic_parity_diff(y_pred, A),
        "equal_opportunity_diff": equal_opportunity_diff(y_true, y_pred, A),
        "equalized_odds_diff": equalized_odds_diff(y_true, y_pred, A),
        "average_odds_diff": average_odds_diff(y_true, y_pred, A),
        "disparate_impact": disparate_impact(y_pred, A),
        "theil_index": theil_index(y_pred, A),
        "calibration_by_group": (
            calibration_by_group(y_true, y_prob, A) if y_prob is not None else np.nan
        ),
    }
    return report


def sanitize_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    """Convert NaN to None so metrics serialize cleanly to JSON."""
    return {
        k: (None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v))
        for k, v in metrics.items()
    }
