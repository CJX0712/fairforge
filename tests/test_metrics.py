"""Tests: fairness metric directionality, bounds and NaN conventions.

Author: 晨星
"""

import numpy as np
import pytest

from fairforge.metrics.fairness import (
    average_odds_diff,
    calibration_by_group,
    demographic_parity_diff,
    disparate_impact,
    equal_opportunity_diff,
    equalized_odds_diff,
    fairness_report,
    sanitize_metrics,
    theil_index,
)


def _perfect(n0=50, n1=50, seed=0):
    """Perfectly fair data: identical label pattern in both groups (exact equality)."""
    rng = np.random.default_rng(seed)
    pattern = (rng.random(n0) < 0.5).astype(int)
    A = np.array([0] * n0 + [1] * n1)
    y = np.concatenate([pattern, pattern])
    yhat = y.copy()
    return y, yhat, A


def test_perfect_fairness_gives_zero_all():
    y, yhat, A = _perfect()
    report = fairness_report(y, yhat, A, y_prob=np.array(y, dtype=float) * 0.8 + 0.1)
    for k, v in report.items():
        assert v == pytest.approx(0.0, abs=1e-12), f"{k} should be 0, got {v}"


def test_dp_diff_directionality():
    A = np.array([0] * 100 + [1] * 100)
    yhat = np.array([1] * 90 + [0] * 100 + [1] * 10)  # 90% vs 10% selection
    assert demographic_parity_diff(yhat, A) == pytest.approx(0.8)


def test_dp_diff_zero_when_equal_rates():
    A = np.array([0, 1, 0, 1])
    yhat = np.array([1, 1, 0, 0])
    assert demographic_parity_diff(yhat, A) == 0.0


def test_equal_opportunity_uses_tpr_only():
    # same TPR (1.0) in both groups, different FPR -> eo diff = 0
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    yhat = np.array([1, 1, 1, 0, 1, 1, 0, 0])
    A = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert equal_opportunity_diff(y, yhat, A) == pytest.approx(0.0)


def test_equalized_odds_is_max_form():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    yhat = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    A = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # group0: TPR=0.5 FPR=0.5 ; group1: TPR=0.5 FPR=0.0 -> diffs 0.0 / 0.5
    assert equalized_odds_diff(y, yhat, A) == pytest.approx(0.5)
    assert average_odds_diff(y, yhat, A) == pytest.approx(0.25)


def test_disparate_impact_symmetric_and_bounded():
    A = np.array([0] * 100 + [1] * 100)
    # unprivileged 80%, privileged 20% -> ratio = 4 -> sym 0.25 -> viol 0.75
    yhat = np.array([1] * 80 + [0] * 20 + [1] * 20 + [0] * 80)
    v = disparate_impact(yhat, A)
    assert 0.0 <= v <= 1.0
    assert v == pytest.approx(0.75)
    # mirrored case gives the same violation
    yhat_m = yhat[::-1]
    assert disparate_impact(yhat_m, A) == pytest.approx(0.75)


def test_theil_zero_for_constant_and_fair():
    A = np.array([0] * 5 + [1] * 5)
    assert theil_index(np.zeros(10), A) == 0.0
    assert theil_index(np.ones(10), A) == 0.0
    y, yhat, A2 = _perfect()
    assert theil_index(yhat, A2) == pytest.approx(0.0, abs=1e-12)


def test_theil_positive_for_between_group_gap():
    yhat = np.array([1] * 8 + [0] * 92)
    A = np.array([0] * 50 + [1] * 50)  # all positives concentrated in group 0
    assert theil_index(yhat, A) > 0.0


def test_nan_on_empty_group():
    y = np.array([1, 0, 1, 1])
    yhat = np.array([1, 0, 1, 0])
    A = np.array([1, 1, 1, 1])  # no unprivileged samples
    assert np.isnan(demographic_parity_diff(yhat, A))
    assert np.isnan(equal_opportunity_diff(y, yhat, A))
    assert np.isnan(calibration_by_group(y, yhat.astype(float), A))


def test_calibration_by_group_value():
    y = np.array([1, 1, 1, 1])
    p = np.array([0.8, 0.8, 0.5, 0.5])  # group0 gap 0.2, group1 gap 0.5
    A = np.array([0, 0, 1, 1])
    assert calibration_by_group(y, p, A) == pytest.approx(0.3)


def test_violations_bounded_unit_interval():
    rng = np.random.default_rng(7)
    A = rng.binomial(1, 0.5, 400)
    y = rng.binomial(1, 0.5, 400)
    yhat = rng.binomial(1, 0.5, 400)
    for v in [
        demographic_parity_diff(yhat, A),
        equalized_odds_diff(y, yhat, A),
        disparate_impact(yhat, A),
    ]:
        assert 0.0 <= v <= 1.0


def test_sanitize_metrics_nan_to_none():
    out = sanitize_metrics({"a": float("nan"), "b": 0.5})
    assert out == {"a": None, "b": 0.5}


def test_report_needs_prob_for_calibration():
    y, yhat, A = _perfect()
    report = fairness_report(y, yhat, A)
    assert np.isnan(report["calibration_by_group"])
