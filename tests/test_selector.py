"""Tests: flagship FairPareto - Pareto front math, hypervolume, coverage, non-inferiority guard.

Author: 晨星
"""

import numpy as np
import pytest

from fairforge.hpo.selector import (
    FairParetoSelector,
    front_coverage,
    hypervolume_2d,
    pareto_front_mask,
)


def test_pareto_front_mask_basics():
    acc = np.array([0.8, 0.9, 0.7, 0.9])
    viol = np.array([0.3, 0.3, 0.1, 0.3])
    idx = set(pareto_front_mask(acc, viol).tolist())
    assert idx == {1, 2}  # point3 dominated by point1; point0 dominated by point1


def test_pareto_dominance_ties():
    acc = np.array([0.8, 0.8])
    viol = np.array([0.2, 0.2])
    assert len(pareto_front_mask(acc, viol)) == 1


def test_hypervolume_empty_and_single():
    assert hypervolume_2d(np.array([]), np.array([])) == 0.0
    hv = hypervolume_2d(np.array([0.8]), np.array([0.2]))
    assert hv == pytest.approx((1.0 - 0.2) * 0.8)  # (r_u - viol) * acc


def test_hypervolume_known_case():
    # two-point front: (acc .9, viol .4), (acc .7, viol .2)
    hv = hypervolume_2d(np.array([0.9, 0.7]), np.array([0.4, 0.2]))
    # sorted by u: (.2, v=.3), (.4, v=.1); contributions [u_i, u_{i+1}] x (r_v - v_i):
    expected = (0.4 - 0.2) * (1.0 - 0.3) + (1.0 - 0.4) * (1.0 - 0.1)
    assert hv == pytest.approx(expected)


def test_hypervolume_increases_with_better_front():
    hv1 = hypervolume_2d(np.array([0.8]), np.array([0.3]))
    hv2 = hypervolume_2d(np.array([0.9]), np.array([0.2]))
    assert hv2 > hv1


def test_coverage_bounds():
    cov = front_coverage(np.array([0.9, 0.6]), np.array([0.1, 0.5]))
    assert 0.0 <= cov <= 1.0
    assert cov > 0.0


def test_flagship_end_to_end_small():
    """Small but real flagship run on one scenario; structural + sanity checks."""
    from fairforge.data.generator import generate_scenario

    ds = generate_scenario("mixed", n_per_group=300, seed=42)
    sel = FairParetoSelector(
        constraint="demographic_parity_diff",
        lambda_fair=1.0,
        algorithms=["reweighing", "group_threshold", "reject_option", "threshold_optimizer"],
    )
    # shrink trial budget for test speed
    from fairforge.core.config import Config

    sel.config = Config(seed=42, n_trials=5, timeout_s=60.0)
    res = sel.fit_select(ds)
    assert res.n_trials_total >= 4
    assert len(res.pareto_front) >= 1
    assert res.recommended is not None
    assert res.recommended.violation <= res.baseline_violation + 0.01 + 1e-9 or res.fallback_event
    assert 0.0 <= res.coverage <= 1.0
    d = res.as_dict()
    assert {
        "pareto_front",
        "recommended",
        "hv_improvement_rel",
        "coverage",
        "fallback_event",
    } <= set(d)


def test_noninferiority_guard_fallback():
    """Synthetic front where the best-objective point violates the guard -> fallback."""
    acc = np.array([0.95, 0.85])
    viol = np.array([0.6, 0.05])
    baseline_viol = 0.3
    slack = 0.01
    lam = 0.1  # accuracy-heavy objective so the high-violation point wins the objective
    front_idx = pareto_front_mask(acc, viol)
    front_viol = viol[front_idx]
    front_acc = acc[front_idx]
    objs = front_acc - lam * front_viol
    rec_i = int(np.argmax(objs))
    fallback = bool(front_viol[rec_i] > baseline_viol + slack)
    assert fallback is True  # 0.6 > 0.31
    safe_i = int(np.argmin(front_viol))
    assert front_viol[safe_i] == pytest.approx(0.05)


def test_flagship_invalid_constraint():
    with pytest.raises(ValueError):
        FairParetoSelector(constraint="bogus")
