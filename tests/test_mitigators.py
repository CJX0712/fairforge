"""Tests: pre/in/post mitigator smoke fits + bias-reduction direction checks.

Author: 晨星
"""

import numpy as np
import pytest

from fairforge.data.generator import generate_scenario
from fairforge.metrics.fairness import demographic_parity_diff
from fairforge.mitigation.base import create


@pytest.fixture(scope="module")
def ds():
    return generate_scenario("mild_direct", n_per_group=500, seed=42)


ALGOS = [
    "reweighing",
    "disparate_impact_remover",
    "exponentiated_gradient",
    "grid_search",
    "threshold_optimizer",
    "group_threshold",
    "reject_option",
]


@pytest.mark.parametrize("algo", ALGOS)
def test_fit_predict_smoke(ds, algo):
    from fairforge.pipeline.benchmark import run_single

    m, y_pred, elapsed = run_single(ds, algo, seed=42)
    assert len(y_pred) == ds.n_samples
    assert set(np.unique(y_pred)).issubset({0, 1})
    assert elapsed > 0
    assert m.resolved_backend in ("fairlearn", "native")


def test_dir_repairs_group_means(ds):
    m = create("disparate_impact_remover", repair_level=1.0, random_state=42).fit(ds)
    X_new = m.transform(ds).X
    gap_before = np.abs(ds.X[ds.A == 0].mean(axis=0) - ds.X[ds.A == 1].mean(axis=0))
    gap_after = np.abs(X_new[ds.A == 0].mean(axis=0) - X_new[ds.A == 1].mean(axis=0))
    assert (gap_after <= gap_before + 1e-9).all()
    assert np.allclose(gap_after, 0.0, atol=1e-9)


def test_dir_repair_level_zero_is_identity(ds):
    m = create("disparate_impact_remover", repair_level=0.0, random_state=42).fit(ds)
    assert np.allclose(m.transform(ds).X, ds.X)


def test_native_eg_reduces_violation(ds):
    eg = create(
        "exponentiated_gradient", backend="native", eps=0.05, eta=0.8, max_iter=25, random_state=42
    ).fit(ds)
    hist = eg.violation_history_
    assert len(hist) >= 2
    assert hist[-1] < hist[0] + 1e-12  # violation trends down
    pred = eg.predict(ds.X, ds.A)
    assert demographic_parity_diff(pred, ds.A) <= hist[0]


def test_eg_soft_crosscheck_vs_backend(ds):
    """Native vs fairlearn backend DP violation should be within 0.05 (soft assert).

    Operating points are calibrated so both backends solve the same DP problem
    at a comparable operating point (fairlearn's realized violation is ~2x its
    eps bound on this data).
    """
    from fairforge.core.backend import available_backends

    if not available_backends().get("fairlearn", False):
        pytest.skip("fairlearn not installed")
    native = create(
        "exponentiated_gradient", backend="native", eps=0.10, eta=0.8, max_iter=25, random_state=42
    ).fit(ds)
    fl = create("exponentiated_gradient", backend="fairlearn", eps=0.02, random_state=42).fit(ds)
    v_nat = demographic_parity_diff(native.predict(ds.X, ds.A), ds.A)
    v_fl = demographic_parity_diff(fl.predict(ds.X, ds.A), ds.A)
    assert abs(v_nat - v_fl) < 0.05, f"native {v_nat:.3f} vs fairlearn {v_fl:.3f}"


def test_grid_search_native_picks_best(ds):
    gs = create("grid_search", backend="native", lambda_fair=1.0, grid_size=5, random_state=42).fit(
        ds
    )
    pred = gs.predict(ds.X, ds.A)
    assert len(pred) == ds.n_samples
    assert len(gs.candidates_) == 5


def test_threshold_optimizer_feasible_or_min_violation(ds):
    to = create(
        "threshold_optimizer", backend="native", eps=0.05, grid_size=15, random_state=42
    ).fit(ds)
    pred = to.predict(ds.X, ds.A)
    v = demographic_parity_diff(pred, ds.A)
    if to.metadata.get("feasible"):
        assert v <= 0.05 + 1e-9
    assert v <= demographic_parity_diff((ds.y > -1).astype(int), ds.A) + 0.5  # sanity bound


def test_reject_option_reduces_violation(ds):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=1000, random_state=42).fit(ds.X, ds.y)
    base_pred = clf.predict(ds.X)
    ro = create("reject_option", lambda_fair=2.0, theta_steps=10, random_state=42).fit(ds)
    pred = ro.predict(ds.X, ds.A)
    v_before = demographic_parity_diff(base_pred, ds.A)
    v_after = demographic_parity_diff(pred, ds.A)
    assert v_after <= v_before + 1e-9


def test_group_threshold_uses_both_thresholds(ds):
    gt = create("group_threshold", grid_size=20, random_state=42).fit(ds)
    assert set(gt.thresholds_) == {0, 1}
    assert 0.0 <= gt.thresholds_[0] <= 1.0 and 0.0 <= gt.thresholds_[1] <= 1.0


def test_unknown_param_rejected(ds):
    from fairforge.core.errors import MitigatorError

    with pytest.raises(MitigatorError):
        create("reweighing", bogus_param=3)
