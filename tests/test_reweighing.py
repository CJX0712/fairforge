"""Tests: Reweighing weight-sum property + bias reduction effect.

Author: 晨星
"""

import pytest
from sklearn.linear_model import LogisticRegression

from fairforge.core.errors import MitigatorError
from fairforge.data.generator import generate_scenario
from fairforge.metrics.fairness import demographic_parity_diff
from fairforge.mitigation.base import create
from fairforge.mitigation.pre.reweighing import Reweighing


@pytest.fixture(scope="module")
def ds():
    return generate_scenario("mild_direct", n_per_group=500, seed=42)


def test_weight_sum_equals_n(ds):
    rw = Reweighing(random_state=42).fit(ds)
    w = rw.transform(ds).sample_weight
    assert w.sum() == pytest.approx(ds.n_samples, rel=1e-9)


def test_weights_positive(ds):
    rw = Reweighing(random_state=42).fit(ds)
    w = rw.transform(ds).sample_weight
    assert (w > 0).all()


def test_weights_table_shape(ds):
    rw = Reweighing(random_state=42).fit(ds)
    assert set(rw.weights_table_) == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_reweighing_equalizes_conditional_positive_rates(ds):
    """After reweighing, P(y=1 | A=g) weighted must equal P(y=1) in both groups."""
    rw = Reweighing(random_state=42).fit(ds)
    w = rw.transform(ds).sample_weight
    rates = []
    for g in (0, 1):
        mg = ds.A == g
        rates.append(w[mg & (ds.y == 1)].sum() / w[mg].sum())
    assert abs(rates[0] - rates[1]) < 1e-9


def test_reweighed_model_reduces_violation(ds):
    base = LogisticRegression(max_iter=1000, random_state=42).fit(ds.X, ds.y)
    pred0 = base.predict(ds.X)
    rw = Reweighing(random_state=42).fit(ds)
    w = rw.transform(ds).sample_weight
    clf = LogisticRegression(max_iter=1000, random_state=42).fit(ds.X, ds.y, sample_weight=w)
    pred1 = clf.predict(ds.X)
    assert demographic_parity_diff(pred1, ds.A) <= demographic_parity_diff(pred0, ds.A) + 1e-9


def test_predict_not_supported(ds):
    rw = Reweighing(random_state=42).fit(ds)
    with pytest.raises(MitigatorError):
        rw.predict(ds.X, ds.A)


def test_registry_lookup():
    from fairforge.mitigation.base import list_mitigators

    assert "reweighing" in list_mitigators()
    assert create("reweighing", random_state=42) is not None
    with pytest.raises(MitigatorError):
        create("does_not_exist")
