"""Tests: biased data generator - shapes, counterfactuals, bias effects, reproducibility.

Author: 晨星
"""

import numpy as np
import pytest

from fairforge.core.errors import DataError
from fairforge.data.generator import SCENARIOS, generate_biased_data, generate_scenario
from fairforge.metrics.fairness import demographic_parity_diff


def test_all_scenarios_exist():
    assert set(SCENARIOS) == {"mild_direct", "strong_sampling", "mixed"}


@pytest.mark.parametrize("scenario", ["mild_direct", "strong_sampling", "mixed"])
def test_scenario_shapes_and_groups(scenario):
    ds = generate_scenario(scenario, n_per_group=500, seed=42)
    assert ds.n_samples == 1000
    assert (ds.A == 0).sum() == 500
    assert (ds.A == 1).sum() == 500
    assert ds.y_cf is not None
    assert len(ds.y_cf) == ds.n_samples


def test_seed_reproducibility():
    a = generate_scenario("mixed", n_per_group=300, seed=42)
    b = generate_scenario("mixed", n_per_group=300, seed=42)
    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.y, b.y)
    assert np.array_equal(a.y_cf, b.y_cf)


def test_counterfactual_is_less_biased_than_observed():
    ds = generate_scenario("mild_direct", n_per_group=800, seed=42)
    viol_observed = demographic_parity_diff(ds.y, ds.A)
    viol_cf = demographic_parity_diff(ds.y_cf, ds.A)
    assert viol_cf < viol_observed


def test_strong_sampling_drops_unprivileged_favorables():
    ds = generate_scenario("strong_sampling", n_per_group=500, seed=42)
    # unprivileged favorable share should be much lower than privileged
    share0 = ds.y_cf[ds.A == 0].mean()
    share1 = ds.y_cf[ds.A == 1].mean()
    assert share0 < share1


def test_direct_discrimination_raises_privileged_positive_rate():
    unbiased = generate_biased_data(n_per_group=600, seed=42, delta=0.0)
    biased = generate_biased_data(n_per_group=600, seed=42, delta=1.5)
    assert biased.y[biased.A == 1].mean() > unbiased.y[unbiased.A == 1].mean()


def test_invalid_params_raise():
    with pytest.raises(DataError):
        generate_biased_data(sampling_bias=1.5)
    with pytest.raises(DataError):
        generate_biased_data(delta=99.0)
    with pytest.raises(DataError):
        generate_scenario("nope")


def test_dataset_validation_errors():
    from fairforge.core.types import Dataset

    with pytest.raises(DataError):
        Dataset(X=np.zeros((3, 2)), A=np.array([0, 1, 2]), y=np.array([0, 1, 1]))
    with pytest.raises(DataError):
        Dataset(X=np.zeros((3, 2)), A=np.array([0, 1, 1]), y=np.array([0, 1]))
