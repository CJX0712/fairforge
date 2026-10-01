"""Tests: backend detection, degradation matrix (monkeypatched missing deps), config, errors.

Author: 晨星
"""

import json

import numpy as np
import pytest

from fairforge.core import backend as backend_mod
from fairforge.core.backend import available_backends, get_backend, reset_cache
from fairforge.core.config import Config
from fairforge.core.errors import ConfigError, MitigatorError
from fairforge.data.generator import generate_scenario


@pytest.fixture(autouse=True)
def _fresh_cache():
    reset_cache()
    yield
    reset_cache()


def test_available_backends_structure():
    b = available_backends()
    assert set(b) == {"fairlearn", "optuna"}
    assert all(isinstance(v, bool) for v in b.values())


def test_get_backend_auto_resolution():
    has = available_backends()["fairlearn"]
    resolved, fallback = get_backend("fairlearn", "auto")
    assert (
        (resolved == "fairlearn" and not fallback) if has else (resolved == "native" and fallback)
    )


def test_get_backend_forced_native():
    resolved, fallback = get_backend("fairlearn", "native")
    assert resolved == "native" and fallback is True


def test_get_backend_invalid_request():

    with pytest.raises(ValueError):
        get_backend("fairlearn", "torch")


def test_degradation_without_fairlearn(monkeypatch):
    """Simulate a machine without fairlearn: detection must report native fallback."""
    monkeypatch.setattr(backend_mod, "_probe", lambda name: False)
    b = available_backends()
    assert b["fairlearn"] is False and b["optuna"] is False
    resolved, fallback = get_backend("fairlearn", "auto")
    assert resolved == "native" and fallback is True


def test_degradation_end_to_end_without_fairlearn(monkeypatch):
    monkeypatch.setattr(backend_mod, "_probe", lambda name: False)
    ds = generate_scenario("mixed", n_per_group=300, seed=42)
    from fairforge.pipeline.benchmark import run_single

    m, y_pred, _ = run_single(ds, "threshold_optimizer", seed=42)
    assert m.resolved_backend == "native"
    assert m.used_fallback is True
    assert len(y_pred) == ds.n_samples


def test_degradation_tuner_without_optuna(monkeypatch):
    monkeypatch.setattr(backend_mod, "_probe", lambda name: False)
    from fairforge.hpo.tuner import Tuner

    tuner = Tuner(Config(seed=42))
    trials = tuner.tune(lambda p: p["x"], {"x": ("float", 0.0, 1.0)}, n_trials=5)
    assert len(trials) == 5
    assert all(0.0 <= t["value"] <= 1.0 for t in trials)


def test_config_env_overrides(monkeypatch):
    monkeypatch.setenv("FAIRFORGE_SEED", "7")
    monkeypatch.setenv("FAIRFORGE_LAMBDA_FAIRNESS", "2.5")
    cfg = Config.from_env()
    assert cfg.seed == 7 and cfg.lambda_fairness == 2.5


def test_config_bad_env_raises(monkeypatch):
    monkeypatch.setenv("FAIRFORGE_SEED", "abc")
    with pytest.raises(ConfigError):
        Config.from_env()


def test_config_n_jobs_locked():
    with pytest.raises(ConfigError):
        Config(n_jobs=4)


def test_error_codes():
    from fairforge.core.errors import BackendError, DataError, MetricError

    assert str(ConfigError("x")).startswith("[E100]")
    assert str(DataError("x")).startswith("[E200]")
    assert str(MetricError("x")).startswith("[E300]")
    assert str(MitigatorError("x")).startswith("[E400]")
    assert str(BackendError("x")).startswith("[E500]")


def test_json_roundtrip_nan_free():
    from fairforge.metrics.fairness import fairness_report, sanitize_metrics

    rng = np.random.default_rng(3)
    A = rng.binomial(1, 0.5, 200)
    y = rng.binomial(1, 0.5, 200)
    yhat = rng.binomial(1, 0.5, 200)
    blob = json.dumps(sanitize_metrics(fairness_report(y, yhat, A)))
    assert "NaN" not in blob
