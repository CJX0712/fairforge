"""Benchmark runner: scenarios x mitigators -> JSON-ready entries.

Every entry records: accuracy, ALL fairness metrics, backend|fallback flag,
n_trials, elapsed, seed, dependency versions and an environment fingerprint.

Author: 晨星
"""

import logging
import platform
import sys
import time

import numpy as np
import sklearn

from fairforge.core.backend import available_backends, backend_version
from fairforge.core.types import Dataset
from fairforge.data.generator import SCENARIOS, generate_scenario
from fairforge.metrics.fairness import fairness_report, sanitize_metrics
from fairforge.metrics.performance import accuracy as accuracy_metric
from fairforge.mitigation.base import create
from fairforge.mitigation.inproc.exponentiated_gradient import (
    ExponentiatedGradient,  # noqa: F401 (registers)
)
from fairforge.mitigation.inproc.grid_search import GridSearchReductions  # noqa: F401
from fairforge.mitigation.post.group_threshold import GroupThreshold  # noqa: F401
from fairforge.mitigation.post.reject_option import RejectOption  # noqa: F401
from fairforge.mitigation.post.threshold_optimizer import ThresholdOptimizer  # noqa: F401
from fairforge.mitigation.pre.disparate_impact_remover import DisparateImpactRemover  # noqa: F401
from fairforge.mitigation.pre.reweighing import Reweighing  # noqa: F401

logger = logging.getLogger("fairforge")

DEFAULT_MITIGATORS = [
    "reweighing",
    "exponentiated_gradient",
    "grid_search",
    "threshold_optimizer",
    "group_threshold",
    "reject_option",
]


def environment_fingerprint() -> dict:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "fairlearn": backend_version("fairlearn"),
        "optuna": backend_version("optuna"),
        "backends_available": available_backends(),
    }


def dependency_versions() -> dict:
    return {"numpy": np.__version__, "scikit_learn": sklearn.__version__}


def run_single(dataset: Dataset, mitigator: str, seed: int = 42, backend: str = "auto", **params):
    """Fit + predict one mitigator on a dataset. Returns (mitigator, y_pred, elapsed)."""
    start = time.monotonic()
    m = create(mitigator, backend=backend, random_state=seed, **params)
    m.fit(dataset)
    if mitigator in ("reweighing", "disparate_impact_remover"):
        from sklearn.linear_model import LogisticRegression

        res = m.transform(dataset)
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        if res.sample_weight is not None:
            clf.fit(dataset.X, dataset.y, sample_weight=res.sample_weight)
        else:
            clf.fit(res.X, dataset.y)
        y_pred = clf.predict(dataset.X).astype(int)
    else:
        y_pred = m.predict(dataset.X, dataset.A)
    return m, y_pred, time.monotonic() - start


def run_benchmark(
    scenarios: list[str] | None = None,
    mitigators: list[str] | None = None,
    seed: int = 42,
    n_per_group: int = 600,
) -> list[dict]:
    """Run the full benchmark matrix and return JSON-ready entries."""
    scenarios = list(scenarios or SCENARIOS)
    mitigators = list(mitigators or DEFAULT_MITIGATORS)
    entries: list[dict] = []
    for scenario in scenarios:
        dataset = generate_scenario(scenario, n_per_group=n_per_group, seed=seed)
        for mitigator in mitigators:
            m, y_pred, elapsed = run_single(dataset, mitigator, seed=seed)
            acc = accuracy_metric(dataset.y, y_pred)
            metrics = sanitize_metrics(fairness_report(dataset.y, y_pred, dataset.A))
            entries.append(
                {
                    "scenario": scenario,
                    "mitigator": mitigator,
                    "backend": m.resolved_backend,
                    "used_fallback": m.used_fallback,
                    "accuracy": float(acc),
                    "metrics": metrics,
                    "n_trials": m.n_trials,
                    "elapsed_s": round(elapsed, 4),
                    "seed": seed,
                    "dependency_versions": dependency_versions(),
                    "environment_fingerprint": environment_fingerprint(),
                }
            )
            logger.info(
                "benchmark %s/%s: acc=%.3f dp=%.3f backend=%s fallback=%s",
                scenario,
                mitigator,
                acc,
                metrics["demographic_parity_diff"],
                m.resolved_backend,
                m.used_fallback,
            )
    return entries


def save_benchmark(entries: list[dict], path, extra: dict | None = None) -> None:
    """Write benchmark entries (plus optional extra payload) as JSON."""
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"entries": entries, "environment": environment_fingerprint()}
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("benchmark written to %s", path)
