"""Flagship orchestration: FairPareto across scenarios x constraints.

Targets: hypervolume vs strongest single algorithm >= +10% relative,
front coverage >= 80%, fallback events recorded. Actual numbers are reported
honestly whether or not the targets are met.

Author: 晨星
"""

import logging
import time

from fairforge.data.generator import SCENARIOS, generate_scenario
from fairforge.hpo.selector import FairParetoSelector
from fairforge.pipeline.benchmark import environment_fingerprint

logger = logging.getLogger("fairforge")

DEFAULT_CONSTRAINTS = ("demographic_parity_diff", "equal_opportunity_diff")
FLAGSHIP_ALGOS = [
    "none",
    "reweighing",
    "disparate_impact_remover",
    "exponentiated_gradient",
    "grid_search",
    "threshold_optimizer",
    "group_threshold",
    "reject_option",
]


def run_flagship(
    scenarios: list[str] | None = None,
    constraints: tuple[str, ...] = DEFAULT_CONSTRAINTS,
    seed: int = 42,
    n_per_group: int = 600,
    n_trials: int = 30,
    timeout_s: float = 90.0,
) -> dict:
    from fairforge.core.config import Config

    scenarios = list(scenarios or SCENARIOS)
    config = Config(seed=seed, n_per_group=n_per_group, n_trials=n_trials, timeout_s=timeout_s)
    started = time.monotonic()
    runs = []
    for scenario in scenarios:
        dataset = generate_scenario(scenario, n_per_group=n_per_group, seed=seed)
        for constraint in constraints:
            selector = FairParetoSelector(
                constraint=constraint, lambda_fair=1.0, config=config, algorithms=FLAGSHIP_ALGOS
            )
            res = selector.fit_select(dataset)
            runs.append(res.as_dict())
            logger.info(
                "flagship %s/%s: hv=%.4f single_hv=%.4f (+%.1f%%) cov=%.1f%% fb=%s trials=%d",
                scenario,
                constraint,
                res.hypervolume,
                res.single_best_hypervolume,
                100 * res.hv_improvement(),
                100 * res.coverage,
                res.fallback_event,
                res.n_trials_total,
            )

    n = len(runs)
    hv_improvements = [r["hv_improvement_rel"] for r in runs]
    coverages = [r["coverage"] for r in runs]
    fallbacks = [r["fallback_event"] for r in runs]
    mean_hv = float(sum(hv_improvements) / n) if n else 0.0
    mean_cov = float(sum(coverages) / n) if n else 0.0
    summary = {
        "n_runs": n,
        "mean_hv_improvement_rel": mean_hv,
        "mean_coverage": mean_cov,
        "fallback_events": int(sum(fallbacks)),
        "fallback_rate": float(sum(fallbacks) / n) if n else 0.0,
        "targets": {"hv_improvement_rel_min": 0.10, "coverage_min": 0.80},
        "targets_met": {"hv_improvement": mean_hv >= 0.10, "coverage": mean_cov >= 0.80},
        "runs": runs,
        "elapsed_s": round(time.monotonic() - started, 2),
        "environment": environment_fingerprint(),
    }
    return summary
