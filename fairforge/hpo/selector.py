"""FairPareto Auto-Selector - the FairForge flagship.

Pipeline per (dataset, constraint):
  1. Baseline: plain logistic regression -> (accuracy, violation).
  2. For EVERY algorithm family, run Optuna (n_trials<=30, timeout fuse) on
     ``maximize accuracy - lambda * violation`` and keep all trial points.
  3. Build the accuracy-fairness Pareto front over all points.
  4. Non-inferiority guard: the recommended point must satisfy
     ``violation <= baseline_violation + 0.01``; otherwise fall back to the
     safest (minimum-violation) front point and record the fallback event.
  5. Score the front: hypervolume vs the strongest single algorithm's own
     front, and grid coverage (fraction of the normalized [0,1]^2 objective
     space dominated by the front).

Targets (reported honestly whether met or not):
  * hypervolume >= +10% relative vs strongest single algorithm
  * coverage >= 80%

Author: 晨星
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

from fairforge.core.config import Config
from fairforge.core.errors import MitigatorError
from fairforge.core.types import Dataset
from fairforge.hpo.tuner import Tuner
from fairforge.metrics.fairness import demographic_parity_diff, equal_opportunity_diff
from fairforge.metrics.performance import accuracy as accuracy_metric

logger = logging.getLogger("fairforge")

CONSTRAINT_FN = {
    # unified signature (y_true, y_pred, A) -> violation in [0, 1]
    "demographic_parity_diff": lambda y_true, y_pred, A: demographic_parity_diff(y_pred, A),
    "equal_opportunity_diff": equal_opportunity_diff,
}

# Per-algorithm HPO spaces (all small, CPU-friendly). Constraint ranges are
# deliberately wide so trials spread across the whole violation spectrum and
# the union front covers the accuracy-fairness trade-off densely.
ALGO_SPACES: dict[str, dict] = {
    "reweighing": {},  # no hyperparameters -> single deterministic trial
    "none": {},  # unmitigated baseline candidate
    "disparate_impact_remover": {"repair_level": ("float", 0.05, 1.0)},
    "exponentiated_gradient": {
        "eps": ("float", 0.02, 0.3),
        "eta": ("float", 0.2, 1.5),
        "max_iter": ("int", 5, 20),
    },
    "grid_search": {
        "eps": ("float", 0.02, 0.3),
        "lambda_fair": ("float", 0.1, 5.0),
        "grid_size": ("categorical", [4, 6, 8]),
    },
    "threshold_optimizer": {
        "eps": ("float", 0.02, 0.35),
        "grid_size": ("categorical", [10, 20, 50]),
    },
    "group_threshold": {"grid_size": ("categorical", [10, 20, 50])},
    "reject_option": {
        "lambda_fair": ("float", 0.1, 5.0),
        "theta_steps": ("categorical", [5, 10, 15]),
    },
}


@dataclass
class ParetoPoint:
    algorithm: str
    params: dict
    accuracy: float
    violation: float
    backend: str = "native"
    used_fallback: bool = True

    def as_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "params": self.params,
            "accuracy": float(self.accuracy),
            "violation": float(self.violation),
            "backend": self.backend,
            "used_fallback": self.used_fallback,
        }


@dataclass
class FairParetoResult:
    constraint: str
    scenario: str
    baseline_accuracy: float
    baseline_violation: float
    points: list[ParetoPoint] = field(default_factory=list)
    pareto_front: list[ParetoPoint] = field(default_factory=list)
    recommended: ParetoPoint | None = None
    fallback_event: bool = False
    hypervolume: float = 0.0
    single_best_hypervolume: float = 0.0
    coverage: float = 0.0
    n_trials_total: int = 0
    elapsed_s: float = 0.0
    metadata_single_best: str | None = None

    def hv_improvement(self) -> float:
        if self.single_best_hypervolume <= 0:
            return 0.0
        return self.hypervolume / self.single_best_hypervolume - 1.0

    def as_dict(self) -> dict:
        return {
            "constraint": self.constraint,
            "scenario": self.scenario,
            "baseline_accuracy": float(self.baseline_accuracy),
            "baseline_violation": float(self.baseline_violation),
            "pareto_front": [p.as_dict() for p in self.pareto_front],
            "recommended": self.recommended.as_dict() if self.recommended else None,
            "fallback_event": self.fallback_event,
            "hypervolume": float(self.hypervolume),
            "single_best_hypervolume": float(self.single_best_hypervolume),
            "hv_improvement_rel": float(self.hv_improvement()),
            "coverage": float(self.coverage),
            "n_trials_total": self.n_trials_total,
            "elapsed_s": float(self.elapsed_s),
            "single_best_algorithm": self.metadata_single_best,
        }


def pareto_front_mask(acc: np.ndarray, viol: np.ndarray) -> np.ndarray:
    """Indices of non-dominated points (maximize acc, minimize viol)."""
    acc = np.asarray(acc, dtype=float)
    viol = np.asarray(viol, dtype=float)
    keep = np.ones(len(acc), dtype=bool)
    for i in range(len(acc)):
        if not keep[i]:
            continue
        dominated = (acc >= acc[i]) & (viol <= viol[i]) & ((acc > acc[i]) | (viol < viol[i]))
        if dominated.any():
            keep[i] = False
    # drop within-front duplicates (same acc & viol)
    idx = np.where(keep)[0]
    seen, final = set(), []
    for i in idx:
        key = (round(float(acc[i]), 12), round(float(viol[i]), 12))
        if key not in seen:
            seen.add(key)
            final.append(int(i))
    return np.array(final, dtype=int)


def hypervolume_2d(front_acc: np.ndarray, front_viol: np.ndarray, ref=(1.0, 1.0)) -> float:
    """Exact 2-D hypervolume. Maximize acc, minimize viol; ref point (r_acc, r_viol).

    Internally converted to minimization: u = viol, v = 1 - acc. Front points
    sorted by u ascending contribute rectangles u in [u_i, u_{i+1}] (last one
    up to r_u) of height (r_v - v_i).
    """
    acc = np.asarray(front_acc, dtype=float)
    viol = np.asarray(front_viol, dtype=float)
    if len(acc) == 0:
        return 0.0
    # re-derive the front to guarantee an antichain
    idx = pareto_front_mask(acc, viol)
    u, v = viol[idx], 1.0 - acc[idx]
    order = np.argsort(u, kind="stable")
    u, v = u[order], v[order]
    r_u, r_v = float(ref[1]), float(ref[0])  # (viol_ref, acc_ref) -> (u_ref, v_ref)
    area = 0.0
    n = len(u)
    for i in range(n):
        u_next = min(float(u[i + 1]), r_u) if i + 1 < n else r_u
        if u[i] >= r_u:
            break
        area += (u_next - float(u[i])) * (r_v - float(v[i]))
    return float(max(area, 0.0))


def front_coverage(front_acc: np.ndarray, front_viol: np.ndarray, grid: int = 10) -> float:
    """Fraction of grid cells in the normalized objective space dominated by the front.

    Cell (i, j) center = (viol=(i+0.5)/grid, acc=1 - (j+0.5)/grid); dominated if some
    front point has viol <= cell viol and acc >= cell acc.
    """
    acc = np.asarray(front_acc, dtype=float)
    viol = np.asarray(front_viol, dtype=float)
    if len(acc) == 0:
        return 0.0
    dominated = 0
    total = grid * grid
    for i in range(grid):
        cv = (i + 0.5) / grid
        for j in range(grid):
            ca = 1.0 - (j + 0.5) / grid
            if np.any((viol <= cv) & (acc >= ca)):
                dominated += 1
    return dominated / total


class FairParetoSelector:
    """Full 'algorithm x hyperparameter' sweep -> Pareto front -> guarded recommendation."""

    def __init__(
        self,
        constraint: str = "demographic_parity_diff",
        lambda_fair: float = 1.0,
        config: Config | None = None,
        algorithms: list[str] | None = None,
        noninferiority_slack: float = 0.01,
    ) -> None:
        if constraint not in CONSTRAINT_FN:
            raise ValueError(
                f"unknown constraint {constraint!r}; available: {sorted(CONSTRAINT_FN)}"
            )
        self.constraint = constraint
        self.lambda_fair = float(lambda_fair)
        self.config = config or Config()
        self.algorithms = algorithms or sorted(ALGO_SPACES)
        self.slack = float(noninferiority_slack)

    # -- objective ---------------------------------------------------------
    def _evaluate(self, dataset: Dataset, algorithm: str, params: dict):
        from fairforge.mitigation.base import create

        merged = dict(params)
        if "lambda_fair" not in merged:
            merged["lambda_fair"] = self.lambda_fair
        try:
            mitigator = create(
                algorithm,
                backend="auto",
                random_state=self.config.seed,
                constraint=self.constraint,
                **merged,
            )
        except MitigatorError:
            # algorithm does not accept lambda_fair -> drop it and retry
            merged.pop("lambda_fair", None)
            mitigator = create(
                algorithm,
                backend="auto",
                random_state=self.config.seed,
                constraint=self.constraint,
                **merged,
            )
        mitigator.fit(dataset)
        if algorithm in ("reweighing", "disparate_impact_remover"):
            # pre-processing: train a plain LR on the mitigated representation
            from sklearn.linear_model import LogisticRegression

            res = mitigator.transform(dataset)
            clf = LogisticRegression(max_iter=1000, random_state=self.config.seed)
            if res.sample_weight is not None:
                clf.fit(dataset.X, dataset.y, sample_weight=res.sample_weight)
            else:
                clf.fit(res.X, dataset.y)
            pred = clf.predict(dataset.X).astype(int)
        else:
            pred = mitigator.predict(dataset.X, dataset.A)
        acc = accuracy_metric(dataset.y, pred)
        viol = float(CONSTRAINT_FN[self.constraint](dataset.y, pred, dataset.A))
        if np.isnan(viol):
            viol = 1.0
        return acc, viol, mitigator.resolved_backend, mitigator.used_fallback

    def _baseline(self, dataset: Dataset):
        clf = LogisticRegression(max_iter=1000, random_state=self.config.seed)
        clf.fit(dataset.X, dataset.y)
        pred = clf.predict(dataset.X).astype(int)
        acc = accuracy_metric(dataset.y, pred)
        viol = float(CONSTRAINT_FN[self.constraint](dataset.y, pred, dataset.A))
        return float(acc), float(viol if not np.isnan(viol) else 1.0)

    # -- main entry --------------------------------------------------------
    def fit_select(self, dataset: Dataset) -> FairParetoResult:
        start = time.monotonic()
        base_acc, base_viol = self._baseline(dataset)
        result = FairParetoResult(
            constraint=self.constraint,
            scenario=dataset.scenario,
            baseline_accuracy=base_acc,
            baseline_violation=base_viol,
        )

        tuner = Tuner(self.config)
        per_algo_trials: dict[str, list] = {}
        for algo in self.algorithms:
            space = ALGO_SPACES.get(algo, {})

            def objective(params, _algo=algo):
                try:
                    acc, viol, be, fb = self._evaluate(dataset, _algo, params)
                except Exception as exc:
                    logger.info("trial failed for %s %s: %s", _algo, params, exc)
                    return -1.0  # worst possible objective
                result.points.append(ParetoPoint(_algo, params, acc, viol, be, fb))
                return acc - self.lambda_fair * viol

            if not space:
                params = {}
                value = objective(params)
                tuner.trials.append({"params": params, "value": value})
                per_algo_trials[algo] = list(tuner.trials)
            else:
                per_algo_trials[algo] = tuner.tune(
                    objective, space, n_trials=self.config.n_trials, timeout_s=self.config.timeout_s
                )
        result.n_trials_total = sum(len(v) for v in per_algo_trials.values())

        if not result.points:
            result.elapsed_s = time.monotonic() - start
            return result

        acc = np.array([p.accuracy for p in result.points])
        viol = np.array([p.violation for p in result.points])
        front_idx = pareto_front_mask(acc, viol)
        result.pareto_front = [result.points[i] for i in front_idx]

        # hypervolume: flagship front vs strongest single algorithm's front
        result.hypervolume = hypervolume_2d(acc[front_idx], viol[front_idx])
        best_algo, best_algo_hv, best_algo_obj = None, 0.0, -np.inf
        for algo in per_algo_trials:
            t_acc = np.array([p.accuracy for p in result.points if p.algorithm == algo])
            t_viol = np.array([p.violation for p in result.points if p.algorithm == algo])
            if len(t_acc) == 0:
                continue
            hv = hypervolume_2d(t_acc, t_viol)
            objs = t_acc - self.lambda_fair * t_viol
            if objs.max() > best_algo_obj:
                best_algo_obj, best_algo, best_algo_hv = float(objs.max()), algo, hv
        result.single_best_hypervolume = best_algo_hv
        result.metadata_single_best = best_algo

        # recommendation + non-inferiority guard
        front_acc = acc[front_idx]
        front_viol = viol[front_idx]
        objs = front_acc - self.lambda_fair * front_viol
        rec_i = int(np.argmax(objs))
        if front_viol[rec_i] <= base_viol + self.slack:
            result.recommended = result.pareto_front[rec_i]
            result.fallback_event = False
        else:
            safe_i = int(np.argmin(front_viol))
            result.recommended = result.pareto_front[safe_i]
            result.fallback_event = True
            logger.info(
                "non-inferiority guard triggered (%.3f > baseline %.3f + %.2f) -> safe point",
                front_viol[rec_i],
                base_viol,
                self.slack,
            )

        result.coverage = front_coverage(front_acc, front_viol)
        result.elapsed_s = time.monotonic() - start
        return result
