"""Threshold Optimizer (constraint-aware).

Backend: fairlearn.postprocessing.ThresholdOptimizer (explicit random_state;
constraints demographic_parity / equalized_odds). Fallback: per-group
threshold grid search maximizing accuracy subject to the requested
constraint's violation <= eps; minimum-violation pair if none is feasible.

Author: 晨星
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from fairforge.core.types import Dataset
from fairforge.metrics.fairness import demographic_parity_diff, equal_opportunity_diff
from fairforge.metrics.performance import accuracy as accuracy_metric
from fairforge.mitigation.base import Mitigator, MitigatorResult, register

FAIRLEARN_CONSTRAINTS = {
    "demographic_parity_diff": "demographic_parity",
    "equal_opportunity_diff": "equalized_odds",
}


def _violation(constraint: str, y, pred, A) -> float:
    if constraint == "equal_opportunity_diff":
        return equal_opportunity_diff(y, pred, A)
    return demographic_parity_diff(pred, A)


def _grid(n_steps: int) -> np.ndarray:
    return np.linspace(0.02, 0.98, int(n_steps))


@register("threshold_optimizer")
class ThresholdOptimizer(Mitigator):
    def valid_params(self):
        return {"eps", "grid_size"}

    def __init__(self, eps: float = 0.05, grid_size: int = 20, **kw):
        super().__init__(eps=eps, grid_size=grid_size, **kw)

    def fit(self, dataset: Dataset) -> "ThresholdOptimizer":
        self.resolve_backend("fairlearn")
        if self.resolved_backend == "fairlearn":
            try:
                from fairlearn.postprocessing import ThresholdOptimizer as FairlearnTO

                est = FairlearnTO(
                    estimator=LogisticRegression(max_iter=1000, random_state=self.random_state),
                    constraints=FAIRLEARN_CONSTRAINTS.get(self.constraint, "demographic_parity"),
                    objective="accuracy_score",
                    grid_size=200,
                    predict_method="predict",
                    random_state=self.random_state,
                )
                est.fit(dataset.X, dataset.y, sensitive_features=dataset.A)
                self.model_ = est
                self.scorer_ = None
                self._fitted = True
                return self
            except Exception:
                self.used_fallback = True
                self.resolved_backend = "native"
        self._fit_native(dataset)
        return self

    def _fit_native(self, dataset: Dataset) -> None:
        X, A, y = dataset.X, dataset.A, dataset.y
        eps = self.params["eps"]
        constraint = self.constraint
        grid = _grid(self.params["grid_size"])
        self.scorer_ = LogisticRegression(max_iter=1000, random_state=self.random_state).fit(X, y)
        scores = self.scorer_.predict_proba(X)[:, 1]
        best_feasible, best_any = None, None
        for t0 in grid:
            p0 = (scores[A == 0] > t0).astype(int)
            for t1 in grid:
                p1 = (scores[A == 1] > t1).astype(int)
                pred = np.empty_like(y, dtype=int)
                pred[A == 0] = p0
                pred[A == 1] = p1
                acc = accuracy_metric(y, pred)
                viol = _violation(constraint, y, pred, A)
                cand = (acc, viol, float(t0), float(t1))
                if best_any is None or (acc, -viol) > (best_any[0], -best_any[1]):
                    best_any = cand
                if viol <= eps and (best_feasible is None or acc > best_feasible[0]):
                    best_feasible = cand
        chosen = best_feasible if best_feasible is not None else best_any
        self.thresholds_ = {0: chosen[2], 1: chosen[3]}
        self.metadata = {
            "feasible": best_feasible is not None,
            "thresholds": dict(self.thresholds_),
        }
        self._fitted = True

    def predict(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        A = np.asarray(A).astype(int).ravel()
        if self.resolved_backend == "fairlearn":
            return np.asarray(
                self.model_.predict(X, sensitive_features=A, random_state=self.random_state),
                dtype=int,
            )
        scores = self.scorer_.predict_proba(X)[:, 1]
        pred = np.empty(len(A), dtype=int)
        for g, t in self.thresholds_.items():
            pred[A == g] = (scores[A == g] > t).astype(int)
        return pred

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if self.resolved_backend == "fairlearn":
            return np.asarray(
                self.model_._pmf_predict(X, sensitive_features=np.zeros(len(X), dtype=int))[:, 1]
            )
        return self.scorer_.predict_proba(X)[:, 1]

    def result(self) -> MitigatorResult:
        self._check_fitted()
        return MitigatorResult(metadata=dict(getattr(self, "metadata", {})))
