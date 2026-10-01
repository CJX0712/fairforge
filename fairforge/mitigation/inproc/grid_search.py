"""Grid Search (reductions family).

Backend: fairlearn.reductions.GridSearch when available (constraint-aware:
DemographicParity / TruePositiveRateParity). Fallback: fixed lambda grid over
group-reweighting strengths; each candidate is a weighted logistic regression,
the winner maximizes ``accuracy - lambda_fair * violation`` under the
requested constraint.

Author: 晨星
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from fairforge.core.types import Dataset
from fairforge.metrics.fairness import demographic_parity_diff, equal_opportunity_diff
from fairforge.metrics.performance import accuracy as accuracy_metric
from fairforge.mitigation.base import Mitigator, MitigatorResult, register

FAIRLEARN_MOMENT = {
    "demographic_parity_diff": "DemographicParity",
    "equal_opportunity_diff": "TruePositiveRateParity",
}


def _violation(constraint: str, y, pred, A) -> float:
    if constraint == "equal_opportunity_diff":
        return equal_opportunity_diff(y, pred, A)
    return demographic_parity_diff(pred, A)


@register("grid_search")
class GridSearchReductions(Mitigator):
    def valid_params(self):
        return {"eps", "lambda_fair", "grid_size"}

    def __init__(self, eps: float = 0.05, lambda_fair: float = 1.0, grid_size: int = 6, **kw):
        super().__init__(eps=eps, lambda_fair=lambda_fair, grid_size=grid_size, **kw)

    def fit(self, dataset: Dataset) -> "GridSearchReductions":
        self.resolve_backend("fairlearn")
        if self.resolved_backend == "fairlearn":
            try:
                import fairlearn.reductions as flr

                moment_name = FAIRLEARN_MOMENT.get(self.constraint, "DemographicParity")
                est = flr.GridSearch(
                    LogisticRegression(max_iter=1000, random_state=self.random_state),
                    constraints=getattr(flr, moment_name)(difference_bound=self.params["eps"]),
                    grid_size=self.params["grid_size"],
                )
                est.fit(dataset.X, dataset.y, sensitive_features=dataset.A)
                self.model_ = est
                self._fitted = True
                return self
            except Exception:
                self.used_fallback = True
                self.resolved_backend = "native"
        self._fit_native(dataset)
        return self

    def _fit_native(self, dataset: Dataset) -> None:
        X, A, y = dataset.X, dataset.A, dataset.y
        lambda_fair = self.params["lambda_fair"]
        constraint = self.constraint
        grid = np.linspace(0.0, 8.0, int(self.params["grid_size"]))
        best = None
        self.candidates_ = []
        for lam in grid:
            w = np.where(A == 0, 1.0 + lam, 1.0 - 0.1 * lam)
            w = np.clip(w, 1e-3, None)
            model = LogisticRegression(max_iter=1000, random_state=self.random_state)
            model.fit(X, y, sample_weight=w)
            pred = model.predict(X).astype(int)
            acc = accuracy_metric(y, pred)
            viol = _violation(constraint, y, pred, A)
            obj = acc - lambda_fair * viol
            self.candidates_.append({"lambda": float(lam), "accuracy": acc, "violation": viol})
            if best is None or obj > best[0]:
                best = (obj, model, float(lam))
        self.model_ = best[1]
        self.metadata = {"best_lambda": best[2], "n_candidates": len(self.candidates_)}
        self._fitted = True

    def predict(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.asarray(self.model_.predict(np.asarray(X, dtype=float)), dtype=int)

    def result(self) -> MitigatorResult:
        self._check_fitted()
        return MitigatorResult(metadata=dict(getattr(self, "metadata", {})))
