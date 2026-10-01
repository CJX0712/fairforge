"""Group Threshold: per-group independent threshold search (native only).

Constraint-aware: DP mode matches each group's selection rate to the global
rate; EO mode matches each group's TPR (on fit labels) to the global TPR.
A cheap post-processing that never touches the model.

Author: 晨星
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from fairforge.core.types import Dataset
from fairforge.mitigation.base import Mitigator, MitigatorResult, register


@register("group_threshold")
class GroupThreshold(Mitigator):
    def valid_params(self):
        return {"grid_size"}

    def __init__(self, grid_size: int = 20, **kw):
        super().__init__(grid_size=grid_size, **kw)

    def fit(self, dataset: Dataset) -> "GroupThreshold":
        self.resolve_backend("fairlearn")  # always native; declared for uniformity
        grid = np.linspace(0.02, 0.98, int(self.params["grid_size"]))
        X, A, y = dataset.X, dataset.A, dataset.y
        self.scorer_ = LogisticRegression(max_iter=1000, random_state=self.random_state).fit(X, y)
        scores = self.scorer_.predict_proba(X)[:, 1]
        if self.constraint == "equal_opportunity_diff":
            # EO: match each group's TPR to the global TPR
            base_pred = (scores > 0.5).astype(int)
            target = float(np.mean(base_pred[y == 1]))
            self.thresholds_ = {}
            for g in (0, 1):
                mg = (A == g) & (y == 1)
                if not mg.any():
                    continue
                sg = scores[mg]
                best_t, best_diff = 0.5, np.inf
                for t in grid:
                    diff = abs(float(np.mean(sg > t)) - target)
                    if diff < best_diff:
                        best_diff, best_t = diff, float(t)
                self.thresholds_[g] = best_t
        else:
            # DP: match each group's selection rate to the global rate
            target = float(np.mean(scores > 0.5))  # global selection rate at the default cut
            self.thresholds_ = {}
            for g in (0, 1):
                mg = A == g
                if not mg.any():
                    continue
                sg = scores[mg]
                best_t, best_diff = 0.5, np.inf
                for t in grid:
                    diff = abs(float(np.mean(sg > t)) - target)
                    if diff < best_diff:
                        best_diff, best_t = diff, float(t)
                self.thresholds_[g] = best_t
        self.metadata = {
            "target": target,
            "constraint": self.constraint,
            "thresholds": dict(self.thresholds_),
        }
        self._fitted = True
        return self

    def predict(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        self._check_fitted()
        A = np.asarray(A).astype(int).ravel()
        scores = self.scorer_.predict_proba(np.asarray(X, dtype=float))[:, 1]
        pred = np.empty(len(A), dtype=int)
        for g, t in self.thresholds_.items():
            pred[A == g] = (scores[A == g] > t).astype(int)
        return pred

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.scorer_.predict_proba(np.asarray(X, dtype=float))[:, 1]

    def result(self) -> MitigatorResult:
        self._check_fitted()
        return MitigatorResult(metadata=dict(getattr(self, "metadata", {})))
