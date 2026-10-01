"""No-mitigation passthrough: plain logistic regression baseline candidate.

A legitimate member of the flagship portfolio - it anchors the high-accuracy
end of the Pareto front.

Author: 晨星
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from fairforge.core.types import Dataset
from fairforge.mitigation.base import Mitigator, MitigatorResult, register


@register("none")
class NoMitigation(Mitigator):
    """Plain logistic regression, no fairness intervention."""

    def valid_params(self):
        return set()

    def fit(self, dataset: Dataset) -> "NoMitigation":
        self.resolve_backend("fairlearn")  # no optional backend involved; uniform bookkeeping
        self.model_ = LogisticRegression(max_iter=1000, random_state=self.random_state).fit(
            dataset.X, dataset.y
        )
        self._fitted = True
        return self

    def predict(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.asarray(self.model_.predict(np.asarray(X, dtype=float)), dtype=int)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model_.predict_proba(np.asarray(X, dtype=float))[:, 1]

    def result(self) -> MitigatorResult:
        self._check_fitted()
        return MitigatorResult(metadata={"mitigation": "none"})
