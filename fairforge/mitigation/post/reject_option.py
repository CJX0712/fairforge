"""Reject Option Classification (native only).

Confidence band ``[0.5 - theta, 0.5 + theta]``: predictions inside the band
that are *unfavorable for the unprivileged group* are flipped to favorable.
Theta is grid-searched to maximize ``accuracy - lambda_fair * violation``.

Author: 晨星
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from fairforge.core.types import Dataset
from fairforge.metrics.fairness import demographic_parity_diff, equal_opportunity_diff
from fairforge.metrics.performance import accuracy as accuracy_metric
from fairforge.mitigation.base import Mitigator, MitigatorResult, register


def _violation(constraint: str, y, pred, A) -> float:
    if constraint == "equal_opportunity_diff":
        return equal_opportunity_diff(y, pred, A)
    return demographic_parity_diff(pred, A)


@register("reject_option")
class RejectOption(Mitigator):
    def valid_params(self):
        return {"lambda_fair", "theta_steps"}

    def __init__(self, lambda_fair: float = 1.0, theta_steps: int = 10, **kw):
        super().__init__(lambda_fair=lambda_fair, theta_steps=theta_steps, **kw)

    def fit(self, dataset: Dataset) -> "RejectOption":
        self.resolve_backend("fairlearn")  # always native; declared for uniformity
        X, A, y = dataset.X, dataset.A, dataset.y
        lambda_fair = self.params["lambda_fair"]
        self.scorer_ = LogisticRegression(max_iter=1000, random_state=self.random_state).fit(X, y)
        scores = self.scorer_.predict_proba(X)[:, 1]
        base_pred = (scores > 0.5).astype(int)

        best = None
        history = []
        for theta in np.linspace(0.0, 0.5, int(self.params["theta_steps"]) + 1):
            pred = base_pred.copy()
            in_band = np.abs(scores - 0.5) <= theta
            # flip: unprivileged & predicted unfavorable -> favorable
            flip = in_band & (A == 0) & (pred == 0)
            pred[flip] = 1
            acc = accuracy_metric(y, pred)
            viol = _violation(self.constraint, y, pred, A)
            obj = acc - lambda_fair * viol
            history.append({"theta": float(theta), "accuracy": acc, "violation": viol})
            if best is None or obj > best[0]:
                best = (obj, float(theta))
        self.theta_ = best[1]
        self.metadata = {"theta": self.theta_, "history": history}
        self._fitted = True
        return self

    def predict(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        self._check_fitted()
        A = np.asarray(A).astype(int).ravel()
        scores = self.scorer_.predict_proba(np.asarray(X, dtype=float))[:, 1]
        pred = (scores > 0.5).astype(int)
        in_band = np.abs(scores - 0.5) <= self.theta_
        pred[in_band & (A == 0) & (pred == 0)] = 1
        return pred

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.scorer_.predict_proba(np.asarray(X, dtype=float))[:, 1]

    def result(self) -> MitigatorResult:
        self._check_fitted()
        return MitigatorResult(metadata={"theta": self.theta_})
