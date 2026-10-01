"""Exponentiated Gradient (reductions family).

Backend: fairlearn.reductions.ExponentiatedGradient(LogisticRegression(),
DemographicParity) when available. Fallback: a simplified native reductions
loop (EDUCATIONAL-GRADE, not a re-implementation of the full EG oracle
theory):
  1. fit a weighted logistic regression; group weights are exponentiated by
     the signed DP gap: ``w0 <- w0 * exp(eta * gap)``, mirrored for the
     privileged group (multiplier form of lambda <- lambda * exp(eta * v));
  2. per-group quantile threshold correction drives the group selection
     rates to a common target, then the best of {equal-rate, both eps-boundary
     orientations} is kept by accuracy - all satisfy violation <= eps.

A cross-check against the fairlearn backend is asserted softly in tests.

Author: 晨星
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from fairforge.core.types import Dataset
from fairforge.mitigation.base import Mitigator, MitigatorResult, register
from fairforge.mitigation.inproc.common import signed_gap

#: constraint -> fairlearn reductions Moment
FAIRLEARN_MOMENT = {
    "demographic_parity_diff": "DemographicParity",
    "equal_opportunity_diff": "TruePositiveRateParity",
}


@register("exponentiated_gradient")
class ExponentiatedGradient(Mitigator):
    def valid_params(self):
        return {"eps", "eta", "max_iter"}

    def __init__(self, eps: float = 0.05, eta: float = 0.8, max_iter: int = 25, **kw):
        super().__init__(eps=eps, eta=eta, max_iter=max_iter, **kw)

    def fit(self, dataset: Dataset) -> "ExponentiatedGradient":
        self.resolve_backend("fairlearn")
        if self.resolved_backend == "fairlearn":
            try:
                import fairlearn.reductions as flr

                eps = self.params["eps"]
                moment_name = FAIRLEARN_MOMENT.get(self.constraint, "DemographicParity")
                moment = getattr(flr, moment_name)(difference_bound=eps)
                est = flr.ExponentiatedGradient(
                    LogisticRegression(max_iter=1000, random_state=self.random_state),
                    constraints=moment,
                    eps=eps,
                    max_iter=self.params["max_iter"],
                )
                est.fit(dataset.X, dataset.y, sensitive_features=dataset.A)
                self.model_ = est
                self.violation_history_ = []
                self.thresholds_ = None
                self._fitted = True
                return self
            except Exception:
                self.used_fallback = True
                self.resolved_backend = "native"
        self._fit_native(dataset)
        return self

    # ------------------------------------------------------------------ native
    def _pred_with_thresholds(self, scores: np.ndarray, A: np.ndarray, thr: dict) -> np.ndarray:
        pred = np.empty(len(A), dtype=int)
        for g, t in thr.items():
            pred[A == g] = (scores[A == g] > t).astype(int)
        return pred

    def _rates_at_target(self, scores, A, y, target: float) -> dict:
        """Per-group thresholds driving the constrained quantity to ``target``.

        DP: quantiles over all group scores (selection rate target).
        EO: quantiles over group positives (TPR target).
        """
        thr = {}
        for g in (0, 1):
            mg = A == g
            if self.constraint == "equal_opportunity_diff":
                mg = mg & (y == 1)
            if mg.any():
                thr[g] = float(np.quantile(scores[mg], 1.0 - target))
        return thr

    def _accuracy(self, y, pred) -> float:
        return float(np.mean(np.asarray(pred) == y))

    def _fit_native(self, dataset: Dataset) -> None:
        eps = self.params["eps"]
        eta = self.params["eta"]
        max_iter = self.params["max_iter"]
        X, A, y = dataset.X, dataset.A, dataset.y
        constraint = self.constraint
        w = np.ones(len(A))
        self.models_ = []
        self.violation_history_ = []
        thr = {0: 0.5, 1: 0.5}
        for _ in range(max_iter):
            model = LogisticRegression(max_iter=1000, random_state=self.random_state)
            model.fit(X, y, sample_weight=w)
            scores = model.predict_proba(X)[:, 1]
            pred = self._pred_with_thresholds(scores, A, thr)
            viol = abs(float(signed_gap(constraint, y, pred, A)))
            gap = float(signed_gap(constraint, y, pred, A))
            self.violation_history_.append(viol)
            self.models_.append(model)

            if viol <= eps:
                self.model_, self.thresholds_ = model, dict(thr)
                break

            # (1) threshold correction: common target for the constrained quantity
            if constraint == "equal_opportunity_diff":
                t0 = (
                    float(np.mean(pred[(A == 0) & (y == 1)]))
                    if ((A == 0) & (y == 1)).any()
                    else 0.0
                )
                t1 = (
                    float(np.mean(pred[(A == 1) & (y == 1)]))
                    if ((A == 1) & (y == 1)).any()
                    else 0.0
                )
            else:
                t0 = float(np.mean(pred[A == 0])) if (A == 0).any() else 0.0
                t1 = float(np.mean(pred[A == 1])) if (A == 1).any() else 0.0
            q = float(np.clip((t0 + t1) / 2.0, 0.02, 0.98))
            candidates = [self._rates_at_target(scores, A, y, q)]
            # (2) eps-boundary orientations (both directions), still within eps
            candidates.append(
                self._rates_at_target(scores, A, y, float(np.clip(q + eps / 2, 0.02, 0.98)))
            )
            candidates.append(
                self._rates_at_target(scores, A, y, float(np.clip(q - eps / 2, 0.02, 0.98)))
            )
            best_thr, best_acc = None, -np.inf
            for cand in candidates:
                if len(cand) < 2:
                    continue
                p = self._pred_with_thresholds(scores, A, cand)
                if abs(float(signed_gap(constraint, y, p, A))) <= eps + 1e-9:
                    acc = self._accuracy(y, p)
                    if acc > best_acc:
                        best_thr, best_acc = cand, acc
            if best_thr is None:
                best_thr = candidates[0]
            thr = best_thr

            # (3) exponentiate the group weights by the signed gap -> gap shrinks
            w = w.copy()
            w[A == 0] *= np.exp(eta * gap)
            w[A == 1] *= np.exp(-eta * gap)
            w = np.clip(w, 1e-3, 1e3)
        else:
            self.model_, self.thresholds_ = self.models_[-1], thr
        self._fitted = True

    # ------------------------------------------------------------------ api
    def predict(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        A = np.asarray(A).astype(int).ravel()
        if self.resolved_backend == "fairlearn":
            return np.asarray(self.model_.predict(X), dtype=int)
        scores = self.model_.predict_proba(X)[:, 1]
        return self._pred_with_thresholds(scores, A, self.thresholds_)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model_.predict_proba(np.asarray(X, dtype=float))[:, 1]

    def result(self) -> MitigatorResult:
        self._check_fitted()
        self.metadata = {
            "violation_history": list(self.violation_history_),
            "n_inner_models": len(getattr(self, "models_", [])) or 1,
            "thresholds": dict(getattr(self, "thresholds_", {}) or {}),
        }
        return MitigatorResult(metadata=dict(self.metadata))
