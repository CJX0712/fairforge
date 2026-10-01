"""Disparate Impact Remover (simplified): align per-group feature means.

For each feature, shift each group's column toward the global mean, blended
by ``repair_level`` (0 = untouched, 1 = group means fully aligned).

Author: 晨星
"""

import numpy as np

from fairforge.core.types import Dataset
from fairforge.mitigation.base import Mitigator, MitigatorResult, register


@register("disparate_impact_remover")
class DisparateImpactRemover(Mitigator):
    def valid_params(self):
        return {"repair_level"}

    def __init__(self, repair_level: float = 1.0, **kw):
        super().__init__(**kw)
        if not (0.0 <= repair_level <= 1.0):
            from fairforge.core.errors import MitigatorError

            raise MitigatorError("repair_level must be in [0, 1]")
        self.params["repair_level"] = float(repair_level)

    def fit(self, dataset: Dataset) -> "DisparateImpactRemover":
        self.resolve_backend("fairlearn")  # native is the primary path here
        self.global_mean_ = dataset.X.mean(axis=0)
        self.group_mean_ = {
            g: dataset.X[dataset.A == g].mean(axis=0) for g in (0, 1) if (dataset.A == g).any()
        }
        self._fitted = True
        return self

    def transform(self, dataset: Dataset) -> MitigatorResult:
        self._check_fitted()
        repair = self.params["repair_level"]
        X_new = dataset.X.copy()
        for g, gm in self.group_mean_.items():
            mg = dataset.A == g
            X_new[mg] -= repair * (gm - self.global_mean_)
        return MitigatorResult(X=X_new, metadata={"repair_level": repair})

    def predict(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        from fairforge.core.errors import MitigatorError

        raise MitigatorError("disparate_impact_remover is pre-processing; use transform()")
