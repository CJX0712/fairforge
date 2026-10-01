"""Reweighing (Kamiran & Calders 2012) - the native workhorse pre-processing method.

Weight for joint group/label cell: ``w(A=a, Y=y) = P(A=a) * P(Y=y) / P(A=a, Y=y)``.
Sum of weights over all samples is exactly n.

Author: 晨星
"""

import numpy as np

from fairforge.core.errors import MitigatorError
from fairforge.core.types import Dataset
from fairforge.mitigation.base import Mitigator, MitigatorResult, register


@register("reweighing")
class Reweighing(Mitigator):
    def valid_params(self):
        return set()

    def fit(self, dataset: Dataset) -> "Reweighing":
        self.resolve_backend("fairlearn")  # native is the primary path here
        A, y = dataset.A, dataset.y
        n = len(A)
        p_a = {g: float((A == g).sum()) / n for g in (0, 1)}
        p_y = {v: float((y == v).sum()) / n for v in (0, 1)}
        weights: dict[tuple[int, int], float] = {}
        for g in (0, 1):
            for v in (0, 1):
                n_ay = float(((A == g) & (y == v)).sum())
                if n_ay == 0:
                    raise MitigatorError(
                        f"reweighing: empty joint cell A={g}, Y={v}; need both groups and labels"
                    )
                weights[(g, v)] = p_a[g] * p_y[v] / (n_ay / n)
        self.weights_table_ = weights
        self._fitted = True
        return self

    def transform(self, dataset: Dataset) -> MitigatorResult:
        self._check_fitted()
        pairs = zip(dataset.A, dataset.y, strict=True)
        w = np.array([self.weights_table_[(g, v)] for g, v in pairs])
        return MitigatorResult(
            sample_weight=w, metadata={"weights_table": dict(self.weights_table_)}
        )

    def predict(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        raise MitigatorError("reweighing is pre-processing; use transform() + your own model")
